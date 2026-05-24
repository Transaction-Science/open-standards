//! [`NativeCalculator`] — the bundled in-process tax engine.
//!
//! Uses a [`RateTable`] and does its own compounding. No external
//! dependency, no network call. Operators on small / single-
//! jurisdiction deployments can run the whole tax surface from this
//! one struct; everyone else swaps it for one of the commercial
//! adapters.
//!
//! ## Compounding rules
//!
//! - **VAT / GST** are *replace-style*. We pick exactly one layer —
//!   the most specific `Vat` or `Gst` entry in the ship-to jurisdiction
//!   chain — and apply it once. EU member states levy VAT at the
//!   country level; subnational layers (e.g. German *Länder*, French
//!   *régions*) do not have independent VAT authority.
//! - **Sales / Use** are *additive*. We walk the ship-to chain from
//!   broadest to narrowest and sum every rate we find. US state +
//!   county + city + special district stacks this way.
//! - **Excise** is additive on top of either of the above. Most
//!   alcohol / tobacco / fuel taxes appear here.
//! - **Import duty** is additive at country level when
//!   `ship_from.country != ship_to.country`.
//!
//! ## Inclusive vs exclusive math
//!
//! For an exclusive base (US sales tax): `tax = amount * rate`.
//!
//! For an inclusive base (EU VAT consumer price): the line `amount`
//! already contains the tax. We back it out:
//!   `net = amount / (1 + rate)`
//!   `tax = amount - net`
//!
//! Either way the per-jurisdiction `taxable_amount` reported on the
//! breakdown is the *net* amount the rate was applied to, so
//! downstream reconciliation always sees consistent numbers.

use async_trait::async_trait;
use chrono::Utc;
use op_core::Money;
use rust_decimal::Decimal;
use rust_decimal::prelude::ToPrimitive;
use std::collections::BTreeMap;

use crate::calculator::{
    CustomerType, JurisdictionTax, LineTaxBreakdown, TaxCalculator, TaxContext, TaxResult,
    TaxableLine,
};
use crate::error::{Error, Result};
use crate::exemption::{self, ExemptionCertificate};
use crate::jurisdiction::Jurisdiction;
use crate::rate_table::{RateKind, RateTable, TaxBase, TaxRate};

/// In-process tax calculator backed by a [`RateTable`].
pub struct NativeCalculator {
    /// The rate data. Cloned on construction; the calculator owns
    /// its table so swapping in a new snapshot is a new calculator.
    pub table: RateTable,
}

impl NativeCalculator {
    /// Construct from a rate table.
    #[must_use]
    pub const fn new(table: RateTable) -> Self {
        Self { table }
    }

    /// Construct using the bundled starter table. Convenience for
    /// tests and small deployments.
    #[must_use]
    pub fn bundled() -> Self {
        Self::new(RateTable::bundled())
    }

    /// Decide whether the buyer qualifies for EU reverse-charge on
    /// this line: cross-border B2B inside the EU.
    fn is_reverse_charge(line: &TaxableLine, ctx: &TaxContext) -> bool {
        let CustomerType::Business { tax_id } = &ctx.customer_type else {
            return false;
        };
        if tax_id.is_empty() {
            return false;
        }
        let to_eu = line.ship_to.is_eu_member();
        let Some(from) = &line.ship_from else {
            return false;
        };
        let from_eu = from.is_eu_member();
        // Cross-border (different countries) AND both in EU.
        to_eu && from_eu && from.country != line.ship_to.country
    }

    /// Find the first exemption certificate that applies to this line.
    fn matching_exemption<'c>(
        line: &TaxableLine,
        ctx: &'c TaxContext,
    ) -> Option<&'c ExemptionCertificate> {
        ctx.exemption_certs
            .iter()
            .find(|c| exemption::applies(c, line, ctx))
    }

    /// Apply any per-category caps / floors that aren't expressible
    /// as a plain rate. Currently:
    ///
    /// - NY clothing exemption: only items under $110 qualify. Above
    ///   the cap, fall back to the NY-state TangibleGoods rate.
    ///
    /// Returns a list of `(jurisdiction, TaxRate)` overrides to use in
    /// place of the natural lookup at that jurisdiction. An empty
    /// vector means "no overrides — use the table as-is."
    fn category_overrides(
        &self,
        line: &TaxableLine,
    ) -> Vec<(Jurisdiction, TaxRate)> {
        let mut overrides = Vec::new();
        if line.category == crate::category::ProductTaxCategory::Clothing {
            let ny = Jurisdiction::region("US", "NY");
            // $110 cap = 11_000 minor units USD.
            if line.ship_to.ancestors().contains(&ny)
                && line.amount.minor_units >= 11_000
                && let Some(rate) = self
                    .table
                    .lookup(&ny, &crate::category::ProductTaxCategory::TangibleGoods)
            {
                overrides.push((ny, rate.clone()));
            }
        }
        overrides
    }

    /// Compute the tax breakdown for one line.
    fn calc_line(&self, line: &TaxableLine, ctx: &TaxContext) -> Result<LineTaxBreakdown> {
        // 1. Nexus filter. If the seller has not declared nexus
        //    *anywhere*, we skip the filter entirely (operator opt-
        //    out for development / freight-from-elsewhere flows).
        //    Otherwise any line shipping to a non-nexused jurisdiction
        //    is zero-rated.
        if !ctx.nexus_jurisdictions.is_empty()
            && !ctx
                .nexus_jurisdictions
                .iter()
                .any(|n| line.ship_to.ancestors().contains(n))
        {
            return Ok(zero_line(
                &line.line_id,
                line.amount,
                Some("no nexus in destination jurisdiction"),
            ));
        }

        // 2. Reverse charge — EU B2B cross-border zero-rates at PoS.
        if Self::is_reverse_charge(line, ctx) {
            return Ok(zero_line(
                &line.line_id,
                line.amount,
                Some("EU B2B reverse charge"),
            ));
        }

        // 3. Exemption certificate.
        if let Some(cert) = Self::matching_exemption(line, ctx) {
            return Ok(zero_line(
                &line.line_id,
                line.amount,
                Some(&format!("exemption certificate {}", cert.id)),
            ));
        }

        // 4. Compound the rate stack.
        let overrides = self.category_overrides(line);
        let mut layers: Vec<(Jurisdiction, TaxRate)> = Vec::new();
        let mut have_vat_or_gst = false;

        for j in line.ship_to.ancestors() {
            // Check override first.
            if let Some((_, r)) = overrides.iter().find(|(oj, _)| oj == &j) {
                push_layer(&mut layers, j, r.clone(), &mut have_vat_or_gst);
                continue;
            }
            let Some(rate) = self.table.lookup(&j, &line.category) else {
                continue;
            };
            push_layer(&mut layers, j, rate.clone(), &mut have_vat_or_gst);
        }

        // 5. Import duty — additive country-level when crossing borders.
        if let Some(from) = &line.ship_from
            && from.country != line.ship_to.country
        {
            let country_j = Jurisdiction::country(&line.ship_to.country.0);
            if let Some(r) = self
                .table
                .lookup(&country_j, &line.category)
                .filter(|r| r.kind == RateKind::ImportDuty)
            {
                layers.push((country_j, r.clone()));
            }
        }

        if layers.is_empty() {
            return Err(Error::NoRate {
                jurisdiction: line.ship_to.to_string(),
                category: line.category.tag().to_owned(),
            });
        }

        // 6. Determine inclusive / exclusive base. If any layer is
        //    inclusive (VAT/GST), the line amount is treated as gross
        //    and we back the tax out. Otherwise (US sales tax), we
        //    apply each rate to the line net amount.
        let any_inclusive = layers.iter().any(|(_, r)| r.base == TaxBase::Inclusive);
        let line_amount_dec = decimal_from_minor(line.amount);

        let mut per_jurisdiction = Vec::with_capacity(layers.len());
        let mut effective_rate = Decimal::ZERO;
        let mut total_tax_dec = Decimal::ZERO;
        let net_amount_dec;

        if any_inclusive {
            // VAT/GST inclusive — sum the inclusive layers' rates and
            // back out tax in a single pass. Per the VAT directive, only
            // one VAT/GST rate applies; any additional layers are excise
            // on top of net.
            let inclusive_rate: Decimal = layers
                .iter()
                .filter(|(_, r)| r.base == TaxBase::Inclusive)
                .map(|(_, r)| r.rate)
                .sum();
            let one = Decimal::ONE;
            let net = line_amount_dec
                .checked_div(one + inclusive_rate)
                .ok_or(Error::Overflow)?;
            let inclusive_tax = line_amount_dec - net;
            net_amount_dec = net;

            // Allocate inclusive tax proportionally across inclusive layers
            // (typically just one entry — VAT or GST).
            for (j, r) in &layers {
                if r.base != TaxBase::Inclusive {
                    continue;
                }
                let layer_share = if inclusive_rate.is_zero() {
                    Decimal::ZERO
                } else {
                    inclusive_tax * (r.rate / inclusive_rate)
                };
                let layer_minor = decimal_to_minor_round_half_up(layer_share)?;
                per_jurisdiction.push(JurisdictionTax {
                    jurisdiction: j.clone(),
                    rate: r.rate,
                    amount: Money::from_minor(layer_minor, line.amount.currency),
                    kind: r.kind,
                });
                effective_rate += r.rate;
                total_tax_dec += layer_share;
            }
            // Additive (excise / import duty) layers stack on top of NET.
            for (j, r) in &layers {
                if r.base == TaxBase::Inclusive {
                    continue;
                }
                let layer_tax = net * r.rate;
                let layer_minor = decimal_to_minor_round_half_up(layer_tax)?;
                per_jurisdiction.push(JurisdictionTax {
                    jurisdiction: j.clone(),
                    rate: r.rate,
                    amount: Money::from_minor(layer_minor, line.amount.currency),
                    kind: r.kind,
                });
                effective_rate += r.rate;
                total_tax_dec += layer_tax;
            }
        } else {
            // US sales-tax style — all layers additive, exclusive base.
            net_amount_dec = line_amount_dec;
            for (j, r) in &layers {
                let layer_tax = line_amount_dec * r.rate;
                let layer_minor = decimal_to_minor_round_half_up(layer_tax)?;
                per_jurisdiction.push(JurisdictionTax {
                    jurisdiction: j.clone(),
                    rate: r.rate,
                    amount: Money::from_minor(layer_minor, line.amount.currency),
                    kind: r.kind,
                });
                effective_rate += r.rate;
                total_tax_dec += layer_tax;
            }
        }

        let total_tax_minor = decimal_to_minor_round_half_up(total_tax_dec)?;
        let net_minor = decimal_to_minor_round_half_up(net_amount_dec)?;

        Ok(LineTaxBreakdown {
            line_id: line.line_id.clone(),
            taxable_amount: Money::from_minor(net_minor, line.amount.currency),
            tax_amount: Money::from_minor(total_tax_minor, line.amount.currency),
            effective_rate,
            jurisdiction_layers: per_jurisdiction,
            exemption_reason: None,
        })
    }
}

fn push_layer(
    layers: &mut Vec<(Jurisdiction, TaxRate)>,
    j: Jurisdiction,
    r: TaxRate,
    have_vat_or_gst: &mut bool,
) {
    if matches!(r.kind, RateKind::Vat | RateKind::Gst) {
        if *have_vat_or_gst {
            // VAT/GST is replace-style: a more-specific layer would
            // overwrite the broader one. Since we walk broadest-first,
            // we replace the existing VAT layer with this narrower one.
            layers.retain(|(_, existing)| {
                !matches!(existing.kind, RateKind::Vat | RateKind::Gst)
            });
        }
        *have_vat_or_gst = true;
    }
    layers.push((j, r));
}

fn zero_line(
    line_id: &str,
    amount: Money,
    reason: Option<&str>,
) -> LineTaxBreakdown {
    LineTaxBreakdown {
        line_id: line_id.to_owned(),
        taxable_amount: amount,
        tax_amount: Money::from_minor(0, amount.currency),
        effective_rate: Decimal::ZERO,
        jurisdiction_layers: Vec::new(),
        exemption_reason: reason.map(str::to_owned),
    }
}

fn decimal_from_minor(m: Money) -> Decimal {
    Decimal::from(m.minor_units)
}

fn decimal_to_minor_round_half_up(d: Decimal) -> Result<i64> {
    let rounded = d.round_dp_with_strategy(0, rust_decimal::RoundingStrategy::MidpointAwayFromZero);
    rounded.to_i64().ok_or(Error::Overflow)
}

#[async_trait]
impl TaxCalculator for NativeCalculator {
    async fn calculate(&self, lines: &[TaxableLine], ctx: &TaxContext) -> Result<TaxResult> {
        if lines.is_empty() {
            return Err(Error::Config("calculate called with zero lines".into()));
        }
        let currency = lines[0].amount.currency;
        let mut total_tax = Money::from_minor(0, currency);
        let mut per_line: BTreeMap<String, LineTaxBreakdown> = BTreeMap::new();
        let mut by_jurisdiction: BTreeMap<(Jurisdiction, RateKind), JurisdictionTax> =
            BTreeMap::new();

        for line in lines {
            if line.amount.currency != currency {
                return Err(Error::Money(op_core::Error::CurrencyMismatch));
            }
            let bd = self.calc_line(line, ctx)?;
            total_tax = total_tax.checked_add(bd.tax_amount)?;
            for layer in &bd.jurisdiction_layers {
                let key = (layer.jurisdiction.clone(), layer.kind);
                by_jurisdiction
                    .entry(key)
                    .and_modify(|j| {
                        if let Ok(sum) = j.amount.checked_add(layer.amount) {
                            j.amount = sum;
                        }
                    })
                    .or_insert_with(|| layer.clone());
            }
            per_line.insert(bd.line_id.clone(), bd);
        }

        let jurisdictions_charged = by_jurisdiction.into_values().collect();

        Ok(TaxResult {
            total_tax,
            per_line,
            jurisdictions_charged,
            calculator: "native".to_owned(),
            calculated_at: Utc::now(),
        })
    }

    fn name(&self) -> &'static str {
        "native"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::category::ProductTaxCategory;
    use chrono::NaiveDate;
    use op_core::Currency;
    use std::collections::HashSet;

    fn ctx_consumer() -> TaxContext {
        TaxContext {
            transaction_date: NaiveDate::parse_from_str("2026-06-15", "%Y-%m-%d").unwrap(),
            customer_type: CustomerType::Consumer,
            exemption_certs: vec![],
            nexus_jurisdictions: HashSet::new(),
        }
    }

    #[tokio::test]
    async fn us_sales_tax_compounds_state_city_district() {
        let calc = NativeCalculator::bundled();
        let line = TaxableLine {
            line_id: "L1".into(),
            // $100.00
            amount: Money::from_minor(10_000, Currency::USD),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::full("US", "WA", "Seattle", "KingTransit"),
        };
        let r = calc.calculate(&[line], &ctx_consumer()).await.unwrap();
        // WA 6.5 + Seattle 3.85 + KingTransit 0.9 = 11.25% on $100 = $11.25
        assert_eq!(r.total_tax.minor_units, 1125);
        let bd = r.per_line.get("L1").unwrap();
        assert_eq!(bd.jurisdiction_layers.len(), 3);
    }

    #[tokio::test]
    async fn eu_vat_inclusive_backs_out_correctly() {
        let calc = NativeCalculator::bundled();
        // €100.00 gross with 20% French VAT.
        // Net = 100/1.20 = 83.333...; tax = 16.666... ; rounded = 16.67
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::EUR),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::country("FR"),
        };
        let r = calc.calculate(&[line], &ctx_consumer()).await.unwrap();
        assert_eq!(r.total_tax.minor_units, 1667);
        let bd = r.per_line.get("L1").unwrap();
        assert_eq!(bd.taxable_amount.minor_units, 8333);
        assert_eq!(bd.jurisdiction_layers.len(), 1);
        assert_eq!(bd.jurisdiction_layers[0].kind, RateKind::Vat);
    }

    #[tokio::test]
    async fn vat_replace_not_additive() {
        // Even if a region-level entry existed, VAT only compounds once.
        let mut t = RateTable::bundled();
        t = t.with(
            Jurisdiction::region("FR", "75"),
            ProductTaxCategory::TangibleGoods,
            TaxRate::vat(Decimal::new(2500, 4)), // hypothetical Paris VAT 25%
        );
        let calc = NativeCalculator::new(t);
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::EUR),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::locality("FR", "75", "Paris"),
        };
        let r = calc.calculate(&[line], &ctx_consumer()).await.unwrap();
        // Should be exactly ONE VAT layer (Paris 25%, replacing FR 20%).
        let bd = r.per_line.get("L1").unwrap();
        let vat_layers = bd
            .jurisdiction_layers
            .iter()
            .filter(|l| l.kind == RateKind::Vat)
            .count();
        assert_eq!(vat_layers, 1);
        // Tax = 100 - 100/1.25 = 20.00
        assert_eq!(r.total_tax.minor_units, 2000);
    }

    #[tokio::test]
    async fn nexus_filter_zero_rates_outside_nexus() {
        let calc = NativeCalculator::bundled();
        let mut ctx = ctx_consumer();
        // Seller has nexus only in WA. NY shipment should be zero-rated.
        ctx.nexus_jurisdictions.insert(Jurisdiction::region("US", "WA"));
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::USD),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::region("US", "NY"),
        };
        let r = calc.calculate(&[line], &ctx).await.unwrap();
        assert_eq!(r.total_tax.minor_units, 0);
        let bd = r.per_line.get("L1").unwrap();
        assert!(bd.exemption_reason.as_deref().unwrap().contains("nexus"));
    }

    #[tokio::test]
    async fn ny_clothing_under_110_exempt() {
        let calc = NativeCalculator::bundled();
        // $50 shirt to NY — should be zero-rated (state + city override).
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(5_000, Currency::USD),
            category: ProductTaxCategory::Clothing,
            ship_from: None,
            ship_to: Jurisdiction::region("US", "NY"),
        };
        let r = calc.calculate(&[line], &ctx_consumer()).await.unwrap();
        // Only the NY-state override (0%) applies — no city in this jurisdiction.
        assert_eq!(r.total_tax.minor_units, 0);
    }

    #[tokio::test]
    async fn ny_clothing_over_110_taxed() {
        let calc = NativeCalculator::bundled();
        // $200 jacket to NY state. Override kicks back to TangibleGoods rate (4%).
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(20_000, Currency::USD),
            category: ProductTaxCategory::Clothing,
            ship_from: None,
            ship_to: Jurisdiction::region("US", "NY"),
        };
        let r = calc.calculate(&[line], &ctx_consumer()).await.unwrap();
        // 4% of $200 = $8.00
        assert_eq!(r.total_tax.minor_units, 800);
    }

    #[tokio::test]
    async fn returns_no_rate_for_unknown_jurisdiction() {
        let calc = NativeCalculator::bundled();
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::USD),
            category: ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::country("XX"), // unknown country
        };
        assert!(matches!(
            calc.calculate(&[line], &ctx_consumer()).await,
            Err(Error::NoRate { .. })
        ));
    }
}
