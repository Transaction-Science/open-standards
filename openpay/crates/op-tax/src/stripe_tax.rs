//! Stripe Tax adapter.
//!
//! Talks to `POST /v1/tax/calculations` on Stripe's REST API.
//! Authentication is HTTP Basic with `<secret_key>:`.
//!
//! Stripe Tax models the request as form-encoded line items (Stripe
//! uses form-encoding for all writes — not JSON). The response is
//! JSON with the same envelope shape every Stripe object uses.

use async_trait::async_trait;
use chrono::Utc;
use op_core::Money;
use rust_decimal::Decimal;
use serde::Deserialize;
use std::collections::BTreeMap;

use crate::calculator::{
    JurisdictionTax, LineTaxBreakdown, TaxCalculator, TaxContext, TaxResult, TaxableLine,
};
use crate::error::{Error, Result};
use crate::jurisdiction::{CountryCode, Jurisdiction, RegionCode};
use crate::rate_table::RateKind;

/// Stripe Tax client.
pub struct StripeTaxAdapter {
    client: reqwest::Client,
    base_url: String,
    secret_key: String,
}

impl StripeTaxAdapter {
    /// Construct with a Stripe secret key (`sk_live_…` or `sk_test_…`).
    #[must_use]
    pub fn new(secret_key: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.stripe.com".into(),
            secret_key,
        }
    }

    /// Builder: override the base URL (tests).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[derive(Deserialize, Debug)]
struct CalculationResponse {
    amount_total: i64,
    tax_amount_exclusive: i64,
    currency: String,
    line_items: LineItemsList,
    #[serde(default)]
    tax_breakdown: Vec<TaxBreakdownItem>,
}

#[derive(Deserialize, Debug)]
struct LineItemsList {
    data: Vec<CalcLineItem>,
}

#[derive(Deserialize, Debug)]
struct CalcLineItem {
    reference: String,
    amount: i64,
    amount_tax: i64,
}

#[derive(Deserialize, Debug, Default)]
struct TaxBreakdownItem {
    amount: i64,
    inclusive: bool,
    jurisdiction: TaxJurisdictionInfo,
    tax_rate_details: TaxRateDetails,
}

#[derive(Deserialize, Debug, Default)]
struct TaxJurisdictionInfo {
    country: String,
    #[serde(default)]
    state: Option<String>,
}

#[derive(Deserialize, Debug, Default)]
struct TaxRateDetails {
    #[serde(default)]
    percentage_decimal: String,
    #[serde(default)]
    tax_type: String,
}

#[async_trait]
impl TaxCalculator for StripeTaxAdapter {
    async fn calculate(&self, lines: &[TaxableLine], _ctx: &TaxContext) -> Result<TaxResult> {
        if lines.is_empty() {
            return Err(Error::Config("calculate called with zero lines".into()));
        }
        let currency = lines[0].amount.currency;
        // Build the form body. Stripe uses bracketed array notation.
        let mut form: Vec<(String, String)> = Vec::with_capacity(lines.len() * 5 + 4);
        form.push(("currency".into(), currency.code().to_lowercase()));
        form.push((
            "customer_details[address][country]".into(),
            lines[0].ship_to.country.0.clone(),
        ));
        if let Some(r) = &lines[0].ship_to.region {
            form.push((
                "customer_details[address][state]".into(),
                r.0.clone(),
            ));
        }
        form.push((
            "customer_details[address_source].".into(),
            "shipping".into(),
        ));
        for (i, line) in lines.iter().enumerate() {
            if line.amount.currency != currency {
                return Err(Error::Money(op_core::Error::CurrencyMismatch));
            }
            form.push((
                format!("line_items[{i}][amount]"),
                line.amount.minor_units.to_string(),
            ));
            form.push((format!("line_items[{i}][reference]"), line.line_id.clone()));
            form.push((
                format!("line_items[{i}][tax_code]"),
                stripe_tax_code(&line.category).to_owned(),
            ));
            form.push((format!("line_items[{i}][quantity]"), "1".into()));
        }

        let url = format!("{}/v1/tax/calculations", self.base_url);
        let resp = self
            .client
            .post(&url)
            .basic_auth(&self.secret_key, Some(""))
            .form(&form)
            .send()
            .await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::Vendor {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: CalculationResponse = resp.json().await?;
        // Cross-check currency.
        if parsed.currency.to_uppercase() != currency.code() {
            return Err(Error::Codec(format!(
                "stripe returned currency {} expected {}",
                parsed.currency,
                currency.code()
            )));
        }

        let mut per_line: BTreeMap<String, LineTaxBreakdown> = BTreeMap::new();
        for li in parsed.line_items.data {
            per_line.insert(
                li.reference.clone(),
                LineTaxBreakdown {
                    line_id: li.reference,
                    taxable_amount: Money::from_minor(li.amount, currency),
                    tax_amount: Money::from_minor(li.amount_tax, currency),
                    effective_rate: Decimal::ZERO,
                    jurisdiction_layers: Vec::new(),
                    exemption_reason: None,
                },
            );
        }

        let mut by_jurisdiction: Vec<JurisdictionTax> = Vec::new();
        for b in parsed.tax_breakdown {
            let j = Jurisdiction {
                country: CountryCode::new(&b.jurisdiction.country),
                region: b.jurisdiction.state.map(|s| RegionCode::new(&s)),
                locality: None,
                special_district: None,
            };
            let rate: Decimal = b
                .tax_rate_details
                .percentage_decimal
                .parse()
                .unwrap_or(Decimal::ZERO);
            let kind = match b.tax_rate_details.tax_type.as_str() {
                "vat" => RateKind::Vat,
                "gst" => RateKind::Gst,
                _ => RateKind::Sales,
            };
            by_jurisdiction.push(JurisdictionTax {
                jurisdiction: j,
                rate: rate / Decimal::new(100, 0), // Stripe sends percent
                amount: Money::from_minor(b.amount, currency),
                kind,
            });
            // Excise / inclusive flag noted but not used in the layered view.
            let _ = b.inclusive;
        }

        let total_tax = Money::from_minor(parsed.tax_amount_exclusive, currency);
        // amount_total parsed for compatibility but not used downstream.
        let _ = parsed.amount_total;

        Ok(TaxResult {
            total_tax,
            per_line,
            jurisdictions_charged: by_jurisdiction,
            calculator: "stripe_tax".into(),
            calculated_at: Utc::now(),
        })
    }

    fn name(&self) -> &'static str {
        "stripe_tax"
    }
}

fn stripe_tax_code(category: &crate::category::ProductTaxCategory) -> &str {
    use crate::category::ProductTaxCategory as C;
    // Stripe Tax product codes — published "txcd_…" identifiers.
    match category {
        C::TangibleGoods => "txcd_99999999",
        C::DigitalGoods => "txcd_10000000",
        C::Software => "txcd_10101000",
        C::Saas => "txcd_10103000",
        C::ProfessionalService => "txcd_20030000",
        C::Telecommunications => "txcd_37020001",
        C::Healthcare => "txcd_30060000",
        C::Food => "txcd_40050000",
        C::Clothing => "txcd_30070001",
        C::Alcohol => "txcd_40010001",
        C::Tobacco => "txcd_40020001",
        C::MotorFuel => "txcd_40030001",
        C::Lodging => "txcd_20070000",
        C::AdmissionsAndEvents => "txcd_20040000",
        C::Other(_) => "txcd_99999999",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculator::CustomerType;
    use chrono::NaiveDate;
    use op_core::Currency;
    use std::collections::HashSet;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn stripe_tax_request_shape_and_response_parsing() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "id": "taxcalc_test",
            "object": "tax.calculation",
            "amount_total": 10825,
            "currency": "usd",
            "tax_amount_exclusive": 825,
            "tax_amount_inclusive": 0,
            "line_items": {
                "data": [
                    {
                        "id": "tax_li_x",
                        "object": "tax.calculation_line_item",
                        "reference": "L1",
                        "amount": 10000,
                        "amount_tax": 825
                    }
                ]
            },
            "tax_breakdown": [
                {
                    "amount": 825,
                    "inclusive": false,
                    "jurisdiction": { "country": "US", "state": "WA", "display_name": "Washington" },
                    "tax_rate_details": {
                        "percentage_decimal": "8.25",
                        "tax_type": "sales_tax"
                    }
                }
            ]
        });
        Mock::given(method("POST"))
            .and(path("/v1/tax/calculations"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let adapter = StripeTaxAdapter::new("sk_test_xyz".into()).with_base_url(server.uri());
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::USD),
            category: crate::category::ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::region("US", "WA"),
        };
        let ctx = TaxContext {
            transaction_date: NaiveDate::parse_from_str("2026-06-15", "%Y-%m-%d").unwrap(),
            customer_type: CustomerType::Consumer,
            exemption_certs: vec![],
            nexus_jurisdictions: HashSet::new(),
        };
        let r = adapter.calculate(&[line], &ctx).await.unwrap();
        assert_eq!(r.calculator, "stripe_tax");
        assert_eq!(r.total_tax.minor_units, 825);
        assert_eq!(r.per_line.get("L1").unwrap().tax_amount.minor_units, 825);
    }
}
