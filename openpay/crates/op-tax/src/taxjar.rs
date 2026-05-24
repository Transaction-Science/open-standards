//! TaxJar (now Stripe-owned) adapter.
//!
//! TaxJar's API is the cleanest of the bunch — JSON-in, JSON-out,
//! Bearer token, predictable shapes. We talk to
//! `POST /v2/taxes` on `https://api.taxjar.com`.
//!
//! Reference: <https://developers.taxjar.com/api/reference/#post-calculate-sales-tax-for-an-order>.

use async_trait::async_trait;
use chrono::Utc;
use op_core::{Currency, Money};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::calculator::{
    JurisdictionTax, LineTaxBreakdown, TaxCalculator, TaxContext, TaxResult, TaxableLine,
};
use crate::error::{Error, Result};
use crate::jurisdiction::{CountryCode, Jurisdiction, RegionCode};
use crate::rate_table::RateKind;

/// TaxJar client.
pub struct TaxJarAdapter {
    client: reqwest::Client,
    base_url: String,
    api_token: String,
}

impl TaxJarAdapter {
    /// Construct with a TaxJar API token.
    #[must_use]
    pub fn new(api_token: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://api.taxjar.com".into(),
            api_token,
        }
    }

    /// Builder: override the base URL.
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

#[derive(Serialize)]
struct TaxJarRequest<'a> {
    from_country: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    from_state: Option<&'a str>,
    to_country: &'a str,
    #[serde(skip_serializing_if = "Option::is_none")]
    to_state: Option<&'a str>,
    #[serde(skip_serializing_if = "Option::is_none")]
    to_city: Option<&'a str>,
    amount: f64,
    shipping: f64,
    line_items: Vec<TaxJarLineItem<'a>>,
}

#[derive(Serialize)]
struct TaxJarLineItem<'a> {
    id: &'a str,
    quantity: u32,
    unit_price: f64,
    #[serde(skip_serializing_if = "Option::is_none")]
    product_tax_code: Option<&'a str>,
}

#[derive(Deserialize, Debug)]
struct TaxJarResponse {
    tax: TaxJarTax,
}

#[derive(Deserialize, Debug)]
struct TaxJarTax {
    amount_to_collect: f64,
    #[serde(default)]
    breakdown: Option<TaxJarBreakdown>,
}

#[derive(Deserialize, Debug, Default)]
struct TaxJarBreakdown {
    #[serde(default)]
    state_tax_collectable: f64,
    #[serde(default)]
    county_tax_collectable: f64,
    #[serde(default)]
    city_tax_collectable: f64,
    #[serde(default)]
    special_district_tax_collectable: f64,
    #[serde(default)]
    line_items: Vec<TaxJarLineBreakdown>,
}

#[derive(Deserialize, Debug, Default)]
struct TaxJarLineBreakdown {
    id: String,
    #[serde(default)]
    tax_collectable: f64,
    #[serde(default)]
    taxable_amount: f64,
    #[serde(default)]
    combined_tax_rate: f64,
}

fn major_to_minor(major: f64, currency: Currency) -> i64 {
    let exp = i32::from(currency.exponent());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = (major * 10f64.powi(exp)).round() as i64;
    v
}

fn minor_to_major(minor: i64, currency: Currency) -> f64 {
    let exp = i32::from(currency.exponent());
    #[allow(clippy::cast_precision_loss)]
    let v = (minor as f64) / 10f64.powi(exp);
    v
}

#[async_trait]
impl TaxCalculator for TaxJarAdapter {
    async fn calculate(&self, lines: &[TaxableLine], _ctx: &TaxContext) -> Result<TaxResult> {
        if lines.is_empty() {
            return Err(Error::Config("calculate called with zero lines".into()));
        }
        let currency = lines[0].amount.currency;
        for l in lines {
            if l.amount.currency != currency {
                return Err(Error::Money(op_core::Error::CurrencyMismatch));
            }
        }
        let to = &lines[0].ship_to;
        let total_minor: i64 = lines.iter().map(|l| l.amount.minor_units).sum();
        let from = lines.iter().find_map(|l| l.ship_from.as_ref());
        let body = TaxJarRequest {
            from_country: from.map_or("US", |j| j.country.0.as_str()),
            from_state: from.and_then(|j| j.region.as_ref().map(|r| r.0.as_str())),
            to_country: &to.country.0,
            to_state: to.region.as_ref().map(|r| r.0.as_str()),
            to_city: to.locality.as_ref().map(|l| l.0.as_str()),
            amount: minor_to_major(total_minor, currency),
            shipping: 0.0,
            line_items: lines
                .iter()
                .map(|l| TaxJarLineItem {
                    id: l.line_id.as_str(),
                    quantity: 1,
                    unit_price: minor_to_major(l.amount.minor_units, currency),
                    product_tax_code: taxjar_code(&l.category),
                })
                .collect(),
        };
        let url = format!("{}/v2/taxes", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.api_token)
            .json(&body)
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
        let parsed: TaxJarResponse = resp.json().await?;
        let total_tax_minor = major_to_minor(parsed.tax.amount_to_collect, currency);
        let mut per_line = BTreeMap::new();
        let mut jurisdictions_charged: Vec<JurisdictionTax> = Vec::new();
        if let Some(bd) = parsed.tax.breakdown {
            for li in bd.line_items {
                per_line.insert(
                    li.id.clone(),
                    LineTaxBreakdown {
                        line_id: li.id,
                        taxable_amount: Money::from_minor(
                            major_to_minor(li.taxable_amount, currency),
                            currency,
                        ),
                        tax_amount: Money::from_minor(
                            major_to_minor(li.tax_collectable, currency),
                            currency,
                        ),
                        effective_rate: Decimal::from_f64_retain(li.combined_tax_rate)
                            .unwrap_or(Decimal::ZERO),
                        jurisdiction_layers: Vec::new(),
                        exemption_reason: None,
                    },
                );
            }
            let to_country = CountryCode::new(&to.country.0);
            let to_region = to.region.as_ref().map(|r| RegionCode::new(&r.0));
            for (label, amount, kind, with_region) in [
                ("state", bd.state_tax_collectable, RateKind::Sales, true),
                ("county", bd.county_tax_collectable, RateKind::Sales, true),
                ("city", bd.city_tax_collectable, RateKind::Sales, true),
                (
                    "district",
                    bd.special_district_tax_collectable,
                    RateKind::Sales,
                    true,
                ),
            ] {
                if amount > 0.0 {
                    let _ = label;
                    let j = Jurisdiction {
                        country: to_country.clone(),
                        region: if with_region { to_region.clone() } else { None },
                        locality: None,
                        special_district: None,
                    };
                    jurisdictions_charged.push(JurisdictionTax {
                        jurisdiction: j,
                        rate: Decimal::ZERO,
                        amount: Money::from_minor(major_to_minor(amount, currency), currency),
                        kind,
                    });
                }
            }
        }
        Ok(TaxResult {
            total_tax: Money::from_minor(total_tax_minor, currency),
            per_line,
            jurisdictions_charged,
            calculator: "taxjar".into(),
            calculated_at: Utc::now(),
        })
    }

    fn name(&self) -> &'static str {
        "taxjar"
    }
}

fn taxjar_code(category: &crate::category::ProductTaxCategory) -> Option<&str> {
    use crate::category::ProductTaxCategory as C;
    // TaxJar published product-tax-codes (subset). When None, TaxJar
    // treats the line as general tangible-goods.
    match category {
        C::Clothing => Some("20010"),
        C::Food => Some("40030"),
        C::Software => Some("30070"),
        C::Saas => Some("30070"),
        C::DigitalGoods => Some("31000"),
        C::Healthcare => Some("51010"),
        _ => None,
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
    async fn taxjar_response_parsed() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "tax": {
                "amount_to_collect": 8.25,
                "breakdown": {
                    "state_tax_collectable": 6.5,
                    "county_tax_collectable": 0.0,
                    "city_tax_collectable": 1.75,
                    "special_district_tax_collectable": 0.0,
                    "line_items": [
                        {
                            "id": "L1",
                            "tax_collectable": 8.25,
                            "taxable_amount": 100.0,
                            "combined_tax_rate": 0.0825
                        }
                    ]
                }
            }
        });
        Mock::given(method("POST"))
            .and(path("/v2/taxes"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let adapter = TaxJarAdapter::new("tk".into()).with_base_url(server.uri());
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
        assert_eq!(r.calculator, "taxjar");
        assert_eq!(r.total_tax.minor_units, 825);
    }
}
