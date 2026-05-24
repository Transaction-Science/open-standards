//! CCH SureTax adapter.
//!
//! CCH SureTax (Wolters Kluwer) is the dominant telecommunications-
//! tax engine in the US — every major carrier uses it for E911, USF,
//! state telecom excise, and the long tail of communications-specific
//! taxes.
//!
//! Talks to `POST /Services/V01/SureTax.asmx/PostRequest` on
//! `https://services.taxrating.net`. Authentication: a `ClientNumber`
//! / `ValidationKey` pair embedded in the request body (SureTax does
//! NOT use HTTP-level auth — historically the SureTax SDK predates
//! widespread Bearer-token use).

use async_trait::async_trait;
use chrono::Utc;
use op_core::Money;
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::calculator::{
    LineTaxBreakdown, TaxCalculator, TaxContext, TaxResult, TaxableLine,
};
use crate::error::{Error, Result};

/// CCH SureTax client.
pub struct CchSureTaxAdapter {
    client: reqwest::Client,
    base_url: String,
    client_number: String,
    validation_key: String,
    business_unit: String,
}

impl CchSureTaxAdapter {
    /// Construct with SureTax `ClientNumber`, `ValidationKey`, and
    /// business unit code.
    #[must_use]
    pub fn new(
        client_number: String,
        validation_key: String,
        business_unit: String,
    ) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://services.taxrating.net".into(),
            client_number,
            validation_key,
            business_unit,
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
struct SureTaxRequest<'a> {
    #[serde(rename = "ClientNumber")]
    client_number: &'a str,
    #[serde(rename = "ValidationKey")]
    validation_key: &'a str,
    #[serde(rename = "BusinessUnit")]
    business_unit: &'a str,
    #[serde(rename = "DataYear")]
    data_year: String,
    #[serde(rename = "DataMonth")]
    data_month: String,
    #[serde(rename = "TotalRevenue")]
    total_revenue: String,
    #[serde(rename = "ResponseGroup")]
    response_group: &'a str,
    #[serde(rename = "ResponseType")]
    response_type: &'a str,
    #[serde(rename = "ItemList")]
    item_list: Vec<SureTaxItem<'a>>,
}

#[derive(Serialize)]
struct SureTaxItem<'a> {
    #[serde(rename = "LineNumber")]
    line_number: &'a str,
    #[serde(rename = "Revenue")]
    revenue: String,
    #[serde(rename = "BillToCountry")]
    bill_to_country: &'a str,
    #[serde(rename = "BillToState")]
    bill_to_state: String,
    #[serde(rename = "TransTypeCode")]
    trans_type_code: &'a str,
}

#[derive(Deserialize, Debug)]
struct SureTaxResponse {
    #[serde(rename = "ResponseCode")]
    response_code: String,
    #[serde(rename = "TotalTax", default)]
    total_tax: String,
    #[serde(rename = "GroupList", default)]
    group_list: Vec<SureTaxGroup>,
}

#[derive(Deserialize, Debug, Default)]
struct SureTaxGroup {
    #[serde(rename = "LineNumber", default)]
    line_number: String,
    #[serde(rename = "TaxList", default)]
    tax_list: Vec<SureTaxLineTax>,
}

#[derive(Deserialize, Debug, Default)]
struct SureTaxLineTax {
    #[serde(rename = "TaxAmount", default)]
    tax_amount: String,
    #[serde(rename = "TaxOnTax", default)]
    _tax_on_tax: String,
}

#[async_trait]
impl TaxCalculator for CchSureTaxAdapter {
    async fn calculate(&self, lines: &[TaxableLine], ctx: &TaxContext) -> Result<TaxResult> {
        if lines.is_empty() {
            return Err(Error::Config("calculate called with zero lines".into()));
        }
        let currency = lines[0].amount.currency;
        let total_revenue: i64 = lines.iter().map(|l| l.amount.minor_units).sum();
        let exp = i32::from(currency.exponent());
        let divisor = Decimal::new(10_i64.pow(u32::try_from(exp).unwrap_or(0)), 0);

        let item_list: Vec<SureTaxItem<'_>> = lines
            .iter()
            .map(|l| SureTaxItem {
                line_number: l.line_id.as_str(),
                revenue: (Decimal::from(l.amount.minor_units) / divisor).to_string(),
                bill_to_country: &l.ship_to.country.0,
                bill_to_state: l
                    .ship_to
                    .region
                    .as_ref()
                    .map_or_else(String::new, |r| r.0.clone()),
                trans_type_code: cch_trans_code(&l.category),
            })
            .collect();

        let body = SureTaxRequest {
            client_number: &self.client_number,
            validation_key: &self.validation_key,
            business_unit: &self.business_unit,
            data_year: ctx.transaction_date.format("%Y").to_string(),
            data_month: ctx.transaction_date.format("%m").to_string(),
            total_revenue: (Decimal::from(total_revenue) / divisor).to_string(),
            response_group: "03",
            response_type: "D4",
            item_list,
        };

        let url = format!("{}/Services/V01/SureTax.asmx/PostRequest", self.base_url);
        let resp = self.client.post(&url).json(&body).send().await?;
        let status = resp.status();
        if !status.is_success() {
            let body = resp.text().await.unwrap_or_default();
            return Err(Error::Vendor {
                status: status.as_u16(),
                body,
            });
        }
        let parsed: SureTaxResponse = resp.json().await?;
        if parsed.response_code != "9999" {
            // 9999 = success in SureTax's response-code scheme.
            return Err(Error::Vendor {
                status: 200,
                body: format!("SureTax ResponseCode={}", parsed.response_code),
            });
        }

        let total_tax_decimal: Decimal = parsed.total_tax.parse().unwrap_or(Decimal::ZERO);
        let total_tax_minor = (total_tax_decimal * divisor)
            .round()
            .to_string()
            .parse::<i64>()
            .unwrap_or(0);

        let mut per_line = BTreeMap::new();
        for grp in parsed.group_list {
            let sum_minor: i64 = grp
                .tax_list
                .iter()
                .map(|t| {
                    let d: Decimal = t.tax_amount.parse().unwrap_or(Decimal::ZERO);
                    (d * divisor)
                        .round()
                        .to_string()
                        .parse::<i64>()
                        .unwrap_or(0)
                })
                .sum();
            // Find the original line to recover taxable_amount.
            let taxable = lines
                .iter()
                .find(|l| l.line_id == grp.line_number)
                .map_or(Money::from_minor(0, currency), |l| l.amount);
            per_line.insert(
                grp.line_number.clone(),
                LineTaxBreakdown {
                    line_id: grp.line_number,
                    taxable_amount: taxable,
                    tax_amount: Money::from_minor(sum_minor, currency),
                    effective_rate: Decimal::ZERO,
                    jurisdiction_layers: Vec::new(),
                    exemption_reason: None,
                },
            );
        }

        Ok(TaxResult {
            total_tax: Money::from_minor(total_tax_minor, currency),
            per_line,
            jurisdictions_charged: Vec::new(),
            calculator: "cch_suretax".into(),
            calculated_at: Utc::now(),
        })
    }

    fn name(&self) -> &'static str {
        "cch_suretax"
    }
}

fn cch_trans_code(category: &crate::category::ProductTaxCategory) -> &str {
    use crate::category::ProductTaxCategory as C;
    // SureTax TransTypeCode — the published telecom-tax classifications.
    // Non-telecom categories map to general-sales codes; SureTax
    // installations focus on telecom but accept the general codes.
    match category {
        C::Telecommunications => "010101", // Wireline voice intrastate
        C::DigitalGoods | C::Saas => "060601", // Digital download / SaaS
        C::Software => "010602",
        _ => "010100",
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculator::CustomerType;
    use crate::jurisdiction::Jurisdiction;
    use chrono::NaiveDate;
    use op_core::Currency;
    use std::collections::HashSet;
    use wiremock::matchers::{method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn cch_response_parsed() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "ResponseCode": "9999",
            "TotalTax": "8.25",
            "GroupList": [
                {
                    "LineNumber": "L1",
                    "TaxList": [
                        { "TaxAmount": "6.50", "TaxOnTax": "0.00" },
                        { "TaxAmount": "1.75", "TaxOnTax": "0.00" }
                    ]
                }
            ]
        });
        Mock::given(method("POST"))
            .and(path("/Services/V01/SureTax.asmx/PostRequest"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;

        let adapter = CchSureTaxAdapter::new("CN1".into(), "VK1".into(), "BU1".into())
            .with_base_url(server.uri());
        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::USD),
            category: crate::category::ProductTaxCategory::Telecommunications,
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
        assert_eq!(r.calculator, "cch_suretax");
        assert_eq!(r.total_tax.minor_units, 825);
        assert_eq!(r.per_line.get("L1").unwrap().tax_amount.minor_units, 825);
    }
}
