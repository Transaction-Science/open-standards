//! Vertex O Series adapter.
//!
//! Vertex is the enterprise tax engine of choice for large multi-
//! national retailers and manufacturers; the Fortune 500 list of
//! Vertex customers is unusually long.
//!
//! Talks to `POST /vertex-restapi/v1/sale` on the Vertex O Series
//! REST surface. Authentication: OAuth 2.0 bearer token (operators
//! obtain via the Vertex authentication endpoint and pass as
//! `bearer_token` here — token refresh is out of scope for v1, but
//! the adapter shape makes it easy to layer on).

use async_trait::async_trait;
use chrono::Utc;
use op_core::{Currency, Money};
use rust_decimal::Decimal;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

use crate::calculator::{
    LineTaxBreakdown, TaxCalculator, TaxContext, TaxResult, TaxableLine,
};
use crate::error::{Error, Result};

/// Vertex O Series client.
pub struct VertexAdapter {
    client: reqwest::Client,
    base_url: String,
    bearer_token: String,
    company_code: String,
}

impl VertexAdapter {
    /// Construct with a bearer token and seller company code.
    #[must_use]
    pub fn new(bearer_token: String, company_code: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://restconnect.vertexsmb.com".into(),
            bearer_token,
            company_code,
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
struct VertexSaleRequest<'a> {
    #[serde(rename = "saleMessageType")]
    sale_message_type: &'a str,
    #[serde(rename = "transactionType")]
    transaction_type: &'a str,
    #[serde(rename = "documentDate")]
    document_date: String,
    seller: VertexSeller<'a>,
    customer: VertexCustomer<'a>,
    #[serde(rename = "lineItems")]
    line_items: Vec<VertexLineItem<'a>>,
}

#[derive(Serialize)]
struct VertexSeller<'a> {
    company: &'a str,
}

#[derive(Serialize)]
struct VertexCustomer<'a> {
    destination: VertexAddress<'a>,
}

#[derive(Serialize)]
struct VertexAddress<'a> {
    country: &'a str,
    #[serde(rename = "mainDivision", skip_serializing_if = "Option::is_none")]
    main_division: Option<&'a str>,
    #[serde(rename = "city", skip_serializing_if = "Option::is_none")]
    city: Option<&'a str>,
}

#[derive(Serialize)]
struct VertexLineItem<'a> {
    #[serde(rename = "lineItemNumber")]
    line_item_number: &'a str,
    #[serde(rename = "extendedPrice")]
    extended_price: String,
    product: VertexProduct<'a>,
    quantity: VertexQuantity,
}

#[derive(Serialize)]
struct VertexProduct<'a> {
    #[serde(rename = "productClass")]
    product_class: &'a str,
    #[serde(rename = "value")]
    value: &'a str,
}

#[derive(Serialize)]
struct VertexQuantity {
    value: i32,
}

#[derive(Deserialize, Debug)]
struct VertexSaleResponse {
    data: VertexSaleResponseData,
}

#[derive(Deserialize, Debug)]
struct VertexSaleResponseData {
    #[serde(rename = "totalTax")]
    total_tax: f64,
    #[serde(rename = "lineItems", default)]
    line_items: Vec<VertexResponseLine>,
}

#[derive(Deserialize, Debug, Default)]
struct VertexResponseLine {
    #[serde(rename = "lineItemNumber")]
    line_item_number: String,
    #[serde(rename = "totalTax", default)]
    total_tax: f64,
    #[serde(rename = "extendedPrice", default)]
    extended_price: f64,
}

fn major_to_minor(major: f64, currency: Currency) -> i64 {
    let exp = i32::from(currency.exponent());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = (major * 10f64.powi(exp)).round() as i64;
    v
}

fn minor_to_major(minor: i64, currency: Currency) -> String {
    let exp = i32::from(currency.exponent());
    let divisor = Decimal::new(10_i64.pow(u32::try_from(exp).unwrap_or(0)), 0);
    (Decimal::from(minor) / divisor).to_string()
}

#[async_trait]
impl TaxCalculator for VertexAdapter {
    async fn calculate(&self, lines: &[TaxableLine], ctx: &TaxContext) -> Result<TaxResult> {
        if lines.is_empty() {
            return Err(Error::Config("calculate called with zero lines".into()));
        }
        let currency = lines[0].amount.currency;
        for l in lines {
            if l.amount.currency != currency {
                return Err(Error::Money(op_core::Error::CurrencyMismatch));
            }
        }
        let dest = &lines[0].ship_to;
        let body = VertexSaleRequest {
            sale_message_type: "QUOTATION",
            transaction_type: "SALE",
            document_date: ctx.transaction_date.format("%Y-%m-%d").to_string(),
            seller: VertexSeller {
                company: &self.company_code,
            },
            customer: VertexCustomer {
                destination: VertexAddress {
                    country: &dest.country.0,
                    main_division: dest.region.as_ref().map(|r| r.0.as_str()),
                    city: dest.locality.as_ref().map(|l| l.0.as_str()),
                },
            },
            line_items: lines
                .iter()
                .map(|l| VertexLineItem {
                    line_item_number: l.line_id.as_str(),
                    extended_price: minor_to_major(l.amount.minor_units, currency),
                    product: VertexProduct {
                        product_class: vertex_product_class(&l.category),
                        value: l.line_id.as_str(),
                    },
                    quantity: VertexQuantity { value: 1 },
                })
                .collect(),
        };
        let url = format!("{}/vertex-restapi/v1/sale", self.base_url);
        let resp = self
            .client
            .post(&url)
            .bearer_auth(&self.bearer_token)
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
        let parsed: VertexSaleResponse = resp.json().await?;
        let mut per_line = BTreeMap::new();
        for li in parsed.data.line_items {
            per_line.insert(
                li.line_item_number.clone(),
                LineTaxBreakdown {
                    line_id: li.line_item_number,
                    taxable_amount: Money::from_minor(
                        major_to_minor(li.extended_price, currency),
                        currency,
                    ),
                    tax_amount: Money::from_minor(major_to_minor(li.total_tax, currency), currency),
                    effective_rate: Decimal::ZERO,
                    jurisdiction_layers: Vec::new(),
                    exemption_reason: None,
                },
            );
        }
        Ok(TaxResult {
            total_tax: Money::from_minor(major_to_minor(parsed.data.total_tax, currency), currency),
            per_line,
            jurisdictions_charged: Vec::new(),
            calculator: "vertex".into(),
            calculated_at: Utc::now(),
        })
    }

    fn name(&self) -> &'static str {
        "vertex"
    }
}

fn vertex_product_class(category: &crate::category::ProductTaxCategory) -> &str {
    use crate::category::ProductTaxCategory as C;
    // Vertex product-class identifiers — short labels passed through to
    // the configured tax matrices on the Vertex side.
    match category {
        C::TangibleGoods => "GENERAL",
        C::DigitalGoods => "DIGITAL",
        C::Software => "SOFTWARE",
        C::Saas => "SAAS",
        C::ProfessionalService => "SERVICE",
        C::Telecommunications => "TELECOM",
        C::Healthcare => "MEDICAL",
        C::Food => "FOOD",
        C::Clothing => "APPAREL",
        C::Alcohol => "ALCOHOL",
        C::Tobacco => "TOBACCO",
        C::MotorFuel => "FUEL",
        C::Lodging => "LODGING",
        C::AdmissionsAndEvents => "EVENTS",
        C::Other(_) => "GENERAL",
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
    async fn vertex_response_parsed() {
        let server = MockServer::start().await;
        let body = serde_json::json!({
            "data": {
                "totalTax": 8.25,
                "lineItems": [
                    {
                        "lineItemNumber": "L1",
                        "totalTax": 8.25,
                        "extendedPrice": 100.0
                    }
                ]
            }
        });
        Mock::given(method("POST"))
            .and(path("/vertex-restapi/v1/sale"))
            .respond_with(ResponseTemplate::new(200).set_body_json(body))
            .mount(&server)
            .await;
        let adapter = VertexAdapter::new("bearer123".into(), "OPENPAY".into())
            .with_base_url(server.uri());
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
        assert_eq!(r.calculator, "vertex");
        assert_eq!(r.total_tax.minor_units, 825);
    }
}
