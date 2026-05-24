//! Avalara AvaTax adapter.
//!
//! Talks to `POST /api/v2/transactions/create` on Avalara's REST API.
//! Authentication is HTTP Basic with `<account_id>:<license_key>`.
//!
//! Reference: AvaTax REST v2 — `TransactionModel` schema. We map our
//! [`TaxableLine`] one-to-one onto Avalara's `LineItemModel`.

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
use crate::jurisdiction::Jurisdiction;
use crate::rate_table::RateKind;

/// Avalara AvaTax client.
pub struct AvalaraAdapter {
    client: reqwest::Client,
    base_url: String,
    account_id: String,
    license_key: String,
    company_code: String,
}

impl AvalaraAdapter {
    /// Construct with credentials and a company code. Uses Avalara's
    /// production endpoint by default; override the base URL for the
    /// sandbox / tests.
    #[must_use]
    pub fn new(account_id: String, license_key: String, company_code: String) -> Self {
        Self {
            client: reqwest::Client::new(),
            base_url: "https://rest.avatax.com".into(),
            account_id,
            license_key,
            company_code,
        }
    }

    /// Builder: override the base URL (Avalara sandbox is
    /// `https://sandbox-rest.avatax.com`; tests point at wiremock).
    #[must_use]
    pub fn with_base_url(mut self, url: impl Into<String>) -> Self {
        self.base_url = url.into();
        self
    }
}

// ---- Avalara wire types (subset) ----

#[derive(Serialize)]
struct CreateTransactionModel<'a> {
    #[serde(rename = "type")]
    transaction_type: &'a str,
    #[serde(rename = "companyCode")]
    company_code: &'a str,
    date: String,
    #[serde(rename = "customerCode")]
    customer_code: &'a str,
    currency_code: &'a str,
    lines: Vec<AvalaraLineModel<'a>>,
}

#[derive(Serialize)]
struct AvalaraLineModel<'a> {
    number: &'a str,
    quantity: i32,
    amount: f64,
    #[serde(rename = "taxCode")]
    tax_code: &'a str,
    #[serde(rename = "itemCode")]
    item_code: &'a str,
    addresses: AvalaraAddresses,
}

#[derive(Serialize)]
struct AvalaraAddresses {
    #[serde(rename = "shipFrom")]
    ship_from: Option<AvalaraAddress>,
    #[serde(rename = "shipTo")]
    ship_to: AvalaraAddress,
}

#[derive(Serialize)]
struct AvalaraAddress {
    country: String,
    region: Option<String>,
    city: Option<String>,
}

// ---- Avalara response (subset) ----

#[derive(Deserialize, Debug)]
struct TransactionResponse {
    /// Avalara's totalTax field is captured for envelope-shape
    /// validation; the authoritative per-line tax is summed from
    /// `lines[*].tax`.
    #[serde(rename = "totalTax", default)]
    _total_tax: f64,
    lines: Vec<ResponseLine>,
}

#[derive(Deserialize, Debug)]
struct ResponseLine {
    #[serde(rename = "lineNumber")]
    line_number: String,
    #[serde(rename = "tax")]
    tax: f64,
    #[serde(rename = "taxableAmount", default)]
    taxable_amount: f64,
    #[serde(default)]
    details: Vec<ResponseLineDetail>,
}

#[derive(Deserialize, Debug, Default)]
struct ResponseLineDetail {
    #[serde(default)]
    country: String,
    #[serde(default)]
    region: String,
    #[serde(default, rename = "jurisName")]
    juris_name: String,
    #[serde(default)]
    rate: f64,
    #[serde(default)]
    tax: f64,
    #[serde(default, rename = "taxType")]
    tax_type: String,
}

impl AvalaraAdapter {
    fn map_line(line: &TaxableLine) -> AvalaraLineModel<'_> {
        // Avalara's `amount` is a JSON number; we widen i64 minor
        // units to a float in the major-unit basis. This is the only
        // float in the whole crate and lives only in the wire payload —
        // the response we parse is rounded back to minor units.
        let exp = i32::from(line.amount.currency.exponent());
        #[allow(clippy::cast_precision_loss)]
        let major = (line.amount.minor_units as f64) / 10f64.powi(exp);

        let ship_to = AvalaraAddress {
            country: line.ship_to.country.0.clone(),
            region: line.ship_to.region.as_ref().map(|r| r.0.clone()),
            city: line.ship_to.locality.as_ref().map(|l| l.0.clone()),
        };
        let ship_from = line.ship_from.as_ref().map(|j| AvalaraAddress {
            country: j.country.0.clone(),
            region: j.region.as_ref().map(|r| r.0.clone()),
            city: j.locality.as_ref().map(|l| l.0.clone()),
        });
        AvalaraLineModel {
            number: &line.line_id,
            quantity: 1,
            amount: major,
            tax_code: avalara_tax_code(&line.category),
            item_code: line.line_id.as_str(),
            addresses: AvalaraAddresses { ship_from, ship_to },
        }
    }
}

fn avalara_tax_code(category: &crate::category::ProductTaxCategory) -> &str {
    use crate::category::ProductTaxCategory as C;
    // Avalara System Tax Codes — these are the published default tax
    // codes for each category. Operators with negotiated custom codes
    // can wrap the adapter to override.
    match category {
        C::TangibleGoods => "P0000000",
        C::DigitalGoods => "DC010000",
        C::Software => "DC020000",
        C::Saas => "DC020700",
        C::ProfessionalService => "SP000000",
        C::Telecommunications => "TC010000",
        C::Healthcare => "PH010000",
        C::Food => "PF050000",
        C::Clothing => "PC040100",
        C::Alcohol => "PA010000",
        C::Tobacco => "PT010000",
        C::MotorFuel => "PF010000",
        C::Lodging => "OL010000",
        C::AdmissionsAndEvents => "OE010000",
        C::Other(_) => "P0000000",
    }
}

fn major_to_minor(major: f64, currency: Currency) -> i64 {
    let exp = i32::from(currency.exponent());
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let v = (major * 10f64.powi(exp)).round() as i64;
    v
}

#[async_trait]
impl TaxCalculator for AvalaraAdapter {
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
        let body = CreateTransactionModel {
            transaction_type: "SalesInvoice",
            company_code: &self.company_code,
            date: ctx.transaction_date.format("%Y-%m-%d").to_string(),
            customer_code: "openpay",
            currency_code: currency.code(),
            lines: lines.iter().map(Self::map_line).collect(),
        };
        let url = format!("{}/api/v2/transactions/create", self.base_url);
        let resp = self
            .client
            .post(&url)
            .basic_auth(&self.account_id, Some(&self.license_key))
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
        let parsed: TransactionResponse = resp.json().await?;

        let mut per_line = BTreeMap::new();
        let mut by_jurisdiction: BTreeMap<(Jurisdiction, RateKind), JurisdictionTax> =
            BTreeMap::new();
        let mut total_tax = Money::from_minor(0, currency);

        for resp_line in &parsed.lines {
            let line_id = resp_line.line_number.clone();
            let tax_minor = major_to_minor(resp_line.tax, currency);
            let taxable_minor = major_to_minor(resp_line.taxable_amount, currency);
            let mut layers = Vec::with_capacity(resp_line.details.len());
            let mut effective = Decimal::ZERO;
            for d in &resp_line.details {
                let j = Jurisdiction {
                    country: crate::jurisdiction::CountryCode::new(&d.country),
                    region: if d.region.is_empty() {
                        None
                    } else {
                        Some(crate::jurisdiction::RegionCode::new(&d.region))
                    },
                    locality: if d.juris_name.is_empty() {
                        None
                    } else {
                        Some(crate::jurisdiction::LocalityCode::new(&d.juris_name))
                    },
                    special_district: None,
                };
                let kind = match d.tax_type.as_str() {
                    "VAT" | "Vat" => RateKind::Vat,
                    "GST" | "Gst" => RateKind::Gst,
                    "Excise" => RateKind::Excise,
                    "ImportDuty" => RateKind::ImportDuty,
                    _ => RateKind::Sales,
                };
                let rate = Decimal::from_f64_retain(d.rate).unwrap_or(Decimal::ZERO);
                effective += rate;
                let amt = Money::from_minor(major_to_minor(d.tax, currency), currency);
                let jt = JurisdictionTax {
                    jurisdiction: j.clone(),
                    rate,
                    amount: amt,
                    kind,
                };
                layers.push(jt.clone());
                by_jurisdiction
                    .entry((j, kind))
                    .and_modify(|existing| {
                        if let Ok(sum) = existing.amount.checked_add(amt) {
                            existing.amount = sum;
                        }
                    })
                    .or_insert(jt);
            }
            total_tax =
                total_tax.checked_add(Money::from_minor(tax_minor, currency))?;
            per_line.insert(
                line_id.clone(),
                LineTaxBreakdown {
                    line_id,
                    taxable_amount: Money::from_minor(taxable_minor, currency),
                    tax_amount: Money::from_minor(tax_minor, currency),
                    effective_rate: effective,
                    jurisdiction_layers: layers,
                    exemption_reason: None,
                },
            );
        }

        Ok(TaxResult {
            total_tax,
            per_line,
            jurisdictions_charged: by_jurisdiction.into_values().collect(),
            calculator: "avalara".into(),
            calculated_at: Utc::now(),
        })
    }

    fn name(&self) -> &'static str {
        "avalara"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::calculator::CustomerType;
    use chrono::NaiveDate;
    use std::collections::HashSet;
    use wiremock::matchers::{header, method, path};
    use wiremock::{Mock, MockServer, ResponseTemplate};

    #[tokio::test]
    async fn avalara_request_shape_and_response_parsing() {
        let server = MockServer::start().await;

        // Avalara's published response shape — abridged.
        let response_body = serde_json::json!({
            "id": 42,
            "code": "TEST-001",
            "totalTax": 8.25,
            "lines": [
                {
                    "lineNumber": "L1",
                    "tax": 8.25,
                    "taxableAmount": 100.0,
                    "details": [
                        {
                            "country": "US",
                            "region": "WA",
                            "jurisName": "WASHINGTON",
                            "rate": 0.065,
                            "tax": 6.5,
                            "taxType": "Sales"
                        },
                        {
                            "country": "US",
                            "region": "WA",
                            "jurisName": "Seattle",
                            "rate": 0.0175,
                            "tax": 1.75,
                            "taxType": "Sales"
                        }
                    ]
                }
            ]
        });

        Mock::given(method("POST"))
            .and(path("/api/v2/transactions/create"))
            .and(header("authorization", "Basic YWNjdDpsaWM=")) // base64("acct:lic")
            .respond_with(ResponseTemplate::new(200).set_body_json(response_body))
            .mount(&server)
            .await;

        let adapter = AvalaraAdapter::new("acct".into(), "lic".into(), "DEFAULT".into())
            .with_base_url(server.uri());

        let line = TaxableLine {
            line_id: "L1".into(),
            amount: Money::from_minor(10_000, Currency::USD),
            category: crate::category::ProductTaxCategory::TangibleGoods,
            ship_from: None,
            ship_to: Jurisdiction::locality("US", "WA", "Seattle"),
        };
        let ctx = TaxContext {
            transaction_date: NaiveDate::parse_from_str("2026-06-15", "%Y-%m-%d").unwrap(),
            customer_type: CustomerType::Consumer,
            exemption_certs: vec![],
            nexus_jurisdictions: HashSet::new(),
        };
        let r = adapter.calculate(&[line], &ctx).await.unwrap();
        assert_eq!(r.calculator, "avalara");
        assert_eq!(r.total_tax.minor_units, 825);
        let bd = r.per_line.get("L1").unwrap();
        assert_eq!(bd.jurisdiction_layers.len(), 2);
    }

    #[tokio::test]
    async fn avalara_non_success_returns_vendor_error() {
        let server = MockServer::start().await;
        Mock::given(method("POST"))
            .and(path("/api/v2/transactions/create"))
            .respond_with(ResponseTemplate::new(401).set_body_string("Unauthorized"))
            .mount(&server)
            .await;
        let adapter = AvalaraAdapter::new("acct".into(), "lic".into(), "C".into())
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
        let err = adapter.calculate(&[line], &ctx).await.unwrap_err();
        assert!(matches!(err, Error::Vendor { status: 401, .. }));
    }
}
