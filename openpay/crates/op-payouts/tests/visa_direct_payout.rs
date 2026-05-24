//! Integration: Visa Direct OCT body shape.

use op_core::{Currency, Money};
use op_payouts::visa_direct::VisaDirectDriver;
use op_payouts::{
    Beneficiary, BeneficiaryAccount, FundingSource, Payout, PayoutMethod, PayoutRequest,
    PayoutStatus,
};

fn req() -> PayoutRequest {
    PayoutRequest {
        idempotency_key: "11111111-1111-4111-8111-111111111111".to_string(),
        method: PayoutMethod::VisaDirect,
        amount: Money::from_minor(12_345, Currency::USD),
        beneficiary: Beneficiary {
            name: "JANE DOE".to_string(),
            address: None,
            account: BeneficiaryAccount::CardPan("4111111111111111".to_string()),
            kyc_ref: Some("kyc_abc".to_string()),
        },
        funding: FundingSource::Prefunded {
            account_ref: "vd-funding-1".to_string(),
        },
        memo: Some("payout".to_string()),
    }
}

#[test]
fn visa_direct_prepares_offline_with_json_body() {
    let driver = VisaDirectDriver {
        acquiring_bin: "408999".to_string(),
        acquirer_country_code: "840".to_string(),
        business_application_id: "AA".to_string(),
    };
    let res = driver.submit(&req()).expect("offline build");
    assert_eq!(res.status, PayoutStatus::PreparedOffline);
    let payload = res.wire_payload.expect("json bytes");
    let body: serde_json::Value = serde_json::from_slice(&payload).expect("json");
    assert_eq!(body["acquiringBin"], "408999");
    assert_eq!(body["transactionCurrencyCode"], "USD");
    assert_eq!(body["amount"], "123.45");
    assert_eq!(body["recipientPrimaryAccountNumber"], "4111111111111111");
}

#[test]
fn visa_direct_rejects_iban() {
    let driver = VisaDirectDriver {
        acquiring_bin: "408999".to_string(),
        acquirer_country_code: "840".to_string(),
        business_application_id: "AA".to_string(),
    };
    let mut r = req();
    r.beneficiary.account = BeneficiaryAccount::Iban("DE89370400440532013000".to_string());
    let err = driver.submit(&r).unwrap_err();
    assert!(matches!(
        err,
        op_payouts::Error::UnsupportedMethod {
            rail: "visa_direct"
        }
    ));
}

#[test]
fn visa_direct_rejects_short_pan() {
    let driver = VisaDirectDriver {
        acquiring_bin: "408999".to_string(),
        acquirer_country_code: "840".to_string(),
        business_application_id: "AA".to_string(),
    };
    let mut r = req();
    r.beneficiary.account = BeneficiaryAccount::CardPan("411111".to_string());
    assert!(driver.submit(&r).is_err());
}
