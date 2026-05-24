//! Integration: FedNow payout shape + cap enforcement.

use op_core::{Currency, Money};
use op_payouts::fednow::{FedNowDriver, FEDNOW_MAX_AMOUNT_USD};
use op_payouts::{
    Beneficiary, BeneficiaryAccount, Error, FundingSource, Payout, PayoutMethod, PayoutRequest,
    PayoutStatus,
};

fn driver() -> FedNowDriver {
    FedNowDriver {
        sender_aba: "021000021".to_string(),
        sender_account: "0000123456".to_string(),
        sender_name: "Acme Inc".to_string(),
    }
}

fn req(amount: Money) -> PayoutRequest {
    PayoutRequest {
        idempotency_key: "33333333-3333-4333-8333-333333333333".to_string(),
        method: PayoutMethod::FedNow,
        amount,
        beneficiary: Beneficiary {
            name: "John Doe".to_string(),
            address: None,
            account: BeneficiaryAccount::UsBank {
                aba: "121000358".to_string(),
                account: "9876543210".to_string(),
                account_type: "CHECKING".to_string(),
            },
            kyc_ref: None,
        },
        funding: FundingSource::Prefunded {
            account_ref: "fednow-settlement".to_string(),
        },
        memo: None,
    }
}

#[test]
fn fednow_builds_pacs008_xml() {
    let amount = Money::from_minor(2_500_00, Currency::USD); // $2,500
    let res = driver().submit(&req(amount)).expect("offline build");
    assert_eq!(res.status, PayoutStatus::PreparedOffline);
    let xml = String::from_utf8(res.wire_payload.expect("payload")).expect("utf8");
    assert!(xml.contains("<Cd>FDN</Cd>"));
    assert!(xml.contains("Ccy=\"USD\""));
    assert!(xml.contains("2500.00"));
    assert!(xml.contains("<MmbId>121000358</MmbId>"));
}

#[test]
fn fednow_rejects_above_cap() {
    let amount = Money::from_minor(FEDNOW_MAX_AMOUNT_USD + 1, Currency::USD);
    let err = driver().submit(&req(amount)).unwrap_err();
    assert!(matches!(err, Error::LimitViolation { rail: "fednow", .. }));
}

#[test]
fn fednow_rejects_non_usd() {
    let amount = Money::from_minor(2_500_00, Currency::EUR);
    let err = driver().submit(&req(amount)).unwrap_err();
    assert!(matches!(err, Error::LimitViolation { rail: "fednow", .. }));
}
