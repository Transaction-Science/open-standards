//! Integration: USDC payout on Base produces ERC-20 transfer calldata.

use op_core::{Currency, Money};
use op_payouts::crypto::{CryptoPayoutDriver, USDC_BASE};
use op_payouts::{
    Beneficiary, BeneficiaryAccount, FundingSource, Payout, PayoutMethod, PayoutRequest,
    PayoutStatus,
};

#[test]
fn usdc_on_base_builds_evm_intent() {
    let driver = CryptoPayoutDriver {
        source_wallet: "0xfeedfacefeedfacefeedfacefeedfacefeedface".to_string(),
    };
    // USDC has 6 decimals on Base. ISO 4217 has no slot for crypto and
    // `Currency::try_new` caps exponent at 4, so we model the on-chain
    // amount as raw token-minor-units carried in a Currency with
    // exponent 0. The driver reads `minor_units` verbatim into
    // calldata, which is what we assert.
    let usdc = Currency::try_new(*b"USC", 0).expect("currency");
    let amount = Money::from_minor(25_000_000, usdc); // 25 USDC in 6-dec minor units
    let req = PayoutRequest {
        idempotency_key: "44444444-4444-4444-8444-444444444444".to_string(),
        method: PayoutMethod::Crypto {
            asset: "USDC".to_string(),
            network: "base".to_string(),
        },
        amount,
        beneficiary: Beneficiary {
            name: "wallet".to_string(),
            address: None,
            account: BeneficiaryAccount::EvmAddress(
                "0x000000000000000000000000000000000000beef".to_string(),
            ),
            kyc_ref: None,
        },
        funding: FundingSource::Prefunded {
            account_ref: "hot-wallet-base".to_string(),
        },
        memo: None,
    };

    let res = driver.submit(&req).expect("offline build");
    assert_eq!(res.status, PayoutStatus::PreparedOffline);
    let body: serde_json::Value =
        serde_json::from_slice(&res.wire_payload.expect("payload")).expect("json");
    assert_eq!(body["chain"], "base");
    assert_eq!(body["asset"], "USDC");
    assert_eq!(body["token_contract"], USDC_BASE);
    let calldata = body["calldata_hex"].as_str().expect("calldata");
    assert!(calldata.starts_with("0xa9059cbb"));
    // amount field at the end: 25_000_000 in hex right-padded to 64 chars
    let expected_amt = format!("{:064x}", 25_000_000u128);
    assert!(calldata.ends_with(&expected_amt));
}

#[test]
fn usdc_evm_rejects_bad_address() {
    let driver = CryptoPayoutDriver {
        source_wallet: "0xfeed".to_string(),
    };
    let usdc = Currency::try_new(*b"USC", 0).expect("currency");
    let req = PayoutRequest {
        idempotency_key: "55555555-5555-4555-8555-555555555555".to_string(),
        method: PayoutMethod::Crypto {
            asset: "USDC".to_string(),
            network: "ethereum".to_string(),
        },
        amount: Money::from_minor(1_000_000, usdc),
        beneficiary: Beneficiary {
            name: "wallet".to_string(),
            address: None,
            account: BeneficiaryAccount::EvmAddress("0xnothex".to_string()),
            kyc_ref: None,
        },
        funding: FundingSource::Prefunded {
            account_ref: "hot".to_string(),
        },
        memo: None,
    };
    assert!(driver.submit(&req).is_err());
}
