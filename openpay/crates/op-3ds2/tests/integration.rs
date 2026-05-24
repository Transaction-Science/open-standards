#![allow(
    clippy::inconsistent_digit_grouping,
    clippy::field_reassign_with_default,
    clippy::missing_errors_doc,
    clippy::unreadable_literal,
    clippy::doc_markdown
)]

//! End-to-end integration tests for op-3ds2.
//!
//! These exercise the cross-module wiring that the unit tests inside
//! each module cannot reach: DS routing → version negotiation →
//! AReq → mocked DS → ARes (with correct ECI / CAVV per scheme),
//! decoupled poll loop, fingerprint round-trip.

use chrono::Utc;
use op_3ds2::{
    AReq, AcsConfig, AcsServer, BrowserFingerprint, ChallengeMode, ChallengeSession,
    DecoupledPollResult, DecoupledSession, DeviceChannel, DirectoryServer, EligibleExemption,
    Error, ExemptionContext, FraudRateBracket, MessageCategory, ProtocolVersion,
    TransactionStatus, evaluate,
    amex_ds::AmexDs,
    directory_server::{DsRoute, route_for_pan},
    exemption,
    fingerprint::fingerprint_collector_script,
    mc_ds::MastercardDs,
    risk::BrowserInfo,
    visa_ds::VisaDs,
};
use op_core::{Currency, Money};

fn sample_browser_areq(pan: &str) -> AReq {
    AReq {
        message_version: "2.2.0".into(),
        three_ds_server_trans_id: "11111111-1111-1111-1111-111111111111".into(),
        three_ds_server_ref_number: "3DS_LOA_SER_OPNP_020100_00001".into(),
        three_ds_server_url: "https://3ds.openpay.example/callback".into(),
        three_ds_requestor_id: "openpay_req_id".into(),
        three_ds_requestor_name: "OpenPay Merchant".into(),
        three_ds_requestor_challenge_ind: "01".into(),
        message_category: MessageCategory::Payment,
        device_channel: DeviceChannel::Browser,
        acct_number: pan.to_owned(),
        acct_type: Some("02".into()),
        merchant_name: "OpenPay Merchant".into(),
        mcc: "5411".into(),
        acquirer_bin: "400000".into(),
        acquirer_merchant_id: "MID-OPENPAY-001".into(),
        merchant_country_code: "840".into(),
        purchase_currency: "840".into(),
        purchase_amount: "1234".into(),
        purchase_exponent: 2,
        purchase_date: "20260524093000".into(),
        browser_info: Some(BrowserInfo::sample()),
        sdk_app_id: None,
        sdk_ephem_pub_key: None,
        sdk_reference_number: None,
        sdk_trans_id: None,
        sdk_max_timeout: None,
        three_ri_ind: None,
        three_ds_req_auth_method: None,
        decoupled_auth_ind: None,
        decoupled_auth_max_time: None,
        white_list_status: None,
        spc_incomp: None,
        delegated_auth_data: None,
        merchant_risk_indicator: None,
        acct_info: None,
        extensions: vec![],
    }
}

#[tokio::test]
async fn visa_areq_round_trip_via_stub_ds() {
    let pan = "4111111111111111";
    assert_eq!(route_for_pan(pan).unwrap(), DsRoute::Visa);
    let ds = VisaDs::default();
    let vc = ds.version_check(pan).await.unwrap();
    let chosen = vc.negotiate(&[ProtocolVersion::V2_3, ProtocolVersion::V2_2]).unwrap();
    assert_eq!(chosen, ProtocolVersion::V2_3);

    let areq = sample_browser_areq(pan);
    areq.validate(chosen).unwrap();
    let ares = ds.auth_request(&areq).await.unwrap();
    assert_eq!(ares.trans_status, "Y");
    assert_eq!(ares.eci.as_deref(), Some("05"));
    assert!(ares.authentication_value.is_some(), "Visa ARes must carry CAVV");
}

#[tokio::test]
async fn mastercard_areq_round_trip_with_02_eci() {
    let pan = "5500000000000004";
    assert_eq!(route_for_pan(pan).unwrap(), DsRoute::Mastercard);
    let ds = MastercardDs::default();
    let vc = ds.version_check(pan).await.unwrap();
    let chosen = vc
        .negotiate(&[ProtocolVersion::V2_3, ProtocolVersion::V2_2, ProtocolVersion::V2_1])
        .unwrap();
    assert_eq!(chosen, ProtocolVersion::V2_3);

    let areq = sample_browser_areq(pan);
    let ares = ds.auth_request(&areq).await.unwrap();
    assert_eq!(ares.eci.as_deref(), Some("02"), "Mastercard success ECI is 02");
}

#[tokio::test]
async fn amex_areq_round_trip_with_05_eci() {
    let pan = "378282246310005";
    assert_eq!(route_for_pan(pan).unwrap(), DsRoute::Amex);
    let ds = AmexDs::default();
    let ares = ds.auth_request(&sample_browser_areq(pan)).await.unwrap();
    assert_eq!(ares.eci.as_deref(), Some("05"));
}

#[tokio::test]
async fn version_negotiation_falls_back_when_2_3_unsupported() {
    // Mastercard stub advertises 2.1.0 / 2.2.0 / 2.3.0; trim it.
    let mut ds = MastercardDs::default();
    ds.acs_reference_number = "3DS_LOA_ACS_MCDS_020200_00001".into();
    let vc = ds.version_check("5500000000000004").await.unwrap();
    let chosen = vc.negotiate(&[ProtocolVersion::V2_2, ProtocolVersion::V2_1]).unwrap();
    assert_eq!(chosen, ProtocolVersion::V2_2);
}

#[tokio::test]
async fn decoupled_poll_settles_after_three_pending() {
    let s = DecoupledSession {
        polling_url: "https://acs.example/poll".into(),
        three_ds_server_trans_id: "tid".into(),
        decoupled_auth_max_time: 10,
        poll_interval: std::time::Duration::from_millis(1),
        max_polls: 10,
    };
    let counter = std::sync::Arc::new(std::sync::Mutex::new(0_u32));
    let result = s
        .run(|_| {
            let counter = std::sync::Arc::clone(&counter);
            async move {
                let mut c = counter.lock().unwrap();
                *c += 1;
                if *c < 3 {
                    Ok(DecoupledPollResult::Pending)
                } else {
                    Ok(DecoupledPollResult::Approved {
                        authentication_value: "CAVV==".into(),
                        eci: "05".into(),
                    })
                }
            }
        })
        .await
        .unwrap();
    assert!(matches!(result, DecoupledPollResult::Approved { .. }));
    assert_eq!(*counter.lock().unwrap(), 3);
}

#[test]
fn fingerprint_collector_emits_script_then_parses_back() {
    let script = fingerprint_collector_script("https://collector.example", "tid-x");
    assert!(script.contains("collector.example"));
    let body = "ua=Mozilla&accept=text%2Fhtml&lang=en-US&langs=en-US%2Cen\
                &sw=1920&sh=1080&cd=24&tz=-420&java=0";
    let fp = BrowserFingerprint::parse_form(body).unwrap();
    assert_eq!(fp.screen_width, 1920);
    assert_eq!(fp.timezone_offset, -420);
    let bi = fp.into_browser_info(Some("198.51.100.1".into()));
    assert_eq!(bi.user_agent, "Mozilla");
    assert_eq!(bi.ip.as_deref(), Some("198.51.100.1"));
}

#[test]
fn exemption_low_value_25_eur() {
    let intent = exemption::PaymentIntent {
        amount: Money::from_minor(25_00, Currency::EUR),
        pan: "4111111111111111".into(),
        recurring: false,
        merchant_initiated: false,
        mandate_ref: None,
        subscription_id: None,
    };
    let ctx = ExemptionContext {
        fraud_rate_bracket: FraudRateBracket::AboveTraThreshold,
        ..Default::default()
    };
    let ex = evaluate(&intent, &ctx);
    assert!(
        ex.iter()
            .any(|e| matches!(e, EligibleExemption::LowValueTransaction { .. }))
    );
}

#[test]
fn exemption_100_eur_qualifies_for_tra_only_when_bracket_allows() {
    let intent = exemption::PaymentIntent {
        amount: Money::from_minor(100_00, Currency::EUR),
        pan: "4111111111111111".into(),
        recurring: false,
        merchant_initiated: false,
        mandate_ref: None,
        subscription_id: None,
    };
    // AboveTraThreshold → not eligible.
    let ctx_bad = ExemptionContext {
        fraud_rate_bracket: FraudRateBracket::AboveTraThreshold,
        tra_score: Some(0.05),
        ..Default::default()
    };
    let ex = evaluate(&intent, &ctx_bad);
    assert!(
        !ex.iter()
            .any(|e| matches!(e, EligibleExemption::TransactionRiskAnalysis { .. }))
    );
    // UpTo13Bp (cap 100 EUR) → eligible (boundary inclusive).
    let ctx_ok = ExemptionContext {
        fraud_rate_bracket: FraudRateBracket::UpTo13Bp,
        tra_score: Some(0.05),
        ..Default::default()
    };
    let ex = evaluate(&intent, &ctx_ok);
    assert!(
        ex.iter()
            .any(|e| matches!(e, EligibleExemption::TransactionRiskAnalysis { .. }))
    );
}

#[test]
fn acs_emits_ares_with_decision_letter() {
    let server = AcsServer::new(AcsConfig {
        acs_operator_id: "openpay-acs".into(),
        acs_reference_number: "3DS_LOA_ACS_OPNP_020300_00001".into(),
        acs_url: "https://acs.openpay.example/challenge".into(),
        acs_decoupled_url: Some("https://acs.openpay.example/poll".into()),
    });
    let areq = sample_browser_areq("4111111111111111");
    let ares = server.build_ares(
        &areq,
        TransactionStatus::ChallengeRequired,
        None,
        None,
    );
    assert_eq!(ares.trans_status, "C");
    assert!(ares.acs_url.is_some());
}

#[test]
fn challenge_session_builds_initial_creq() {
    let s = ChallengeSession::new("t-1".into(), "a-1".into(), ChallengeMode::Html);
    let c = s.initial_creq("2.2.0", "05");
    assert_eq!(c.three_ds_server_trans_id, "t-1");
    assert_eq!(c.acs_trans_id, "a-1");
}

#[test]
fn challenge_session_settles_authenticated_cres() {
    let s = ChallengeSession::new("t".into(), "a".into(), ChallengeMode::Html);
    let cres = op_3ds2::CRes {
        message_version: "2.2.0".into(),
        three_ds_server_trans_id: "t".into(),
        acs_trans_id: "a".into(),
        acs_counter_a_to_s: Some("002".into()),
        acs_html: None,
        challenge_completion_ind: Some("Y".into()),
        trans_status: Some("Y".into()),
        trans_status_reason: None,
        oob_app_url: None,
        oob_app_label: None,
        acs_decoupled_url: None,
    };
    let r = s.settle(cres, Utc::now()).unwrap();
    assert_eq!(r.trans_status, TransactionStatus::Authenticated);
}

#[test]
fn invalid_pan_no_route() {
    assert!(matches!(route_for_pan(""), Err(Error::InvalidPan)));
    assert!(matches!(
        route_for_pan("9999999999999999"),
        Err(Error::NoDsRoute { .. })
    ));
}
