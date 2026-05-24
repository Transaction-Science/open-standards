//! End-to-end onboarding test for `op-connect`.
//!
//! Walks an account from BusinessInfo through TosAcceptance, verifying
//! that [`Requirements`] transitions correctly at each step and the
//! flow lands in [`OnboardingStatus::PendingVerification`] after the
//! last required step (because IdentityVerification is `eventually_due`).

use chrono::{NaiveDate, Utc};
use op_connect::{
    AccountId, AccountType, Address, BeneficialOwner, BusinessProfile, BusinessStructure,
    CountryCode, NativeProvider, OnboardingProvider, OnboardingStep, Person, StepPayload, TaxId,
    TosAcceptance,
    onboarding::ExternalAccount,
};

fn business() -> BusinessProfile {
    BusinessProfile {
        legal_name: "Acme Widgets LLC".into(),
        trade_name: Some("Acme".into()),
        structure: BusinessStructure::SingleMemberLlc,
        tax_id: Some(TaxId::Ein("12-3456789".into())),
        mcc: 5734,
        country: CountryCode("US".into()),
        registered_address: address(),
        support_email: Some("support@acme.example".into()),
        support_phone: None,
        website: Some("https://acme.example".into()),
    }
}

fn address() -> Address {
    Address {
        line1: "1 Main St".into(),
        line2: None,
        city: "Austin".into(),
        region: "TX".into(),
        postal_code: "78701".into(),
        country: CountryCode("US".into()),
    }
}

fn owner(name: &str, ownership: f32, control: bool) -> BeneficialOwner {
    BeneficialOwner {
        person: Person {
            legal_name: name.into(),
            dob: NaiveDate::from_ymd_opt(1985, 5, 15).expect("date"),
            address: address(),
            ssn_last_4: Some("1234".into()),
            ssn_or_itin_full: None,
            government_id: None,
        },
        ownership_pct: ownership,
        control,
        is_pep: false,
    }
}

#[tokio::test]
async fn full_onboarding_walk() {
    let provider = NativeProvider::new(AccountType::Custom);
    let acct = provider.create_account(&business()).await.expect("create");

    // Initial snapshot: currently_due has the canonical block of items.
    let snap = provider.snapshot(&acct).expect("snapshot");
    assert!(
        snap.requirements
            .currently_due
            .iter()
            .any(|r| r == "business.legal_name"),
        "expected business.legal_name as currently_due, got {:?}",
        snap.requirements.currently_due
    );

    // ---- 1. Business info ----
    let res = provider
        .submit_step(
            &acct,
            OnboardingStep::BusinessInfo,
            StepPayload::BusinessInfo(business()),
        )
        .await
        .expect("ok");
    assert!(res.accepted);
    assert!(!res
        .requirements
        .currently_due
        .iter()
        .any(|r| r == "business.legal_name"));

    // ---- 2. Representative info ----
    let rep_owner = owner("Jane Doe", 100.0, true);
    let res = provider
        .submit_step(
            &acct,
            OnboardingStep::RepresentativeInfo,
            StepPayload::RepresentativeInfo(rep_owner.clone()),
        )
        .await
        .expect("ok");
    assert!(res.accepted);
    assert!(!res
        .requirements
        .currently_due
        .iter()
        .any(|r| r == "representative.identity"));

    // ---- 3. Beneficial owners (FinCEN CDD) ----
    let res = provider
        .submit_step(
            &acct,
            OnboardingStep::BeneficialOwners,
            StepPayload::BeneficialOwners(vec![rep_owner]),
        )
        .await
        .expect("ok");
    assert!(res.accepted);
    assert!(!res
        .requirements
        .currently_due
        .iter()
        .any(|r| r == "beneficial_owners"));

    // ---- 4. Bank account ----
    let res = provider
        .submit_step(
            &acct,
            OnboardingStep::BankAccount,
            StepPayload::BankAccount(ExternalAccount {
                country: "US".into(),
                currency: "USD".into(),
                routing: "110000000".into(),
                account_number: "000123456789".into(),
                account_holder_name: "Acme Widgets LLC".into(),
            }),
        )
        .await
        .expect("ok");
    assert!(res.accepted);
    assert!(!res
        .requirements
        .currently_due
        .iter()
        .any(|r| r == "external_account"));
    // External-account verification is now currently-due.
    assert!(res
        .requirements
        .currently_due
        .iter()
        .any(|r| r == "external_account.verification"));

    // ---- 5. ToS acceptance ----
    let res = provider
        .submit_step(
            &acct,
            OnboardingStep::TosAcceptance,
            StepPayload::TosAcceptance(TosAcceptance {
                ip: "203.0.113.42".into(),
                user_agent: "Mozilla/5.0".into(),
                accepted_at: Utc::now(),
                version_hash: "tos-2026-01".into(),
            }),
        )
        .await
        .expect("ok");
    assert!(res.accepted);
    assert!(!res
        .requirements
        .currently_due
        .iter()
        .any(|r| r == "tos_acceptance"));

    // ---- 6. External-account verification (micro-deposits) ----
    provider.arm_micro_deposits(&acct, [27, 31]);
    let res = provider
        .submit_step(
            &acct,
            OnboardingStep::ExternalAccountVerification,
            StepPayload::ExternalAccountVerification {
                amounts_minor: [27, 31],
            },
        )
        .await
        .expect("ok");
    assert!(res.accepted);
    assert!(!res
        .requirements
        .currently_due
        .iter()
        .any(|r| r == "external_account.verification"));

    // ---- 7. Identity verification (optional / eventually-due) ----
    let mut docs = std::collections::BTreeMap::new();
    docs.insert("Jane Doe".into(), "tok_v7_id_document_001".into());
    let res = provider
        .submit_step(
            &acct,
            OnboardingStep::IdentityVerification,
            StepPayload::IdentityVerification { documents: docs },
        )
        .await
        .expect("ok");
    assert!(res.accepted);
    // Eventually-due cleared.
    assert!(!res
        .requirements
        .eventually_due
        .iter()
        .any(|r| r == "identity.document.verification"));

    // Now the account is fully clear.
    let status = provider.status(&acct).await.expect("status");
    assert_eq!(
        status,
        op_connect::OnboardingStatus::Active,
        "expected active after final step; got {status:?}"
    );

    let snap = provider.snapshot(&acct).expect("snap");
    assert!(snap.is_active(), "account should be active");
}

#[tokio::test]
async fn unknown_account_errors() {
    let provider = NativeProvider::new(AccountType::Custom);
    let bogus = AccountId("acct_does_not_exist".into());
    let err = provider
        .status(&bogus)
        .await
        .expect_err("unknown account");
    assert!(matches!(err, op_connect::Error::AccountNotFound(_)));
}
