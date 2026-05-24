//! Berlin Group NextGenPSD2 PISP integration test.
//!
//! Wires the vendor-neutral [`PaymentInitiationService`] over a mock
//! Berlin Group transport and exercises the endpoint builder for the
//! various `payment-product` segments.

use op_core::{Currency, Money};
use op_openbanking::aisp::ConsentId;
use op_openbanking::berlin_group::{BerlinPaymentProduct, BerlinService};
use op_openbanking::fapi::OAuth2Token;
use op_openbanking::pisp::{
    PaymentInitiation, PaymentInitiationService, PaymentInitiationStatus, PaymentKind, PaymentRef,
};
use op_openbanking::{Error, Result};

struct MockBerlinPisp {
    svc: BerlinService,
    product: BerlinPaymentProduct,
}

impl PaymentInitiationService for MockBerlinPisp {
    fn create_consent(
        &self,
        _token: &OAuth2Token,
        payment: &PaymentInitiation,
    ) -> Result<ConsentId> {
        payment.validate()?;
        // Real driver would POST to `payments_endpoint(product)`.
        // We just record that the URL we'd hit looks correct.
        let url = self.svc.payments_endpoint(self.product);
        assert!(url.starts_with("https://"));
        Ok(ConsentId("berlin-cs-1".into()))
    }

    fn submit(
        &self,
        _consent: &ConsentId,
        _token: &OAuth2Token,
        payment: &PaymentInitiation,
    ) -> Result<PaymentRef> {
        payment.validate()?;
        if payment.amount.currency != Currency::EUR {
            return Err(Error::CurrencyMismatch(format!(
                "Berlin Group SCT requires EUR; got {}",
                payment.amount.currency
            )));
        }
        Ok(PaymentRef("pay-1".into()))
    }

    fn status(
        &self,
        _token: &OAuth2Token,
        _payment_ref: &PaymentRef,
    ) -> Result<PaymentInitiationStatus> {
        Ok(PaymentInitiationStatus::AcceptedSettlementInProcess)
    }
}

fn token() -> OAuth2Token {
    OAuth2Token {
        access_token: "tok".into(),
        token_type: "Bearer".into(),
        scopes: vec!["payments".into()],
        expires_in: 600,
        refresh_token: None,
        cert_thumbprint: None,
    }
}

fn payment() -> PaymentInitiation {
    PaymentInitiation {
        kind: PaymentKind::ImmediateSingle,
        debtor_account: "DE89370400440532013000".into(),
        creditor_account: "FR1420041010050500013M02606".into(),
        creditor_name: "Acme EU".into(),
        amount: Money::from_minor(99_99, Currency::EUR),
        end_to_end_id: "BG-2026-001".into(),
        remittance: Some("Invoice 9".into()),
        requested_execution_date: None,
    }
}

#[test]
fn instant_sct_endpoint_is_instant_segment() {
    let svc = BerlinService {
        aspsp_base_url: "https://xs2a.aspsp.example".into(),
        aspsp_id: "AAAA1234".into(),
    };
    assert!(
        svc.payments_endpoint(BerlinPaymentProduct::InstantSepaCreditTransfers)
            .ends_with("/payments/instant-sepa-credit-transfers")
    );
}

#[test]
fn pisp_happy_path_creates_and_submits() {
    let mock = MockBerlinPisp {
        svc: BerlinService {
            aspsp_base_url: "https://xs2a.aspsp.example".into(),
            aspsp_id: "AAAA1234".into(),
        },
        product: BerlinPaymentProduct::SepaCreditTransfers,
    };
    let p = payment();
    let consent = mock.create_consent(&token(), &p).expect("consent");
    let r = mock.submit(&consent, &token(), &p).expect("submit");
    let status = mock.status(&token(), &r).expect("status");
    assert_eq!(status, PaymentInitiationStatus::AcceptedSettlementInProcess);
}

#[test]
fn non_eur_rejected_on_sct() {
    let mock = MockBerlinPisp {
        svc: BerlinService {
            aspsp_base_url: "https://xs2a.aspsp.example".into(),
            aspsp_id: "AAAA1234".into(),
        },
        product: BerlinPaymentProduct::SepaCreditTransfers,
    };
    let mut p = payment();
    p.amount = Money::from_minor(10_000, Currency::GBP);
    let err = mock
        .submit(&ConsentId("c".into()), &token(), &p)
        .expect_err("currency");
    assert!(matches!(err, Error::CurrencyMismatch(_)));
}

#[test]
fn zero_amount_rejected_at_validate() {
    let mock = MockBerlinPisp {
        svc: BerlinService {
            aspsp_base_url: "https://xs2a.aspsp.example".into(),
            aspsp_id: "AAAA1234".into(),
        },
        product: BerlinPaymentProduct::SepaCreditTransfers,
    };
    let mut p = payment();
    p.amount = Money::from_minor(0, Currency::EUR);
    let err = mock.create_consent(&token(), &p).expect_err("validate");
    assert!(matches!(err, Error::PaymentInitiationInvalid { .. }));
}
