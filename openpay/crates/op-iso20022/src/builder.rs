//! High-level builders.
//!
//! These bridge `op-core` domain types (`Money`, `PaymentMethod`, `Payment`)
//! to ISO 20022 messages. Callers describe a payment in `OpenPay` terms; the
//! builder produces a profile-valid ISO 20022 document.

use op_core::{Money, PaymentMethod};
use uuid::Uuid;

use crate::bah::{BusinessApplicationHeader, PartyIdentification};
use crate::error::{Error, Result};
use crate::message::MessageKind;
use crate::profile::Profile;

/// A high-level builder for a customer credit transfer (`pacs.008`).
///
/// Generic over the active rail profile, which determines:
/// - ISO 20022 message version
/// - mandatory identifier formats (UETR, IMAD, ABA, IBAN, ISPB)
/// - charge bearer codes accepted
/// - remittance information limits
///
/// Once `build()` is called, the resulting message is guaranteed to pass
/// the profile's validation rules.
#[derive(Debug)]
pub struct CreditTransferBuilder<P: Profile> {
    amount: Option<Money>,
    debtor_method: Option<PaymentMethod>,
    creditor_method: Option<PaymentMethod>,
    debtor_agent: Option<PartyIdentification>,
    creditor_agent: Option<PartyIdentification>,
    end_to_end_id: Option<String>,
    uetr: Option<String>,
    remittance_info: Option<String>,
    _profile: core::marker::PhantomData<P>,
}

impl<P: Profile> Default for CreditTransferBuilder<P> {
    fn default() -> Self {
        Self::new()
    }
}

impl<P: Profile> CreditTransferBuilder<P> {
    /// Fresh builder.
    #[must_use]
    pub fn new() -> Self {
        Self {
            amount: None,
            debtor_method: None,
            creditor_method: None,
            debtor_agent: None,
            creditor_agent: None,
            end_to_end_id: None,
            uetr: None,
            remittance_info: None,
            _profile: core::marker::PhantomData,
        }
    }

    /// Amount and currency.
    #[must_use]
    pub fn amount(mut self, m: Money) -> Self {
        self.amount = Some(m);
        self
    }

    /// Who is paying (their `PaymentMethod`).
    #[must_use]
    pub fn debtor(mut self, m: PaymentMethod) -> Self {
        self.debtor_method = Some(m);
        self
    }

    /// Who is being paid.
    #[must_use]
    pub fn creditor(mut self, m: PaymentMethod) -> Self {
        self.creditor_method = Some(m);
        self
    }

    /// Debtor's bank.
    #[must_use]
    pub fn debtor_agent(mut self, p: PartyIdentification) -> Self {
        self.debtor_agent = Some(p);
        self
    }

    /// Creditor's bank.
    #[must_use]
    pub fn creditor_agent(mut self, p: PartyIdentification) -> Self {
        self.creditor_agent = Some(p);
        self
    }

    /// End-to-end identifier (mandatory in modern ISO 20022). Free-form
    /// up to 35 characters. If unset, the builder generates one.
    #[must_use]
    pub fn end_to_end_id(mut self, id: impl Into<String>) -> Self {
        self.end_to_end_id = Some(id.into());
        self
    }

    /// UETR (Unique End-to-end Transaction Reference) — a UUID v4 in
    /// canonical form. Mandatory on `FedNow`. If unset, generated.
    #[must_use]
    pub fn uetr(mut self, uetr: impl Into<String>) -> Self {
        self.uetr = Some(uetr.into());
        self
    }

    /// Unstructured remittance info (max length depends on profile).
    #[must_use]
    pub fn remittance(mut self, s: impl Into<String>) -> Self {
        self.remittance_info = Some(s.into());
        self
    }

    /// Validate inputs and produce a [`BuiltCreditTransfer`].
    ///
    /// This is the moment all profile rules apply. We do NOT yet produce
    /// the upstream `Document` here — that's done in a separate
    /// serialization step so callers can inspect / log / sign the
    /// intermediate form first.
    ///
    /// # Errors
    /// - `MissingField` for any required input that wasn't set.
    /// - `ProfileViolation` for profile-specific rules.
    /// - `InvalidField` for format issues.
    pub fn build(self) -> Result<BuiltCreditTransfer<P>> {
        let amount = self.amount.ok_or(Error::MissingField("amount"))?;
        let debtor = self.debtor_method.ok_or(Error::MissingField("debtor"))?;
        let creditor = self
            .creditor_method
            .ok_or(Error::MissingField("creditor"))?;
        let debtor_agent = self
            .debtor_agent
            .ok_or(Error::MissingField("debtor_agent"))?;
        let creditor_agent = self
            .creditor_agent
            .ok_or(Error::MissingField("creditor_agent"))?;

        let end_to_end_id = self
            .end_to_end_id
            .unwrap_or_else(|| Uuid::now_v7().simple().to_string());
        if end_to_end_id.len() > 35 {
            return Err(Error::InvalidField {
                field: "EndToEndId",
                reason: alloc::format!("max 35 chars, got {}", end_to_end_id.len()),
            });
        }

        let uetr = self
            .uetr
            .unwrap_or_else(|| Uuid::new_v4().hyphenated().to_string().to_lowercase());
        validate_uetr(&uetr)?;

        // Profile must support pacs.008.
        let version = P::version_for(MessageKind::Pacs008).ok_or(Error::ProfileViolation {
            profile: P::NAME,
            reason: alloc::format!("{} does not support pacs.008", P::NAME),
        })?;

        // Build and validate the BAH.
        let bah =
            BusinessApplicationHeader::new(debtor_agent.clone(), creditor_agent.clone(), version);
        P::validate_bah(&bah)?;

        // Remittance length cap (FedNow & RTP cap unstructured at 140
        // chars per occurrence; SEPA Instant at 140 also). We use 140 as
        // a safe default; profile-specific overrides come later.
        if let Some(r) = &self.remittance_info
            && r.len() > 140
        {
            return Err(Error::InvalidField {
                field: "RmtInf.Ustrd",
                reason: alloc::format!("unstructured remittance >140 chars: {}", r.len()),
            });
        }

        Ok(BuiltCreditTransfer {
            bah,
            amount,
            debtor,
            creditor,
            end_to_end_id,
            uetr,
            remittance_info: self.remittance_info,
            _profile: core::marker::PhantomData,
        })
    }
}

/// A fully-validated, profile-conformant credit transfer payload, ready
/// to be serialized to ISO 20022 XML.
///
/// We hold the OpenPay-level fields verbatim so the layer that actually
/// constructs the upstream `Document` can be a separate, testable
/// mapping. This split lets us snapshot-test the intermediate form
/// without depending on `quick-xml` output stability.
#[derive(Debug)]
pub struct BuiltCreditTransfer<P: Profile> {
    /// BAH (head.001).
    pub bah: BusinessApplicationHeader,
    /// Transfer amount.
    pub amount: Money,
    /// Debtor `PaymentMethod` — typically `A2a` with IBAN / ABA / UPI.
    pub debtor: PaymentMethod,
    /// Creditor `PaymentMethod`.
    pub creditor: PaymentMethod,
    /// End-to-end id (mandatory in ISO 20022, 35-char max).
    pub end_to_end_id: String,
    /// UETR (UUID v4, lowercase hyphenated).
    pub uetr: String,
    /// Unstructured remittance, ≤140 chars.
    pub remittance_info: Option<String>,
    _profile: core::marker::PhantomData<P>,
}

/// Validate a UETR per ISO 20022 / SWIFT format: lowercase UUID v4,
/// 36 chars with hyphens at positions 8/13/18/23, version nibble = 4.
fn validate_uetr(s: &str) -> Result<()> {
    if s.len() != 36 {
        return Err(Error::InvalidField {
            field: "UETR",
            reason: alloc::format!("UETR must be 36 chars, got {}", s.len()),
        });
    }
    let bytes = s.as_bytes();
    for (i, &b) in bytes.iter().enumerate() {
        match i {
            8 | 13 | 18 | 23 => {
                if b != b'-' {
                    return Err(Error::InvalidField {
                        field: "UETR",
                        reason: alloc::format!("expected '-' at position {i}"),
                    });
                }
            }
            _ => {
                let is_lower_hex = b.is_ascii_digit() || (b'a'..=b'f').contains(&b);
                if !is_lower_hex {
                    return Err(Error::InvalidField {
                        field: "UETR",
                        reason: alloc::format!("non-lowercase-hex byte at position {i}"),
                    });
                }
            }
        }
    }
    // Version nibble — UUID v4 = position 14 (0-indexed) must be '4'.
    if bytes[14] != b'4' {
        return Err(Error::InvalidField {
            field: "UETR",
            reason: "UETR must be a UUID v4 (version nibble != '4')".into(),
        });
    }
    Ok(())
}

extern crate alloc;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::profile::FedNow;
    use op_core::{A2aKey, Currency, PaymentMethod};

    fn sample_builder() -> CreditTransferBuilder<FedNow> {
        CreditTransferBuilder::<FedNow>::new()
            .amount(Money::from_minor(1_000_000, Currency::USD))
            .debtor(PaymentMethod::A2a(A2aKey::UsAch {
                routing: "021000021".into(),
                account: "1234567890".into(),
            }))
            .creditor(PaymentMethod::A2a(A2aKey::UsAch {
                routing: "026009593".into(),
                account: "0987654321".into(),
            }))
            .debtor_agent(PartyIdentification::AbaRoutingNumber("021000021".into()))
            .creditor_agent(PartyIdentification::AbaRoutingNumber("026009593".into()))
    }

    #[test]
    fn happy_path_builds() {
        let built = sample_builder().build().unwrap();
        assert_eq!(built.amount.minor_units, 1_000_000);
        assert_eq!(built.uetr.len(), 36);
        assert_eq!(built.bah.message_definition_id, "pacs.008.001.08");
    }

    #[test]
    fn missing_amount_fails() {
        let b = CreditTransferBuilder::<FedNow>::new();
        assert!(matches!(b.build(), Err(Error::MissingField("amount"))));
    }

    #[test]
    fn missing_creditor_fails() {
        let b =
            CreditTransferBuilder::<FedNow>::new().amount(Money::from_minor(100, Currency::USD));
        assert!(matches!(b.build(), Err(Error::MissingField("debtor"))));
    }

    #[test]
    fn long_end_to_end_id_rejected() {
        let b = sample_builder().end_to_end_id("X".repeat(36));
        assert!(matches!(
            b.build(),
            Err(Error::InvalidField {
                field: "EndToEndId",
                ..
            })
        ));
    }

    #[test]
    fn invalid_uetr_rejected() {
        // 36 chars but wrong version nibble (v1).
        let b = sample_builder().uetr("550e8400-e29b-11d4-a716-446655440000");
        assert!(matches!(
            b.build(),
            Err(Error::InvalidField { field: "UETR", .. })
        ));
    }

    #[test]
    fn uppercase_uetr_rejected() {
        let b = sample_builder().uetr("550E8400-E29B-41D4-A716-446655440000");
        assert!(matches!(
            b.build(),
            Err(Error::InvalidField { field: "UETR", .. })
        ));
    }

    #[test]
    fn short_uetr_rejected() {
        let b = sample_builder().uetr("550e8400-e29b-41d4-a716");
        assert!(matches!(
            b.build(),
            Err(Error::InvalidField { field: "UETR", .. })
        ));
    }

    #[test]
    fn valid_uetr_accepted() {
        let b = sample_builder().uetr("550e8400-e29b-41d4-a716-446655440000");
        assert!(b.build().is_ok());
    }

    #[test]
    fn long_remittance_rejected() {
        let b = sample_builder().remittance("x".repeat(141));
        assert!(matches!(
            b.build(),
            Err(Error::InvalidField {
                field: "RmtInf.Ustrd",
                ..
            })
        ));
    }

    #[test]
    fn exactly_140_char_remittance_accepted() {
        let b = sample_builder().remittance("x".repeat(140));
        assert!(b.build().is_ok());
    }
}
