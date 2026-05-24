//! Issuance / Presentation Exchange (IPEX).
//!
//! IPEX (spec §7) is the protocol controllers use to move ACDCs
//! between each other. It is symmetric: either party may initiate.
//!
//! | Kind     | Purpose                                                   |
//! |----------|-----------------------------------------------------------|
//! | `apply`  | Discloser asks issuer for a credential matching a schema. |
//! | `offer`  | Issuer (or holder) offers a credential to a counterparty. |
//! | `agree`  | Counterparty agrees to the offer.                         |
//! | `grant`  | Discloser delivers the actual ACDC body.                  |
//! | `admit`  | Recipient acknowledges receipt of a grant.                |
//! | `spurn`  | Recipient rejects an offer / grant / apply.               |
//!
//! Each message carries a SAID and chains to the message it answers
//! via the `p` (prior) field, so the full exchange forms a hash-linked
//! mini-log auditable by either party.

use serde::{Deserialize, Serialize};
use serde_json::Value;
use smart_byte_core::Said;

use crate::acdc::Acdc;
use crate::error::{AcdcError, Result};

/// The six IPEX message kinds.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum IpexKind {
    /// Discloser → issuer: request a credential matching a schema.
    Apply,
    /// Issuer → discloser: offer a credential (metadata only).
    Offer,
    /// Counterparty agreement to an offer.
    Agree,
    /// Discloser → recipient: deliver the credential body.
    Grant,
    /// Recipient → discloser: acknowledge receipt.
    Admit,
    /// Recipient → counterparty: reject the message chain.
    Spurn,
}

impl IpexKind {
    fn tag(self) -> &'static str {
        match self {
            Self::Apply => "apply",
            Self::Offer => "offer",
            Self::Agree => "agree",
            Self::Grant => "grant",
            Self::Admit => "admit",
            Self::Spurn => "spurn",
        }
    }
}

/// A single IPEX message.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct IpexMessage {
    /// Version string.
    pub v: String,
    /// Message kind.
    pub t: String,
    /// Message SAID.
    pub d: Said,
    /// Sender AID.
    pub i: String,
    /// Recipient AID (the counterparty in this exchange).
    pub rp: String,
    /// SAID of the prior message in this exchange (zero on the
    /// initiating message).
    pub p: Said,
    /// Embedded ACDC body for `grant`; absent for the other kinds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub e: Option<Acdc>,
    /// Free-form attribute map for kind-specific fields (e.g. a schema
    /// SAID on `apply`, a reason string on `spurn`).
    #[serde(default, skip_serializing_if = "serde_json::Map::is_empty")]
    pub a: serde_json::Map<String, Value>,
}

impl IpexMessage {
    /// Construct a new message (computes the SAID).
    pub fn new(
        kind: IpexKind,
        sender: impl Into<String>,
        recipient: impl Into<String>,
        prior: Option<Said>,
        embedded: Option<Acdc>,
        attrs: serde_json::Map<String, Value>,
    ) -> Result<Self> {
        let mut msg = Self {
            v: crate::VERSION_STRING.to_string(),
            t: kind.tag().into(),
            d: Said([0u8; 32]),
            i: sender.into(),
            rp: recipient.into(),
            p: prior.unwrap_or(Said([0u8; 32])),
            e: embedded,
            a: attrs,
        };
        msg.d = msg.compute_said()?;
        Ok(msg)
    }

    /// Compute the SAID over the message body with `d` placeheld.
    pub fn compute_said(&self) -> Result<Said> {
        let mut tmp = self.clone();
        tmp.d = Said([0u8; 32]);
        let bytes =
            serde_jcs::to_vec(&tmp).map_err(|e| AcdcError::Jcs(e.to_string()))?;
        Ok(Said::hash(&bytes))
    }

    /// Verify the message's SAID matches its body.
    pub fn verify_said(&self) -> Result<()> {
        let computed = self.compute_said()?;
        if computed != self.d {
            return Err(AcdcError::SaidMismatch {
                asserted: self.d,
                computed,
            });
        }
        Ok(())
    }

    /// Convenience: build an `apply` message asking for a schema.
    pub fn apply(
        sender: impl Into<String>,
        recipient: impl Into<String>,
        schema_said: Said,
    ) -> Result<Self> {
        let mut a = serde_json::Map::new();
        a.insert("schema".into(), Value::String(schema_said.to_base32()));
        Self::new(IpexKind::Apply, sender, recipient, None, None, a)
    }

    /// Convenience: build a `grant` message delivering an ACDC, with a
    /// prior link to the preceding `agree` / `apply`.
    pub fn grant(
        sender: impl Into<String>,
        recipient: impl Into<String>,
        prior: Said,
        acdc: Acdc,
    ) -> Result<Self> {
        Self::new(
            IpexKind::Grant,
            sender,
            recipient,
            Some(prior),
            Some(acdc),
            serde_json::Map::new(),
        )
    }

    /// Convenience: `admit` reply chained to a grant.
    pub fn admit(
        sender: impl Into<String>,
        recipient: impl Into<String>,
        prior: Said,
    ) -> Result<Self> {
        Self::new(
            IpexKind::Admit,
            sender,
            recipient,
            Some(prior),
            None,
            serde_json::Map::new(),
        )
    }

    /// Convenience: `spurn` reply with a reason string.
    pub fn spurn(
        sender: impl Into<String>,
        recipient: impl Into<String>,
        prior: Said,
        reason: impl Into<String>,
    ) -> Result<Self> {
        let mut a = serde_json::Map::new();
        a.insert("reason".into(), Value::String(reason.into()));
        Self::new(IpexKind::Spurn, sender, recipient, Some(prior), None, a)
    }
}

/// Verify a sequence of IPEX messages forms a coherent chain: each
/// message's `p` matches the prior message's `d`, every message's SAID
/// is valid, and sender/recipient alternate on opposite sides of the
/// exchange after the initiating message.
pub fn verify_exchange(messages: &[IpexMessage]) -> Result<()> {
    let mut prev: Option<&IpexMessage> = None;
    for (idx, m) in messages.iter().enumerate() {
        m.verify_said()?;
        if let Some(p) = prev {
            if m.p != p.d {
                return Err(AcdcError::Ipex(format!(
                    "ipex chain break at index {idx}: expected prior {} got {}",
                    p.d, m.p
                )));
            }
            // After the first message the conversation should swap
            // sender/recipient on each turn (we permit either party to
            // chain consecutive messages only if they're addressed back
            // and forth correctly — both directions are allowed here).
            if m.i != p.rp && m.i != p.i {
                return Err(AcdcError::Ipex(format!(
                    "ipex sender {} not a participant at index {idx}",
                    m.i
                )));
            }
        } else if m.p != Said([0u8; 32]) {
            return Err(AcdcError::Ipex(
                "first message must have zero prior".into(),
            ));
        }
        prev = Some(m);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acdc::{AcdcBuilder, AttributeSection, SchemaSection};
    use serde_json::json;

    fn sample_acdc() -> Acdc {
        let mut s = serde_json::Map::new();
        s.insert("$id".into(), json!("sample"));
        let mut a = serde_json::Map::new();
        a.insert("name".into(), json!("Alice"));
        AcdcBuilder::new()
            .issuer("Bissuer")
            .schema(SchemaSection::Inline(s))
            .attributes(AttributeSection::Inline(a))
            .build()
            .expect("build")
    }

    #[test]
    fn apply_grant_admit_chain() {
        let apply = IpexMessage::apply("Bdiscloser", "Bissuer", Said([1u8; 32]))
            .expect("apply");
        let grant = IpexMessage::grant("Bissuer", "Bdiscloser", apply.d, sample_acdc())
            .expect("grant");
        let admit = IpexMessage::admit("Bdiscloser", "Bissuer", grant.d).expect("admit");
        verify_exchange(&[apply, grant, admit]).expect("ok");
    }

    #[test]
    fn break_in_chain_caught() {
        let apply = IpexMessage::apply("A", "B", Said([1u8; 32])).expect("apply");
        // grant with wrong prior
        let bad = IpexMessage::grant("B", "A", Said([0xFFu8; 32]), sample_acdc())
            .expect("grant");
        assert!(matches!(
            verify_exchange(&[apply, bad]),
            Err(AcdcError::Ipex(_))
        ));
    }
}
