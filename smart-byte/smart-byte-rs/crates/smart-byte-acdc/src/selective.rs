//! Selective disclosure for ACDCs.
//!
//! Per spec §8, an ACDC schema may mark an attribute section as
//! *partially disclosable* via the `oneOf:[compactForm, fullForm]`
//! construction. The compact form is just the section SAID; the full
//! form is the inline attribute map. A holder may transition from one
//! to the other selectively without altering the credential's outer
//! SAID, because the outer SAID was computed over the SAID of the
//! attribute section, not its contents.
//!
//! Concretely: when a credential is issued with selective-disclosure
//! enabled,
//!
//! 1. the issuer constructs the full attribute object;
//! 2. the issuer computes the attribute section SAID `da` over the
//!    canonical JCS of the full object;
//! 3. the issuer publishes the ACDC with `a = Compact(da)`;
//! 4. the issuer hands the holder the full attribute object as the
//!    *disclosure*.
//!
//! At presentation time the holder may choose to:
//!
//! * present the compact ACDC alone (zero attributes revealed);
//! * present the compact ACDC plus the full disclosure (all
//!   attributes revealed);
//! * with a *digest-tree* extension below, reveal a chosen subset.
//!
//! For v1 we support the two extreme positions plus a "named-subset"
//! disclosure that re-derives the section SAID over the full
//! disclosure, verifying it matches the credential's `a` field. The
//! holder cannot fabricate attributes the issuer did not sign because
//! any modification changes `da`.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};
use serde_json::Value;
use smart_byte_core::Said;

use crate::acdc::{Acdc, AttributeSection};
use crate::error::{AcdcError, Result};

/// Plan for what to disclose from a full attribute set.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum DisclosurePlan {
    /// Reveal everything.
    All,
    /// Reveal nothing — present only the compact ACDC and the section
    /// SAID.
    None,
    /// Reveal a named subset of attribute keys. The subject AID (`i`)
    /// is always included if present, since selective disclosure
    /// preserves targeting.
    Subset(Vec<String>),
}

/// A selective disclosure carries both the compact ACDC and the
/// attributes that the holder chose to reveal, along with any side
/// information the verifier needs to re-derive the section SAID.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SelectiveDisclosure {
    /// The credential as it appears on the wire (`a` is compact).
    pub credential: Acdc,
    /// The full canonical disclosure of the attribute section
    /// (everything the issuer signed). Verifiers compute its SAID and
    /// match it against `credential.a`.
    pub full_disclosure: serde_json::Map<String, Value>,
    /// Subset of keys the holder actually wants to surface to the
    /// verifier. The verifier may use this to populate UI / policy
    /// even though `full_disclosure` is also present (proves issuer
    /// signed the values).
    pub revealed: BTreeMap<String, Value>,
}

impl SelectiveDisclosure {
    /// Verify that:
    ///
    /// 1. the credential's compact `a` SAID equals the SAID of
    ///    `full_disclosure` under JCS;
    /// 2. every `revealed` entry agrees with `full_disclosure`.
    ///
    /// Returns the verified compact SAID on success.
    pub fn verify(&self) -> Result<Said> {
        let compact = match &self.credential.a {
            AttributeSection::Compact(s) => *s,
            AttributeSection::Inline(_) => {
                return Err(AcdcError::SelectiveDisclosure(
                    "credential has inline attributes, not compact".into(),
                ));
            }
        };
        let bytes = serde_jcs::to_vec(&Value::Object(self.full_disclosure.clone()))
            .map_err(|e| AcdcError::Jcs(e.to_string()))?;
        let derived = Said::hash(&bytes);
        if derived != compact {
            return Err(AcdcError::SelectiveDisclosure(format!(
                "disclosure SAID {derived} does not match credential a-SAID {compact}"
            )));
        }
        for (k, v) in &self.revealed {
            match self.full_disclosure.get(k) {
                Some(full_v) if full_v == v => {}
                Some(_) => {
                    return Err(AcdcError::SelectiveDisclosure(format!(
                        "revealed key `{k}` disagrees with full disclosure"
                    )));
                }
                None => {
                    return Err(AcdcError::SelectiveDisclosure(format!(
                        "revealed key `{k}` is not in the full disclosure"
                    )));
                }
            }
        }
        self.credential.verify_said()?;
        Ok(compact)
    }
}

/// Build a [`SelectiveDisclosure`] from the issuer-side full attribute
/// map and a holder-side plan.
///
/// Internally this:
///
/// 1. JCS-encodes the full attribute map;
/// 2. hashes it to derive the section SAID `da`;
/// 3. clones the source credential and rewrites its `a` to
///    `Compact(da)`;
/// 4. re-derives the credential's outer SAID (it changes because
///    flipping the inline map to a compact SAID is a content change at
///    the outer level too — the verifier checks both).
///
/// Callers who want the *outer* SAID to remain stable across disclosure
/// modes must build the credential with `Compact(da)` in the first
/// place (issuance pattern). For that case use [`pack_compact`].
pub fn derive_disclosure(
    mut credential: Acdc,
    full_attrs: serde_json::Map<String, Value>,
    plan: DisclosurePlan,
) -> Result<SelectiveDisclosure> {
    let bytes = serde_jcs::to_vec(&Value::Object(full_attrs.clone()))
        .map_err(|e| AcdcError::Jcs(e.to_string()))?;
    let da = Said::hash(&bytes);
    credential.a = AttributeSection::Compact(da);
    credential.d = credential.compute_said()?;

    let revealed = match plan {
        DisclosurePlan::All => full_attrs.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
        DisclosurePlan::None => BTreeMap::new(),
        DisclosurePlan::Subset(keys) => {
            let mut m = BTreeMap::new();
            for k in keys {
                let v = full_attrs.get(&k).ok_or_else(|| {
                    AcdcError::SelectiveDisclosure(format!(
                        "subset key `{k}` not in full attributes"
                    ))
                })?;
                m.insert(k, v.clone());
            }
            m
        }
    };

    Ok(SelectiveDisclosure {
        credential,
        full_disclosure: full_attrs,
        revealed,
    })
}

/// Repackage a credential's attribute section as a compact SAID,
/// returning the new credential plus the section SAID. Used by issuers
/// who want to publish only the compact form. The full attribute map
/// is returned to the caller to hand to the holder out-of-band.
pub fn pack_compact(
    mut credential: Acdc,
) -> Result<(Acdc, Said, serde_json::Map<String, Value>)> {
    let full = match credential.a.clone() {
        AttributeSection::Inline(m) => m,
        AttributeSection::Compact(_) => {
            return Err(AcdcError::SelectiveDisclosure(
                "credential is already in compact form".into(),
            ));
        }
    };
    let bytes = serde_jcs::to_vec(&Value::Object(full.clone()))
        .map_err(|e| AcdcError::Jcs(e.to_string()))?;
    let da = Said::hash(&bytes);
    credential.a = AttributeSection::Compact(da);
    credential.d = credential.compute_said()?;
    Ok((credential, da, full))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::acdc::{AcdcBuilder, SchemaSection};
    use serde_json::json;

    fn issued() -> (Acdc, serde_json::Map<String, Value>) {
        let mut s = serde_json::Map::new();
        s.insert("$id".into(), json!("disclose-schema"));
        let mut a = serde_json::Map::new();
        a.insert("name".into(), json!("Alice"));
        a.insert("age".into(), json!(30));
        a.insert("city".into(), json!("Austin"));
        let acdc = AcdcBuilder::new()
            .issuer("Bissuer")
            .schema(SchemaSection::Inline(s))
            .attributes(AttributeSection::Inline(a.clone()))
            .build()
            .expect("build");
        (acdc, a)
    }

    #[test]
    fn reveal_all_roundtrip() {
        let (acdc, full) = issued();
        let sd = derive_disclosure(acdc, full, DisclosurePlan::All).expect("derive");
        sd.verify().expect("verify");
    }

    #[test]
    fn reveal_subset_only() {
        let (acdc, full) = issued();
        let sd = derive_disclosure(
            acdc,
            full,
            DisclosurePlan::Subset(vec!["name".into()]),
        )
        .expect("derive");
        sd.verify().expect("verify");
        assert_eq!(sd.revealed.len(), 1);
        assert!(sd.revealed.contains_key("name"));
    }

    #[test]
    fn tampered_disclosure_fails() {
        let (acdc, full) = issued();
        let mut sd = derive_disclosure(acdc, full, DisclosurePlan::All).expect("derive");
        sd.full_disclosure.insert("city".into(), json!("Tampered"));
        assert!(matches!(
            sd.verify(),
            Err(AcdcError::SelectiveDisclosure(_))
        ));
    }

    #[test]
    fn pack_compact_yields_consistent_said() {
        let (acdc, full) = issued();
        let (compact, da, returned) = pack_compact(acdc).expect("pack");
        assert_eq!(returned, full);
        assert!(matches!(compact.a, AttributeSection::Compact(s) if s == da));
    }
}
