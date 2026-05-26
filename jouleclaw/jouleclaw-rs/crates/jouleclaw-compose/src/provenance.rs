//! Provenance-as-cache (spec §7.3).
//!
//! Compute a stable content-addressable cache key for a sub-query
//! so two overlapping queries can reuse retrieval + verification
//! work. The key normalizes:
//!
//!   - Sub-query text (lowercase, whitespace-collapsed).
//!   - Required modalities (sorted).
//!   - Target stores (sorted).
//!
//! Hash: SHA-256 over the canonical JSON serialization.

use sha2::{Digest, Sha256};

use jouleclaw_schema::SubQuery;

pub fn cache_key_for_subquery(sub: &SubQuery) -> String {
    let mut modalities: Vec<String> = sub
        .required_modalities
        .iter()
        .map(|m| format!("{m:?}").to_lowercase())
        .collect();
    modalities.sort();
    let mut stores = sub.target_stores.clone();
    stores.sort();

    let canonical = serde_json::json!({
        "text": normalize_text(&sub.text),
        "modalities": modalities,
        "stores": stores,
    });
    let bytes = canonical.to_string();
    let mut hasher = Sha256::new();
    hasher.update(bytes.as_bytes());
    let digest = hasher.finalize();
    hex_encode(&digest)
}

fn normalize_text(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ").to_lowercase()
}

fn hex_encode(bytes: &[u8]) -> String {
    let mut out = String::with_capacity(bytes.len() * 2);
    for b in bytes {
        use std::fmt::Write;
        let _ = write!(out, "{:02x}", b);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use jouleclaw_schema::Modality;

    fn sub(text: &str, modalities: Vec<Modality>, stores: Vec<&str>) -> SubQuery {
        SubQuery {
            sub_id: "q0".into(),
            text: text.into(),
            depends_on: vec![],
            required_modalities: modalities,
            target_stores: stores.into_iter().map(|s| s.into()).collect(),
            priority: 1.0,
            rap_id: "rap".into(),
        }
    }

    #[test]
    fn identical_inputs_produce_identical_keys() {
        let a = sub("capital of France", vec![Modality::Text], vec!["wikidata"]);
        let b = sub("capital of France", vec![Modality::Text], vec!["wikidata"]);
        assert_eq!(cache_key_for_subquery(&a), cache_key_for_subquery(&b));
    }

    #[test]
    fn case_and_whitespace_are_normalized() {
        let a = sub("Capital of France", vec![Modality::Text], vec!["wikidata"]);
        let b = sub("   capital   of    France   ", vec![Modality::Text], vec!["wikidata"]);
        assert_eq!(cache_key_for_subquery(&a), cache_key_for_subquery(&b));
    }

    #[test]
    fn store_order_does_not_matter() {
        let a = sub("x", vec![Modality::Text], vec!["wikidata", "openalex"]);
        let b = sub("x", vec![Modality::Text], vec!["openalex", "wikidata"]);
        assert_eq!(cache_key_for_subquery(&a), cache_key_for_subquery(&b));
    }

    #[test]
    fn different_text_produces_different_keys() {
        let a = sub("capital of France", vec![Modality::Text], vec!["wikidata"]);
        let b = sub("capital of Germany", vec![Modality::Text], vec!["wikidata"]);
        assert_ne!(cache_key_for_subquery(&a), cache_key_for_subquery(&b));
    }

    #[test]
    fn key_is_64_hex_chars() {
        let s = sub("anything", vec![Modality::Text], vec!["x"]);
        let k = cache_key_for_subquery(&s);
        assert_eq!(k.len(), 64);
        assert!(k.chars().all(|c| c.is_ascii_hexdigit()));
    }
}
