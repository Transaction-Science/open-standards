//! String primitives. Operate on Unicode (length is character count,
//! not byte count); case conversion uses Unicode case mapping.

use jouleclaw_cascade::LawfulPrimitive;
use std::sync::Arc;

pub fn primitives() -> Vec<Arc<dyn LawfulPrimitive>> {
    vec![
        Arc::new(Length),
        Arc::new(Uppercase),
        Arc::new(Lowercase),
        Arc::new(Reverse),
        Arc::new(CountWords),
        Arc::new(Contains),
        Arc::new(StartsWith),
        Arc::new(EndsWith),
    ]
}

fn strip_prefix_ci<'a>(q: &'a str, prefix: &str) -> Option<&'a str> {
    let q = q.trim();
    if q.len() < prefix.len() {
        return None;
    }
    let (head, tail) = q.split_at(prefix.len());
    if !head.eq_ignore_ascii_case(prefix) {
        return None;
    }
    let rest = tail.strip_prefix(|c: char| c.is_whitespace())?;
    Some(rest)
}

// ---- length -------------------------------------------------------------

pub struct Length;
impl LawfulPrimitive for Length {
    fn id(&self) -> &str {
        "lawful:strings:length"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "length of")?;
        if rest.is_empty() {
            return None;
        }
        Some(rest.chars().count().to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        50
    }
}

// ---- uppercase / lowercase ---------------------------------------------

pub struct Uppercase;
impl LawfulPrimitive for Uppercase {
    fn id(&self) -> &str {
        "lawful:strings:uppercase"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "uppercase")?;
        if rest.is_empty() {
            return None;
        }
        Some(rest.to_uppercase())
    }
    fn declared_cost_uj(&self) -> u64 {
        50
    }
}

pub struct Lowercase;
impl LawfulPrimitive for Lowercase {
    fn id(&self) -> &str {
        "lawful:strings:lowercase"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "lowercase")?;
        if rest.is_empty() {
            return None;
        }
        Some(rest.to_lowercase())
    }
    fn declared_cost_uj(&self) -> u64 {
        50
    }
}

// ---- reverse ------------------------------------------------------------

pub struct Reverse;
impl LawfulPrimitive for Reverse {
    fn id(&self) -> &str {
        "lawful:strings:reverse"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "reverse")?;
        if rest.is_empty() {
            return None;
        }
        Some(rest.chars().rev().collect())
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

// ---- word count ---------------------------------------------------------

pub struct CountWords;
impl LawfulPrimitive for CountWords {
    fn id(&self) -> &str {
        "lawful:strings:count-words"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        let rest = strip_prefix_ci(query, "count words")?;
        if rest.is_empty() {
            return None;
        }
        Some(rest.split_whitespace().count().to_string())
    }
    fn declared_cost_uj(&self) -> u64 {
        60
    }
}

// ---- contains -----------------------------------------------------------

pub struct Contains;
impl LawfulPrimitive for Contains {
    fn id(&self) -> &str {
        "lawful:strings:contains"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        // Syntax: "does <a> contain <b>"
        let rest = strip_prefix_ci(query, "does")?;
        let lower = rest.to_ascii_lowercase();
        let idx = lower.find(" contain ")?;
        let a = &rest[..idx];
        let b = &rest[idx + " contain ".len()..];
        if a.is_empty() || b.is_empty() {
            return None;
        }
        Some(if a.contains(b) { "true".into() } else { "false".into() })
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

// ---- starts_with / ends_with --------------------------------------------

pub struct StartsWith;
impl LawfulPrimitive for StartsWith {
    fn id(&self) -> &str {
        "lawful:strings:starts-with"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        // Syntax: "does <a> start with <b>"
        let rest = strip_prefix_ci(query, "does")?;
        let lower = rest.to_ascii_lowercase();
        let idx = lower.find(" start with ")?;
        let a = &rest[..idx];
        let b = &rest[idx + " start with ".len()..];
        if a.is_empty() || b.is_empty() {
            return None;
        }
        Some(if a.starts_with(b) { "true".into() } else { "false".into() })
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

pub struct EndsWith;
impl LawfulPrimitive for EndsWith {
    fn id(&self) -> &str {
        "lawful:strings:ends-with"
    }
    fn try_resolve(&self, query: &str) -> Option<String> {
        // Syntax: "does <a> end with <b>"
        let rest = strip_prefix_ci(query, "does")?;
        let lower = rest.to_ascii_lowercase();
        let idx = lower.find(" end with ")?;
        let a = &rest[..idx];
        let b = &rest[idx + " end with ".len()..];
        if a.is_empty() || b.is_empty() {
            return None;
        }
        Some(if a.ends_with(b) { "true".into() } else { "false".into() })
    }
    fn declared_cost_uj(&self) -> u64 {
        70
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn length_works() {
        assert_eq!(Length.try_resolve("length of hello").as_deref(), Some("5"));
        assert_eq!(Length.try_resolve("length of café").as_deref(), Some("4"));
    }

    #[test]
    fn case_works() {
        assert_eq!(Uppercase.try_resolve("uppercase hello").as_deref(), Some("HELLO"));
        assert_eq!(Lowercase.try_resolve("lowercase HELLO").as_deref(), Some("hello"));
    }

    #[test]
    fn reverse_works() {
        assert_eq!(Reverse.try_resolve("reverse abc").as_deref(), Some("cba"));
        assert_eq!(Reverse.try_resolve("reverse ").is_none(), true);
    }

    #[test]
    fn count_words_works() {
        assert_eq!(CountWords.try_resolve("count words the quick brown fox").as_deref(), Some("4"));
        assert_eq!(CountWords.try_resolve("count words single").as_deref(), Some("1"));
    }

    #[test]
    fn contains_works() {
        assert_eq!(Contains.try_resolve("does hello contain ell").as_deref(), Some("true"));
        assert_eq!(Contains.try_resolve("does hello contain xyz").as_deref(), Some("false"));
    }

    #[test]
    fn starts_ends_with() {
        assert_eq!(StartsWith.try_resolve("does hello start with he").as_deref(), Some("true"));
        assert_eq!(StartsWith.try_resolve("does hello start with xy").as_deref(), Some("false"));
        assert_eq!(EndsWith.try_resolve("does hello end with lo").as_deref(), Some("true"));
        assert_eq!(EndsWith.try_resolve("does hello end with hi").as_deref(), Some("false"));
    }

    #[test]
    fn malformed_returns_none() {
        assert!(Length.try_resolve("how long is this").is_none());
        assert!(Contains.try_resolve("does this match").is_none());
        assert!(StartsWith.try_resolve("hello").is_none());
    }

    #[test]
    fn category_count() {
        assert_eq!(primitives().len(), 8);
    }
}
