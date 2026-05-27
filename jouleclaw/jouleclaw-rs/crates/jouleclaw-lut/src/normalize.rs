//! Input normalisation.
//!
//! The LUT MUST normalise identically on `register` and `try_lookup` so
//! casing/whitespace variations of the same query hit the same entry.
//! Specifically:
//!
//! - leading + trailing ASCII whitespace is trimmed
//! - runs of internal ASCII whitespace collapse to a single space
//! - ASCII letters are lower-cased; non-ASCII characters preserve case
//!
//! Examples:
//! ```text
//! "  GCD 12 8  "    → "gcd 12 8"
//! "gcd\t12\n  8"    → "gcd 12 8"
//! "GCD 12 8"        → "gcd 12 8"
//! "Δ Time"          → "Δ time"  // Δ is non-ASCII, case preserved
//! ```

/// Normalise an input string for LUT hashing.
///
/// See the module docs for the exact contract.
pub fn normalize(input: &str) -> String {
    let trimmed = input.trim();
    let mut out = String::with_capacity(trimmed.len());
    let mut last_was_space = false;
    for ch in trimmed.chars() {
        if ch.is_ascii_whitespace() {
            if !last_was_space {
                out.push(' ');
                last_was_space = true;
            }
        } else {
            last_was_space = false;
            if ch.is_ascii_uppercase() {
                out.push(ch.to_ascii_lowercase());
            } else {
                out.push(ch);
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn trims_and_lowercases() {
        assert_eq!(normalize("  GCD 12 8  "), "gcd 12 8");
    }

    #[test]
    fn collapses_internal_whitespace() {
        assert_eq!(normalize("gcd\t12\n  8"), "gcd 12 8");
    }

    #[test]
    fn already_normalized_is_stable() {
        assert_eq!(normalize("gcd 12 8"), "gcd 12 8");
    }

    #[test]
    fn empty_and_whitespace_only() {
        assert_eq!(normalize(""), "");
        assert_eq!(normalize("   \t\n  "), "");
    }

    #[test]
    fn preserves_unicode_case() {
        // Δ (U+0394) is non-ASCII, so its case must be preserved.
        assert_eq!(normalize("Δ Time"), "Δ time");
    }

    #[test]
    fn whitespace_variants_collide() {
        assert_eq!(normalize("  GCD 12 8  "), normalize("gcd 12 8"));
    }
}
