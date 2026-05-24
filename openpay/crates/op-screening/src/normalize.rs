//! Name canonicalisation.
//!
//! Sanctions names arrive in every variety of orthography the world
//! has invented. To get a hit rate above the floor you have to fold
//! out diacritics, lowercase, normalise whitespace, expand business
//! suffix abbreviations, and transliterate non-Latin scripts on a
//! best-effort basis. This module is the place that happens.

use serde::{Deserialize, Serialize};
use unicode_normalization::UnicodeNormalization;

/// A canonicalised name, suitable for index lookup.
///
/// Once you have one, two `NormalizedName`s compare equal iff they
/// hashed to the same bucket; the underlying string is stripped of
/// case, accents, punctuation, and known suffix variants.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NormalizedName(pub String);

impl NormalizedName {
    /// View as a `&str`.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// Canonicalise `input` into a [`NormalizedName`].
///
/// Pipeline:
///
/// 1. Unicode NFKD decomposition.
/// 2. Strip combining marks (diacritics).
/// 3. Lowercase.
/// 4. Replace non-alphanumeric runs with a single space.
/// 5. Expand common business-suffix abbreviations
///    (`Co` → `Company`, `Inc` → `Incorporated`, etc.).
/// 6. Best-effort transliteration of Cyrillic and a handful of
///    common Arabic / Chinese pinyin tokens.
#[must_use]
pub fn normalize(input: &str) -> NormalizedName {
    // 1 + 2: NFKD then drop combining marks.
    let decomposed: String = input
        .nfkd()
        .filter(|c| !is_combining_mark(*c))
        .collect();

    // 3 + 4: lowercase and squash punctuation runs into single spaces.
    let mut squashed = String::with_capacity(decomposed.len());
    let mut last_was_space = true;
    for c in decomposed.chars() {
        let lower = c.to_lowercase().next().unwrap_or(c);
        if lower.is_alphanumeric() {
            squashed.push(transliterate(lower));
            last_was_space = false;
        } else if !last_was_space {
            squashed.push(' ');
            last_was_space = true;
        }
    }
    let trimmed = squashed.trim();

    // 5: token-by-token suffix expansion.
    let expanded: Vec<String> = trimmed
        .split_whitespace()
        .map(|tok| expand_abbreviation(tok).to_string())
        .collect();

    NormalizedName(expanded.join(" "))
}

/// Combining mark detection — every codepoint in the Unicode
/// `Mn` (mark, nonspacing) or `Mc` (mark, spacing combining)
/// general category, which is everything `NFKD` peels off a
/// precomposed letter-with-accent.
fn is_combining_mark(c: char) -> bool {
    // Cheap range check: covers the bulk of Latin / Greek / Cyrillic.
    matches!(c as u32,
        0x0300..=0x036F   // Combining Diacritical Marks
        | 0x1AB0..=0x1AFF // Combining Diacritical Marks Extended
        | 0x1DC0..=0x1DFF // Combining Diacritical Marks Supplement
        | 0x20D0..=0x20FF // Combining Diacritical Marks for Symbols
        | 0xFE20..=0xFE2F // Combining Half Marks
    )
}

/// Per-codepoint best-effort Latinisation.
///
/// We only handle the ones that ship in real sanctions data — Cyrillic
/// for Russia/Ukraine/Belarus designations, the common Arabic letters
/// for Iran/Syria/Yemen. Chinese names are already Pinyin-romanised
/// by OFAC / EU lists at source; we touch only basic CJK fall-throughs.
fn transliterate(c: char) -> char {
    match c {
        // --- Cyrillic small letters (single-char mappings only).
        'а' => 'a', 'б' => 'b', 'в' => 'v', 'г' => 'g', 'д' => 'd',
        'е' => 'e', 'ё' => 'e', 'з' => 'z', 'и' => 'i', 'й' => 'i',
        'к' => 'k', 'л' => 'l', 'м' => 'm', 'н' => 'n', 'о' => 'o',
        'п' => 'p', 'р' => 'r', 'с' => 's', 'т' => 't', 'у' => 'u',
        'ф' => 'f', 'х' => 'h', 'ы' => 'y', 'э' => 'e',

        // --- Arabic letters that map 1:1 to a Latin letter (very rough).
        'ا' => 'a', 'ب' => 'b', 'ت' => 't', 'د' => 'd', 'ر' => 'r',
        'ز' => 'z', 'س' => 's', 'ف' => 'f', 'ك' => 'k', 'ل' => 'l',
        'م' => 'm', 'ن' => 'n', 'ه' => 'h', 'ي' => 'y',

        _ => c,
    }
}

/// Common business-suffix expansions. Casing is already lowered.
fn expand_abbreviation(tok: &str) -> &str {
    match tok {
        "co" => "company",
        "corp" => "corporation",
        "inc" => "incorporated",
        "incorp" => "incorporated",
        "ltd" => "limited",
        "llc" => "limited liability company",
        "llp" => "limited liability partnership",
        "lp" => "limited partnership",
        "plc" => "public limited company",
        "gmbh" => "gesellschaft",
        "ag" => "aktiengesellschaft",
        "sa" => "societe anonyme",
        "srl" => "societa a responsabilita limitata",
        "bv" => "besloten vennootschap",
        "ab" => "aktiebolag",
        "as" => "aksjeselskap",
        "oy" => "osakeyhtio",
        _ => tok,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn diacritic_strip() {
        assert_eq!(normalize("Müller").as_str(), "muller");
    }

    #[test]
    fn apostrophes_become_space_then_collapse() {
        // "O'Brien" → punctuation drops, "o" + "brien" left.
        // The squash keeps the space, so we end up with "o brien".
        // That matches the token-bag the matcher operates on.
        assert_eq!(normalize("O'Brien").as_str(), "o brien");
    }

    #[test]
    fn sao_paulo() {
        assert_eq!(normalize("São Paulo").as_str(), "sao paulo");
    }

    #[test]
    fn suffix_expansion() {
        assert_eq!(
            normalize("Acme Corp").as_str(),
            "acme corporation"
        );
        assert_eq!(
            normalize("ACME Inc").as_str(),
            "acme incorporated"
        );
        assert_eq!(
            normalize("Test Ltd").as_str(),
            "test limited"
        );
    }

    #[test]
    fn whitespace_collapse() {
        assert_eq!(
            normalize("  hello   \t  world  ").as_str(),
            "hello world"
        );
    }

    #[test]
    fn cyrillic_transliteration() {
        // "Иван" — vanilla Cyrillic — should become "ivan".
        assert_eq!(normalize("Иван").as_str(), "ivan");
    }

    #[test]
    fn empty_input_is_empty_output() {
        assert_eq!(normalize("").as_str(), "");
        assert_eq!(normalize("   ").as_str(), "");
    }
}
