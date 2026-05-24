//! Fuzzy name-matching primitives and the top-level [`screen`] entry point.
//!
//! Four independent similarity / distance measures cover the failure
//! modes we see in real sanctions data:
//!
//! - **Levenshtein** — character-edit distance. Tolerant of typos and
//!   single-character drops.
//! - **Jaro-Winkler** — string similarity weighted toward the prefix.
//!   The classic algorithm for short-name matching; OFAC's published
//!   guidance uses it.
//! - **Soundex / Metaphone** — phonetic codes. Match across
//!   transliteration variants where Levenshtein wouldn't
//!   (e.g. `Mohammed` / `Muhammad`).
//! - **Token Jaccard** — bag-of-tokens overlap. Handles re-ordering
//!   ("Smith John" vs "John Smith") and partial-name queries.
//!
//! The four are combined into a single [`MatchScore`] via a weighted
//! blend the [`screen`] entry-point selects per input.

use serde::{Deserialize, Serialize};

use crate::lists::SanctionedEntity;
use crate::normalize::normalize;
use crate::storage::SanctionsIndex;

/// Which low-level algorithm produced a hit.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchMethod {
    /// Exact equality on canonicalised name.
    Exact,
    /// Levenshtein edit distance below threshold.
    Levenshtein,
    /// Jaro-Winkler similarity above threshold.
    JaroWinkler,
    /// Phonetic-code equality (Soundex / Metaphone).
    Phonetic,
    /// Token-bag Jaccard similarity above threshold.
    TokenJaccard,
    /// Combined blend exceeded threshold.
    Combined,
}

/// Which field of the entity matched.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum MatchedField {
    /// Primary `name` field.
    PrimaryName,
    /// One of the `name_aliases`.
    Alias,
    /// An `Identification` value.
    Identification,
}

/// A single ranked hit returned by [`screen`].
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct MatchScore {
    /// The sanctioned entity the query hit.
    pub entity: SanctionedEntity,
    /// Combined similarity in `[0.0, 1.0]`.
    pub score: f32,
    /// Which algorithm topped out.
    pub method: MatchMethod,
    /// Which field the match landed on.
    pub matched_field: MatchedField,
}

/// Levenshtein edit distance between two strings.
///
/// Wagner–Fischer with a single rolling row for memory efficiency.
#[must_use]
pub fn levenshtein(a: &str, b: &str) -> usize {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let (m, n) = (a_chars.len(), b_chars.len());

    if m == 0 {
        return n;
    }
    if n == 0 {
        return m;
    }

    let mut prev: Vec<usize> = (0..=n).collect();
    let mut curr: Vec<usize> = vec![0; n + 1];

    for i in 1..=m {
        curr[0] = i;
        for j in 1..=n {
            let cost = usize::from(a_chars[i - 1] != b_chars[j - 1]);
            curr[j] = (prev[j] + 1) // deletion
                .min(curr[j - 1] + 1) // insertion
                .min(prev[j - 1] + cost); // substitution
        }
        std::mem::swap(&mut prev, &mut curr);
    }
    prev[n]
}

/// Jaro similarity in `[0.0, 1.0]`.
///
/// Classical reference: Jaro (1989). Match window is
/// `max(|a|, |b|) / 2 - 1`. Transpositions are counted as half-edits.
#[must_use]
pub fn jaro(a: &str, b: &str) -> f32 {
    let a: Vec<char> = a.chars().collect();
    let b: Vec<char> = b.chars().collect();
    let (m, n) = (a.len(), b.len());

    if m == 0 && n == 0 {
        return 1.0;
    }
    if m == 0 || n == 0 {
        return 0.0;
    }

    let match_distance = (m.max(n) / 2).saturating_sub(1);

    let mut a_matches = vec![false; m];
    let mut b_matches = vec![false; n];
    let mut matches: usize = 0;

    for (i, ai) in a.iter().enumerate() {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(n);
        for j in start..end {
            if b_matches[j] {
                continue;
            }
            if *ai != b[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1;
            break;
        }
    }

    if matches == 0 {
        return 0.0;
    }

    // Count transpositions.
    let mut transpositions: usize = 0;
    let mut k: usize = 0;
    for i in 0..m {
        if !a_matches[i] {
            continue;
        }
        while !b_matches[k] {
            k += 1;
        }
        if a[i] != b[k] {
            transpositions += 1;
        }
        k += 1;
    }

    let matches = matches as f32;
    (matches / m as f32 + matches / n as f32 + (matches - transpositions as f32 / 2.0) / matches)
        / 3.0
}

/// Jaro-Winkler similarity.
///
/// Adds a `p * l * (1 - jaro)` boost where `l` is the length of the
/// shared prefix (capped at 4) and `p` is the scaling factor (0.1,
/// per Winkler's recommendation).
#[must_use]
pub fn jaro_winkler(a: &str, b: &str) -> f32 {
    let base = jaro(a, b);
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let prefix = a_chars
        .iter()
        .zip(b_chars.iter())
        .take(4)
        .take_while(|(x, y)| x == y)
        .count();
    base + (prefix as f32) * 0.1 * (1.0 - base)
}

/// Soundex code (American Soundex, the algorithm taught by the
/// US National Archives).
///
/// Returns a 4-character ASCII code of the form `[A-Z][0-9]{3}`.
#[must_use]
pub fn soundex(input: &str) -> String {
    let upper: String = input
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();

    if upper.is_empty() {
        return "0000".to_string();
    }

    let first = upper.chars().next().unwrap_or('0');
    let mut code = String::with_capacity(4);
    code.push(first);

    let mut prev_digit: char = soundex_digit(first);

    for c in upper.chars().skip(1) {
        let d = soundex_digit(c);
        if d == '0' {
            // NARA Soundex: vowels (A, E, I, O, U) and Y reset `prev_digit`
            // so that an interleaved consonant-pair across a vowel still
            // gets two distinct code digits. H and W are *transparent*:
            // they neither emit a digit nor reset the suppression state.
            if matches!(c, 'H' | 'W') {
                continue;
            }
            prev_digit = '0';
            continue;
        }
        if d != prev_digit {
            code.push(d);
            if code.len() == 4 {
                break;
            }
        }
        prev_digit = d;
    }

    while code.len() < 4 {
        code.push('0');
    }
    code
}

fn soundex_digit(c: char) -> char {
    match c {
        'B' | 'F' | 'P' | 'V' => '1',
        'C' | 'G' | 'J' | 'K' | 'Q' | 'S' | 'X' | 'Z' => '2',
        'D' | 'T' => '3',
        'L' => '4',
        'M' | 'N' => '5',
        'R' => '6',
        _ => '0',
    }
}

/// Double-Metaphone-lite phonetic code.
///
/// A trimmed Metaphone that's good enough to catch
/// transliteration twins (`Mohammed`/`Muhammad`,
/// `Kaiser`/`Keyser`). Full Double Metaphone is a 700-line
/// finite-state thing we don't need for screening; this covers
/// the cases real watchlists actually hit.
#[must_use]
pub fn metaphone(input: &str) -> String {
    let upper: String = input
        .chars()
        .filter(|c| c.is_ascii_alphabetic())
        .map(|c| c.to_ascii_uppercase())
        .collect();
    let chars: Vec<char> = upper.chars().collect();
    let mut out = String::new();
    let mut i = 0;
    while i < chars.len() {
        let c = chars[i];
        match c {
            'A' | 'E' | 'I' | 'O' | 'U' => {
                if i == 0 {
                    out.push(c);
                }
            }
            'B' => out.push('B'),
            'C' => {
                // 'CH' -> 'X' (sh sound), 'CI/CE/CY' -> 'S', else 'K'.
                let next = chars.get(i + 1).copied().unwrap_or(' ');
                if next == 'H' {
                    out.push('X');
                    i += 1;
                } else if matches!(next, 'I' | 'E' | 'Y') {
                    out.push('S');
                } else {
                    out.push('K');
                }
            }
            'D' => out.push('T'),
            'F' => out.push('F'),
            'G' => out.push('K'),
            'H' => {
                if i == 0 {
                    out.push('H');
                }
            }
            'J' => out.push('J'),
            'K' => out.push('K'),
            'L' => out.push('L'),
            'M' => out.push('M'),
            'N' => out.push('N'),
            'P' => {
                let next = chars.get(i + 1).copied().unwrap_or(' ');
                if next == 'H' {
                    out.push('F');
                    i += 1;
                } else {
                    out.push('P');
                }
            }
            'Q' => out.push('K'),
            'R' => out.push('R'),
            'S' => {
                let next = chars.get(i + 1).copied().unwrap_or(' ');
                if next == 'H' {
                    out.push('X');
                    i += 1;
                } else {
                    out.push('S');
                }
            }
            'T' => {
                let next = chars.get(i + 1).copied().unwrap_or(' ');
                if next == 'H' {
                    out.push('0'); // 'th' sound, encoded as zero
                    i += 1;
                } else {
                    out.push('T');
                }
            }
            'V' => out.push('F'),
            'W' | 'Y' => {
                let next = chars.get(i + 1).copied().unwrap_or(' ');
                if matches!(next, 'A' | 'E' | 'I' | 'O' | 'U') {
                    out.push(c);
                }
            }
            'X' => out.push_str("KS"),
            'Z' => out.push('S'),
            _ => {}
        }
        i += 1;
    }

    // Collapse runs of the same letter.
    let mut collapsed = String::with_capacity(out.len());
    let mut prev: Option<char> = None;
    for c in out.chars() {
        if Some(c) != prev {
            collapsed.push(c);
        }
        prev = Some(c);
    }
    collapsed
}

/// Token-bag Jaccard similarity in `[0.0, 1.0]`.
///
/// Treats whitespace as the only token boundary, which is right after
/// [`crate::normalize::normalize`] has dealt with punctuation.
#[must_use]
pub fn token_jaccard(a: &str, b: &str) -> f32 {
    use std::collections::HashSet;

    let a_set: HashSet<&str> = a.split_whitespace().collect();
    let b_set: HashSet<&str> = b.split_whitespace().collect();

    if a_set.is_empty() && b_set.is_empty() {
        return 1.0;
    }
    let intersection = a_set.intersection(&b_set).count() as f32;
    let union = a_set.union(&b_set).count() as f32;
    if union == 0.0 {
        return 0.0;
    }
    intersection / union
}

/// Weighted blend producing the combined score reported in [`MatchScore`].
///
/// Weights are tuned to put a thumb on Jaro-Winkler (the
/// industry-standard for short-form name screening) without
/// drowning out the long-form-recovery signal Jaccard provides.
fn combined_score(query: &str, candidate: &str) -> f32 {
    let lev = levenshtein(query, candidate);
    let max_len = query.chars().count().max(candidate.chars().count()).max(1);
    let lev_norm = 1.0 - (lev as f32 / max_len as f32);

    let jw = jaro_winkler(query, candidate);
    let jac = token_jaccard(query, candidate);

    let phon = if soundex(query) == soundex(candidate)
        || metaphone(query) == metaphone(candidate)
    {
        1.0
    } else {
        0.0
    };

    // Weights sum to 1.0.
    0.20 * lev_norm + 0.45 * jw + 0.25 * jac + 0.10 * phon
}

/// Pick the best `(field, score, method)` triple a query produces
/// against a single entity.
fn score_entity(query_norm: &str, entity: &SanctionedEntity) -> (f32, MatchMethod, MatchedField) {
    let mut best = (0.0_f32, MatchMethod::Combined, MatchedField::PrimaryName);

    let primary_norm = normalize(&entity.name);
    let primary_score = combined_score(query_norm, primary_norm.as_str());
    if primary_norm.as_str() == query_norm {
        best = (1.0, MatchMethod::Exact, MatchedField::PrimaryName);
    } else if primary_score > best.0 {
        best = (primary_score, MatchMethod::Combined, MatchedField::PrimaryName);
    }

    for alias in &entity.name_aliases {
        let alias_norm = normalize(alias);
        let s = combined_score(query_norm, alias_norm.as_str());
        if alias_norm.as_str() == query_norm && best.0 < 1.0 {
            best = (1.0, MatchMethod::Exact, MatchedField::Alias);
        } else if s > best.0 {
            best = (s, MatchMethod::Combined, MatchedField::Alias);
        }
    }

    // Identifications: an exact-string equality on the value (after
    // lowercasing) counts as a maximum-confidence hit. Useful when
    // operators screen by passport / national-id number directly.
    for ident in &entity.identifications {
        if ident.value.to_lowercase() == query_norm {
            best = (1.0, MatchMethod::Exact, MatchedField::Identification);
        }
    }

    best
}

/// Screen a single `query` name against the prebuilt `index`.
///
/// Returns every entity whose combined score is at or above
/// `threshold`, sorted descending. The default threshold for
/// production use is around `0.85`; operators tune per portfolio
/// risk tolerance.
#[must_use]
pub fn screen(query: &str, index: &SanctionsIndex, threshold: f32) -> Vec<MatchScore> {
    let query_norm = normalize(query);
    let q = query_norm.as_str();

    // Bloom filter pre-filter: candidate IDs only.
    let candidate_ids = index.candidate_ids(q);

    let mut hits: Vec<MatchScore> = Vec::new();
    for id in candidate_ids {
        let Some(entity) = index.by_id.get(&id) else {
            continue;
        };
        let (score, method, matched_field) = score_entity(q, entity);
        if score >= threshold {
            hits.push(MatchScore {
                entity: entity.clone(),
                score,
                method,
                matched_field,
            });
        }
    }

    hits.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    hits
}

#[cfg(test)]
mod tests {
    use super::*;

    // ---- Levenshtein ----
    #[test]
    fn levenshtein_identical() {
        assert_eq!(levenshtein("kitten", "kitten"), 0);
    }

    #[test]
    fn levenshtein_classic_kitten_sitting() {
        // Canonical reference example.
        assert_eq!(levenshtein("kitten", "sitting"), 3);
    }

    #[test]
    fn levenshtein_saturday_sunday() {
        assert_eq!(levenshtein("saturday", "sunday"), 3);
    }

    #[test]
    fn levenshtein_empty() {
        assert_eq!(levenshtein("", ""), 0);
        assert_eq!(levenshtein("abc", ""), 3);
        assert_eq!(levenshtein("", "abc"), 3);
    }

    // ---- Jaro-Winkler ----
    #[test]
    fn jaro_winkler_classic_martha_marhta() {
        // Winkler's published example: 0.961.
        let s = jaro_winkler("MARTHA", "MARHTA");
        assert!((s - 0.961).abs() < 0.01, "expected ~0.961, got {s}");
    }

    #[test]
    fn jaro_winkler_dwayne_duane() {
        // Another canonical pair: 0.840.
        let s = jaro_winkler("DWAYNE", "DUANE");
        assert!((s - 0.840).abs() < 0.02, "expected ~0.840, got {s}");
    }

    #[test]
    fn jaro_winkler_identical_is_one() {
        assert!((jaro_winkler("hello", "hello") - 1.0).abs() < 1e-6);
    }

    // ---- Soundex ----
    #[test]
    fn soundex_robert_rupert() {
        // Knuth's canonical test: both encode to R163.
        assert_eq!(soundex("Robert"), "R163");
        assert_eq!(soundex("Rupert"), "R163");
    }

    #[test]
    fn soundex_ashcraft() {
        // NARA-canonical: A261 (one common interpretation; algorithms vary on H).
        // Our simpler-reset variant returns A261.
        assert_eq!(soundex("Ashcraft"), "A261");
    }

    #[test]
    fn soundex_pads_to_four() {
        assert_eq!(soundex("Lee").len(), 4);
        assert_eq!(soundex("A"), "A000");
    }

    // ---- Metaphone ----
    #[test]
    fn metaphone_thompson() {
        // 'Th' -> '0', 'omps' -> 'MPS', 'on' -> 'N'. Suffices to round-trip.
        let m = metaphone("Thompson");
        assert!(m.starts_with('0'), "got {m}");
    }

    // ---- Jaccard ----
    #[test]
    fn jaccard_full_overlap() {
        assert!((token_jaccard("a b c", "a b c") - 1.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_no_overlap() {
        assert!((token_jaccard("a b", "c d") - 0.0).abs() < 1e-6);
    }

    #[test]
    fn jaccard_reorder_preserves_one() {
        assert!((token_jaccard("john smith", "smith john") - 1.0).abs() < 1e-6);
    }
}
