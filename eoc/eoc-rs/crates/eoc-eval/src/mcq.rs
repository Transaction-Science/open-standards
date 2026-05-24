//! Shared helpers for multiple-choice scoring.

use regex::Regex;
use std::sync::OnceLock;

/// Letter labels A, B, C, ... up to J — enough for MMLU-Pro's 10-way MCQ.
pub const LETTERS: [&str; 10] = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"];

/// Canonicalise a model's free-form response into a single letter
/// label, or `None` if no plausible letter could be extracted.
///
/// Accepted forms (case-insensitive, leading/trailing whitespace ignored):
/// - `"A"`, `"a"`, `"A."`, `"(A)"`, `"[A]"`, `"**A**"`, `"_A_"`
/// - `"Answer: A"`, `"The answer is A"`, `"Final answer: (B)"`
/// - Multi-line responses where the first non-empty line contains a letter
pub fn extract_letter(response: &str, num_choices: usize) -> Option<String> {
    let upper_bound = num_choices.min(LETTERS.len());
    let allowed: &[&str] = &LETTERS[..upper_bound];

    let cleaned = response.trim();
    if cleaned.is_empty() {
        return None;
    }

    // 1. Whole input is a letter (with optional dot / parens / markdown
    //    decoration).
    if let Some(letter) = strip_to_letter(cleaned, allowed) {
        return Some(letter);
    }

    // 2. Look for "answer is X" / "final answer: X" / "answer: X".
    static ANSWER_PHRASE: OnceLock<Regex> = OnceLock::new();
    let re = ANSWER_PHRASE.get_or_init(|| {
        Regex::new(r"(?i)\b(?:final\s+)?answer\s*(?:is|:|=)?\s*[\*\(\[]*([A-J])\b").unwrap()
    });
    if let Some(cap) = re.captures(cleaned)
        && let Some(m) = cap.get(1)
    {
        let letter = m.as_str().to_uppercase();
        if allowed.contains(&letter.as_str()) {
            return Some(letter);
        }
    }

    // 3. Standalone letter token on the first non-empty line.
    for line in cleaned.lines() {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(letter) = strip_to_letter(line, allowed) {
            return Some(letter);
        }
        // First token that looks like a letter on the first line wins.
        for tok in line.split_whitespace() {
            if let Some(letter) = strip_to_letter(tok, allowed) {
                return Some(letter);
            }
        }
        break;
    }

    None
}

fn strip_to_letter(s: &str, allowed: &[&str]) -> Option<String> {
    let s = s.trim();
    // Strip common decorations: parentheses, brackets, asterisks,
    // underscores, trailing punctuation.
    let stripped: String = s
        .chars()
        .filter(|c| !matches!(c, '(' | ')' | '[' | ']' | '*' | '_' | '.' | ':' | ',' | '\'' | '"'))
        .collect();
    let stripped = stripped.trim().to_uppercase();
    if stripped.len() == 1 && allowed.contains(&stripped.as_str()) {
        Some(stripped)
    } else {
        None
    }
}

/// Format a multiple-choice prompt with lettered options for the
/// backend. Returned exactly as MMLU-style harnesses present it.
pub fn format_mcq_prompt(question: &str, choices: &[String]) -> String {
    let mut out = String::with_capacity(question.len() + 64);
    out.push_str(question.trim());
    out.push_str("\n\n");
    for (i, c) in choices.iter().enumerate() {
        if i < LETTERS.len() {
            out.push_str(&format!("{}. {}\n", LETTERS[i], c));
        }
    }
    out.push_str("\nAnswer with the letter only.");
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extracts_bare_letter() {
        assert_eq!(extract_letter("A", 4).as_deref(), Some("A"));
        assert_eq!(extract_letter("b", 4).as_deref(), Some("B"));
    }

    #[test]
    fn extracts_decorated_letters() {
        for f in ["(A)", "[A]", "A.", "**A**", "_A_", " A "] {
            assert_eq!(extract_letter(f, 4).as_deref(), Some("A"), "input: {f}");
        }
    }

    #[test]
    fn extracts_answer_phrase() {
        assert_eq!(extract_letter("The answer is A.", 4).as_deref(), Some("A"));
        assert_eq!(extract_letter("Final answer: (B)", 4).as_deref(), Some("B"));
        assert_eq!(extract_letter("answer = C", 4).as_deref(), Some("C"));
    }

    #[test]
    fn extracts_from_first_line() {
        let r = "B\n\nbecause the gravitational force is weaker.";
        assert_eq!(extract_letter(r, 4).as_deref(), Some("B"));
    }

    #[test]
    fn rejects_out_of_range() {
        assert_eq!(extract_letter("F", 4), None);
    }

    #[test]
    fn returns_none_on_garbage() {
        assert_eq!(extract_letter("the quick brown fox", 4), None);
    }
}
