//! Composition of sub-answers.

/// Joins the per-sub-query answer strings into one final answer.
pub trait Composer: Send + Sync {
    fn compose(&self, parts: &[String]) -> String;
}

/// Joins parts with a separator (default newline). The most honest
/// composer: it does not paraphrase or summarise, so the agent's output
/// is exactly the concatenation of what the sub-dispatches returned —
/// no opportunity to hallucinate a synthesis the parts don't support.
#[derive(Debug, Clone)]
pub struct Concatenator {
    pub separator: String,
}

impl Default for Concatenator {
    fn default() -> Self {
        Self {
            separator: "\n".to_string(),
        }
    }
}

impl Concatenator {
    pub fn with_separator(separator: impl Into<String>) -> Self {
        Self {
            separator: separator.into(),
        }
    }
}

impl Composer for Concatenator {
    fn compose(&self, parts: &[String]) -> String {
        parts.join(&self.separator)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn concatenates_with_newline() {
        let c = Concatenator::default();
        assert_eq!(c.compose(&["a".into(), "b".into()]), "a\nb");
    }

    #[test]
    fn custom_separator() {
        let c = Concatenator::with_separator(" | ");
        assert_eq!(c.compose(&["a".into(), "b".into(), "c".into()]), "a | b | c");
    }

    #[test]
    fn single_part_unchanged() {
        let c = Concatenator::default();
        assert_eq!(c.compose(&["only".into()]), "only");
    }

    #[test]
    fn empty_is_empty() {
        let c = Concatenator::default();
        assert_eq!(c.compose(&[]), "");
    }
}
