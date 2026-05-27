//! Error type for grammar compilation and decoding.

use thiserror::Error;

/// Errors raised by grammar compilation, mask projection, and decoding.
#[derive(Debug, Error)]
pub enum DecodeError {
    /// Regex syntax could not be parsed.
    #[error("regex parse error at byte {pos}: {msg}")]
    RegexParse { pos: usize, msg: String },

    /// JSON-schema is malformed or uses an unsupported construct.
    #[error("json-schema error: {0}")]
    JsonSchema(String),

    /// Context-free grammar references a non-terminal that was never defined,
    /// or has no productions for the start symbol.
    #[error("cfg error: {0}")]
    Cfg(String),

    /// Caller stepped a token id that is not currently permitted by the mask.
    #[error("token id {0} is not allowed in current state")]
    TokenNotAllowed(u32),

    /// Caller stepped a token id outside the vocabulary range.
    #[error("token id {0} is out of vocabulary range (size {1})")]
    TokenOutOfRange(u32, usize),

    /// Vocabulary is empty.
    #[error("vocabulary is empty")]
    EmptyVocabulary,
}
