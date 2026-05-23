//! The pluggable statement-source trait.

use crate::error::Result;
use crate::statement::StatementLine;

/// A source of normalized statement lines.
///
/// Implementors decode their wire format (CAMT XML, webhook JSON, a
/// proprietary CSV) into [`StatementLine`]s. The contract:
///
/// - **Sync.** Mirrors `LedgerStore` / `WebhookStore`; async wrappers
///   belong in adapter crates downstream. The reference sources are
///   all in-memory decoders so this costs nothing.
/// - **Lazy.** `iter_lines` returns an iterator, not a `Vec`, so a
///   large end-of-day statement (or a long webhook backfill) doesn't
///   have to be fully materialized. Each item is a `Result` so one
///   bad line doesn't abort the whole batch — the engine decides
///   whether to skip or fail.
/// - **Idempotent.** Calling `iter_lines` twice yields the same lines
///   (sources hold their input; they don't consume a stream).
pub trait ReconciliationSource {
    /// Iterate the source's lines. Each item is a parsed
    /// [`StatementLine`] or a per-line [`crate::Error`].
    fn iter_lines(&self) -> Box<dyn Iterator<Item = Result<StatementLine>> + '_>;
}
