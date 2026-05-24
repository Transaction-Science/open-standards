//! Render memory items into a prompt-context block.
//!
//! [`inject`] takes a list of [`MemoryItem`]s and renders them into
//! a single string the agent can paste into the prompt under a
//! configurable header. It also enforces a token budget — once the
//! budget is exhausted, remaining items are dropped (lowest-priority
//! last in the input list).

use serde::{Deserialize, Serialize};

use crate::error::{MemoryError, MemoryResult};
use crate::memory::MemoryItem;

/// Injection configuration.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct InjectConfig {
    /// Header text for the injected block.
    pub header: String,
    /// Maximum tokens allowed in the rendered block (excluding header).
    pub max_tokens: u32,
    /// Separator between items.
    pub separator: String,
}

impl InjectConfig {
    /// Validate.
    pub fn new(
        header: impl Into<String>,
        max_tokens: u32,
        separator: impl Into<String>,
    ) -> MemoryResult<Self> {
        if max_tokens == 0 {
            return Err(MemoryError::Config("max_tokens must be > 0".into()));
        }
        Ok(Self {
            header: header.into(),
            max_tokens,
            separator: separator.into(),
        })
    }
}

impl Default for InjectConfig {
    fn default() -> Self {
        Self {
            header: "## Relevant memories".to_string(),
            max_tokens: 1024,
            separator: "\n- ".to_string(),
        }
    }
}

/// Output of an injection pass.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct InjectedContext {
    /// Final rendered text (header + items).
    pub text: String,
    /// Total token cost (sum of `MemoryItem.tokens` for kept items).
    pub tokens: u32,
    /// Number of items kept.
    pub kept: usize,
    /// Number of items dropped because of the budget.
    pub dropped: usize,
}

/// Render `items` into a single string, dropping items past the
/// budget. Items already arrive in priority order (highest first).
pub fn inject(items: &[MemoryItem], cfg: &InjectConfig) -> InjectedContext {
    let mut out = String::with_capacity(cfg.header.len() + 64);
    out.push_str(&cfg.header);
    let mut tokens = 0_u32;
    let mut kept = 0_usize;
    let mut dropped = 0_usize;
    for item in items {
        if tokens.saturating_add(item.tokens) > cfg.max_tokens {
            dropped += 1;
            continue;
        }
        out.push_str(&cfg.separator);
        out.push_str(&item.text);
        tokens = tokens.saturating_add(item.tokens);
        kept += 1;
    }
    InjectedContext {
        text: out,
        tokens,
        kept,
        dropped,
    }
}
