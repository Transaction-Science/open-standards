//! Local tokenizer wrapper.
//!
//! Thin façade over the HuggingFace [`tokenizers`] crate. Each backend
//! tokenizes input the same way the model was trained — BPE for most
//! llama-family models, SentencePiece for older Mistral variants,
//! WordPiece for BERT-style. The HuggingFace crate handles all three
//! behind a single `Tokenizer` object.
//!
//! We add:
//!
//! * Lazy loading from a local file path (no network access).
//! * Convenience `encode` / `decode` returning typed errors that play
//!   well with the rest of EOC.
//!
//! Auto-download from the HuggingFace hub is intentionally **not**
//! implemented here — EOC's local-inference posture is operator-owned
//! disks, not unattended network fetches. Tooling that wants
//! hub-downloads can compose `hf-hub` on top.
//!
//! ## WASM
//!
//! `tokenizers` itself is no_std-friendly when built with the right
//! features, so this module compiles to `wasm32-unknown-unknown`
//! provided the embedding host stubs out file I/O.

use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::error::{LocalError, LocalResult};

/// A lazily-loaded tokenizer.
///
/// The underlying [`tokenizers::Tokenizer`] is created on first use and
/// cached. This avoids paying the (non-trivial) BPE-vocab parse cost
/// up front in code paths that may not actually tokenize anything.
pub struct LocalTokenizer {
    path: PathBuf,
    inner: std::sync::OnceLock<Arc<tokenizers::Tokenizer>>,
}

impl LocalTokenizer {
    /// Construct a tokenizer pointed at a `tokenizer.json` on disk.
    /// The file is **not** read until the first call to [`encode`] or
    /// [`decode`].
    pub fn from_path(path: impl AsRef<Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            inner: std::sync::OnceLock::new(),
        }
    }

    /// Force the tokenizer to load now. Returns an error if the file
    /// can't be read or parsed.
    pub fn load(&self) -> LocalResult<Arc<tokenizers::Tokenizer>> {
        if let Some(t) = self.inner.get() {
            return Ok(t.clone());
        }
        let tok = tokenizers::Tokenizer::from_file(&self.path)
            .map_err(|e| LocalError::Tokenizer(format!("loading {}: {e}", self.path.display())))?;
        let arc = Arc::new(tok);
        // OnceLock::get_or_init guarantees we only initialise once even
        // under racing callers.
        let _ = self.inner.set(arc.clone());
        Ok(self.inner.get().cloned().unwrap_or(arc))
    }

    /// Encode a string into token ids.
    pub fn encode(&self, text: &str, add_special_tokens: bool) -> LocalResult<Vec<u32>> {
        let tok = self.load()?;
        let encoding = tok
            .encode(text, add_special_tokens)
            .map_err(|e| LocalError::Tokenizer(format!("encode: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode a sequence of token ids back into a string. `skip_special_tokens`
    /// drops BOS / EOS / etc.
    pub fn decode(&self, ids: &[u32], skip_special_tokens: bool) -> LocalResult<String> {
        let tok = self.load()?;
        tok.decode(ids, skip_special_tokens)
            .map_err(|e| LocalError::Tokenizer(format!("decode: {e}")))
    }

    /// Vocabulary size (rounded up to the model's reported vocab — the
    /// tokenizer's actual lookup table may be smaller).
    pub fn vocab_size(&self) -> LocalResult<usize> {
        let tok = self.load()?;
        Ok(tok.get_vocab_size(true))
    }

    /// Path the tokenizer is associated with.
    pub fn path(&self) -> &Path {
        &self.path
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::io::Write;

    /// Minimal BPE tokenizer config — vocab + merges in HuggingFace
    /// json-format. We build it inline so the test does not depend on
    /// pulling a real model.
    const TINY_TOKENIZER_JSON: &str = r#"{
        "version": "1.0",
        "truncation": null,
        "padding": null,
        "added_tokens": [],
        "normalizer": null,
        "pre_tokenizer": { "type": "Whitespace" },
        "post_processor": null,
        "decoder": null,
        "model": {
            "type": "WordLevel",
            "vocab": { "hello": 0, "world": 1, "[UNK]": 2 },
            "unk_token": "[UNK]"
        }
    }"#;

    fn write_tokenizer(dir: &Path) -> PathBuf {
        let p = dir.join("tokenizer.json");
        let mut f = fs::File::create(&p).unwrap();
        f.write_all(TINY_TOKENIZER_JSON.as_bytes()).unwrap();
        p
    }

    #[test]
    fn lazy_load_does_not_touch_disk_until_used() {
        let dir = tempfile::tempdir().unwrap();
        // Construct against a path that does not yet exist.
        let phantom = dir.path().join("nope.json");
        let _t = LocalTokenizer::from_path(&phantom);
        // No load yet — construction must not fail just because the
        // file is missing.
    }

    #[test]
    fn encode_decode_round_trip() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_tokenizer(dir.path());
        let t = LocalTokenizer::from_path(&p);
        let ids = t.encode("hello world", false).unwrap();
        assert_eq!(ids, vec![0, 1]);
        let s = t.decode(&ids, false).unwrap();
        // Whitespace pre-tokenizer + WordLevel decode returns space-
        // joined tokens — accept either ordering of trailing whitespace.
        assert!(s.contains("hello"));
        assert!(s.contains("world"));
    }

    #[test]
    fn vocab_size_is_reported() {
        let dir = tempfile::tempdir().unwrap();
        let p = write_tokenizer(dir.path());
        let t = LocalTokenizer::from_path(&p);
        assert!(t.vocab_size().unwrap() >= 3);
    }

    #[test]
    fn missing_file_yields_tokenizer_error() {
        let t = LocalTokenizer::from_path("/no/such/tokenizer.json");
        let r = t.encode("anything", false);
        assert!(matches!(r, Err(LocalError::Tokenizer(_))));
    }
}
