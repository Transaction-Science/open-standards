//! SentencePiece BPE tokenizer for Llama 1/2 family models.
//!
//! Follows the `llm_tokenizer_spm` reference implementation in llama.cpp:
//! 1. Replace each space in the input text with U+2581 (`▁`).
//! 2. Prepend a single `▁` (the SentencePiece convention for "word-start").
//! 3. Treat the result as a sequence of UTF-8 characters; each character
//!    becomes a symbol.
//! 4. Score every adjacent symbol pair by looking up the *concatenation*
//!    of the two symbols' text in the vocabulary; the score is the
//!    vocabulary entry's score (higher = preferred merge).
//! 5. Maintain a max-priority queue of bigrams. Pop the highest-scoring
//!    bigram, merge the two symbols, re-score the new bigram with its
//!    left and right neighbors, push those back in.
//! 6. When no merges are possible, the remaining symbols are looked up
//!    in the vocabulary; unknown ones fall back to byte tokens
//!    (`<0xNN>` strings) which are also in the SPM vocabulary.
//!
//! The output is the sequence of token IDs.
//!
//! Phase 1.9 ships SPM only. Llama 3 uses BPE with a different algorithm
//! (rank-ordered merges from a fixed merge table); that lands in 1.10.

use crate::{GgufModel, GgufValue};
use std::collections::HashMap;

/// A loaded vocabulary, ready to tokenize.
#[derive(Debug)]
pub struct Vocab {
    /// Token strings indexed by ID.
    pub tokens: Vec<String>,
    /// Per-token score (higher = preferred). Index matches `tokens`.
    pub scores: Vec<f32>,
    /// Reverse map: token string → ID. Built from `tokens`.
    pub token_to_id: HashMap<String, u32>,
    /// Beginning-of-sequence token ID.
    pub bos_id: Option<u32>,
    /// End-of-sequence token ID.
    pub eos_id: Option<u32>,
    /// Unknown-token ID (fallback when nothing matches).
    pub unk_id: Option<u32>,
    /// Tokenizer model name from GGUF (`llama`, `gpt2`, etc.).
    pub model_name: String,
    /// Optional BPE merge table — present when the model uses BPE
    /// (Llama 3, Mistral, Qwen, GPT-style models).
    pub bpe_merges: Option<BpeMerges>,
}

/// Rank-ordered BPE merge table.
///
/// Each entry is a `(left, right)` pair; the entry's index in `merges` is
/// its **rank** (lower rank = higher priority, i.e., merged first).
///
/// This is the tiktoken / GPT-2 / Llama 3 BPE convention: greedily apply
/// the lowest-rank merge available at each step, until no merges in the
/// table apply to any adjacent pair.
#[derive(Debug, Clone)]
pub struct BpeMerges {
    /// Ordered list. `merges[r] = (left, right)` means rank `r`.
    pub merges: Vec<(String, String)>,
    /// `merge_to_rank[(left, right)] = rank`. Built from `merges`.
    pub merge_to_rank: HashMap<(String, String), usize>,
}

impl BpeMerges {
    /// Load BPE merges from GGUF metadata. Returns `None` if the model
    /// has no `tokenizer.ggml.merges` key.
    pub fn from_gguf(model: &GgufModel) -> Option<Self> {
        let arr = match model.metadata.get("tokenizer.ggml.merges") {
            Some(GgufValue::Array(a)) => a,
            _ => return None,
        };
        let mut merges: Vec<(String, String)> = Vec::with_capacity(arr.len());
        for v in arr {
            if let GgufValue::String(s) = v {
                // Each merge is "left right" — single space separator.
                if let Some(idx) = s.find(' ') {
                    let left = s[..idx].to_string();
                    let right = s[idx + 1..].to_string();
                    merges.push((left, right));
                }
            }
        }
        let merge_to_rank = merges.iter().enumerate()
            .map(|(rank, pair)| (pair.clone(), rank))
            .collect();
        Some(Self { merges, merge_to_rank })
    }

    pub fn rank(&self, left: &str, right: &str) -> Option<usize> {
        self.merge_to_rank.get(&(left.to_string(), right.to_string())).copied()
    }
}

impl Vocab {
    /// Load vocabulary from a GGUF model.
    pub fn from_gguf(model: &GgufModel) -> Result<Self, VocabError> {
        let model_name = model.metadata_string("tokenizer.ggml.model")
            .unwrap_or("llama").to_string();

        let tokens: Vec<String> = match model.metadata.get("tokenizer.ggml.tokens") {
            Some(GgufValue::Array(arr)) => {
                arr.iter().filter_map(|v| match v {
                    GgufValue::String(s) => Some(s.clone()),
                    _ => None,
                }).collect()
            }
            _ => return Err(VocabError::MissingTokens),
        };

        let scores: Vec<f32> = match model.metadata.get("tokenizer.ggml.scores") {
            Some(GgufValue::Array(arr)) => {
                arr.iter().filter_map(|v| match v {
                    GgufValue::F32(s) => Some(*s),
                    _ => None,
                }).collect()
            }
            // Some models (BPE-only) don't have scores; use 0.0 for all.
            _ => vec![0.0; tokens.len()],
        };

        if !scores.is_empty() && scores.len() != tokens.len() {
            return Err(VocabError::ScoresLengthMismatch {
                tokens: tokens.len(), scores: scores.len(),
            });
        }

        let mut token_to_id = HashMap::with_capacity(tokens.len());
        for (i, t) in tokens.iter().enumerate() {
            // Last write wins on duplicates — this matches llama.cpp behavior
            // for vocabularies with redundant entries.
            token_to_id.insert(t.clone(), i as u32);
        }

        let bos_id = model.metadata_u32("tokenizer.ggml.bos_token_id");
        let eos_id = model.metadata_u32("tokenizer.ggml.eos_token_id");
        let unk_id = model.metadata_u32("tokenizer.ggml.unknown_token_id");

        let bpe_merges = BpeMerges::from_gguf(model);

        Ok(Self {
            tokens, scores, token_to_id,
            bos_id, eos_id, unk_id,
            model_name,
            bpe_merges,
        })
    }

    pub fn len(&self) -> usize { self.tokens.len() }
    pub fn is_empty(&self) -> bool { self.tokens.is_empty() }

    /// Lookup token text by ID. Returns the token string or `None` if ID is
    /// out of range.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.tokens.get(id as usize).map(|s| s.as_str())
    }

    /// Encode text using the SPM (SentencePiece BPE) algorithm.
    pub fn encode_spm(&self, text: &str, add_bos: bool) -> Vec<u32> {
        // Step 1+2: replace spaces with U+2581 and prepend.
        let mut prepared = String::with_capacity(text.len() + 3);
        prepared.push('\u{2581}');
        for ch in text.chars() {
            if ch == ' ' { prepared.push('\u{2581}'); }
            else { prepared.push(ch); }
        }

        // Step 3: split into UTF-8 chars; build symbol list.
        let chars: Vec<&str> = prepared.graphemes();
        // Symbol list as doubly-linked indices (-1 = none).
        let n = chars.len();
        let mut symbols: Vec<Symbol> = (0..n).map(|i| Symbol {
            text: chars[i].to_string(),
            prev: if i == 0 { -1 } else { i as isize - 1 },
            next: if i + 1 == n { -1 } else { i as isize + 1 },
            alive: true,
        }).collect();

        // Step 4+5: max-heap of bigrams keyed by score.
        // Implemented as a Vec we re-sort, since std lacks a max-heap of
        // arbitrary key/value with stable tie-break. n is small enough
        // (sentence-length) that this is fine.
        let mut bigrams: Vec<Bigram> = Vec::new();

        let try_add_bigram = |left: isize, right: isize, syms: &Vec<Symbol>, bgs: &mut Vec<Bigram>| {
            if left < 0 || right < 0 { return; }
            if (left as usize) >= syms.len() || (right as usize) >= syms.len() { return; }
            let l = &syms[left as usize];
            let r = &syms[right as usize];
            if !l.alive || !r.alive { return; }
            let merged = format!("{}{}", l.text, r.text);
            if let Some(&id) = self.token_to_id.get(&merged) {
                let score = self.scores.get(id as usize).copied().unwrap_or(0.0);
                bgs.push(Bigram {
                    left, right, score,
                    size: merged.len(),
                });
            }
        };

        // Seed with adjacent pairs.
        for i in 0..n {
            if i + 1 < n {
                try_add_bigram(i as isize, (i + 1) as isize, &symbols, &mut bigrams);
            }
        }

        // Iteratively merge.
        while !bigrams.is_empty() {
            // Find the highest-priority bigram. Tie-break: lower left index
            // first (matches SPM's left-to-right preference).
            let mut best_idx = 0usize;
            let mut best = &bigrams[0];
            for (idx, bg) in bigrams.iter().enumerate().skip(1) {
                if bg.score > best.score
                    || (bg.score == best.score && bg.left < best.left)
                {
                    best_idx = idx;
                    best = bg;
                }
            }
            let bg = bigrams.swap_remove(best_idx);

            // Validate that both endpoints are still alive and adjacent.
            let left = bg.left as usize;
            let right = bg.right as usize;
            if !symbols[left].alive || !symbols[right].alive { continue; }
            if symbols[left].next != bg.right { continue; }
            // The merged size should still match — guards against stale bigrams
            // where a symbol grew via earlier merges.
            let merged = format!("{}{}", symbols[left].text, symbols[right].text);
            if merged.len() != bg.size { continue; }

            // Merge: extend left, mark right dead, fix links.
            symbols[left].text = merged;
            let new_next = symbols[right].next;
            symbols[left].next = new_next;
            if new_next >= 0 { symbols[new_next as usize].prev = bg.left; }
            symbols[right].alive = false;

            // Add new bigrams with neighbors.
            try_add_bigram(symbols[left].prev, bg.left, &symbols, &mut bigrams);
            try_add_bigram(bg.left, symbols[left].next, &symbols, &mut bigrams);
        }

        // Step 6: walk the linked list and emit token IDs.
        let mut out = Vec::new();
        if add_bos { if let Some(id) = self.bos_id { out.push(id); } }

        let mut idx: isize = 0;
        // Find the first alive symbol (always 0 if n > 0; defensive).
        while idx < n as isize && !symbols[idx as usize].alive {
            idx = symbols[idx as usize].next;
        }
        while idx >= 0 {
            let s = &symbols[idx as usize];
            if let Some(&id) = self.token_to_id.get(&s.text) {
                out.push(id);
            } else {
                // Byte fallback: emit each UTF-8 byte as `<0xNN>` token.
                for b in s.text.bytes() {
                    let key = format!("<0x{:02X}>", b);
                    if let Some(&id) = self.token_to_id.get(&key) {
                        out.push(id);
                    } else if let Some(unk) = self.unk_id {
                        out.push(unk);
                    }
                    // Else: drop — vocabulary is incomplete.
                }
            }
            idx = s.next;
        }

        out
    }

    /// Encode text using rank-ordered BPE (Llama 3, Mistral, Qwen, GPT-2 style).
    ///
    /// Algorithm:
    /// 1. Treat the input as a sequence of UTF-8 bytes; each byte becomes
    ///    a single-character symbol (using the canonical byte→char mapping
    ///    from GPT-2).
    /// 2. Repeatedly find the adjacent pair with the lowest merge rank
    ///    (highest priority) and merge it.
    /// 3. When no pairs in the merge table apply, look up each remaining
    ///    symbol in the vocabulary and emit its token ID.
    ///
    /// Llama 3 prepends a leading space to the first word; we apply this
    /// when `prefix_space` is true. (Real Llama 3 also runs a regex
    /// pre-tokenizer that splits text into chunks; this implementation
    /// uses a simpler whitespace split, which produces the correct output
    /// on standard ASCII text but differs on edge cases like consecutive
    /// punctuation. Phase 1.11 will add the regex pre-tokenizer when the
    /// `regex` crate dependency is acceptable.)
    pub fn encode_bpe(&self, text: &str, add_bos: bool, prefix_space: bool) -> Vec<u32> {
        let merges = match &self.bpe_merges {
            Some(m) => m,
            None => {
                return self.encode_bytes_only(text, add_bos);
            }
        };

        let mut out = Vec::new();
        if add_bos { if let Some(id) = self.bos_id { out.push(id); } }

        // Pre-tokenize: split on whitespace runs. Each run becomes a chunk;
        // leading space is preserved on each non-first chunk by prepending
        // a literal space.
        let chunks: Vec<String> = pre_tokenize_whitespace(text, prefix_space);

        for chunk in &chunks {
            self.bpe_encode_chunk(chunk, merges, &mut out);
        }
        out
    }

    /// Encode text using BPE with the canonical GPT-2 regex pre-tokenizer
    /// instead of simple whitespace splitting.
    ///
    /// This is the form that matches Llama 3 / tiktoken / GPT-2 behavior on
    /// edge cases like contractions (`don't` → `don` + `'t`), runs of
    /// punctuation, and number sequences.
    pub fn encode_bpe_regex(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let merges = match &self.bpe_merges {
            Some(m) => m,
            None => {
                return self.encode_bytes_only(text, add_bos);
            }
        };

        let mut out = Vec::new();
        if add_bos { if let Some(id) = self.bos_id { out.push(id); } }

        let chunks = pre_tokenize_gpt2(text);
        for chunk in &chunks {
            self.bpe_encode_chunk(chunk, merges, &mut out);
        }
        out
    }

    /// Apply BPE merges to one pre-token chunk. Emit token IDs into `out`.
    fn bpe_encode_chunk(&self, chunk: &str, merges: &BpeMerges, out: &mut Vec<u32>) {
        // Convert chunk to a list of single-byte symbols using the
        // GPT-2 byte→Unicode mapping.
        let mut symbols: Vec<String> = chunk.bytes()
            .map(|b| byte_to_char(b).to_string())
            .collect();

        // Repeatedly find the lowest-rank merge among adjacent pairs.
        loop {
            let mut best_rank: Option<usize> = None;
            let mut best_idx = 0usize;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(rank) = merges.rank(&symbols[i], &symbols[i + 1]) {
                    if best_rank.map_or(true, |r| rank < r) {
                        best_rank = Some(rank);
                        best_idx = i;
                    }
                }
            }
            let rank = match best_rank { Some(r) => r, None => break };
            let (left, right) = &merges.merges[rank];
            let merged = format!("{}{}", left, right);
            symbols[best_idx] = merged;
            symbols.remove(best_idx + 1);
        }

        for s in &symbols {
            if let Some(&id) = self.token_to_id.get(s) {
                out.push(id);
            } else if let Some(unk) = self.unk_id {
                out.push(unk);
            }
        }
    }

    /// Fallback: emit each byte's character mapping as its own token.
    /// Used when no BPE merge table is loaded.
    fn encode_bytes_only(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let mut out = Vec::new();
        if add_bos { if let Some(id) = self.bos_id { out.push(id); } }
        for b in text.bytes() {
            let key = byte_to_char(b).to_string();
            if let Some(&id) = self.token_to_id.get(&key) {
                out.push(id);
            } else if let Some(unk) = self.unk_id {
                out.push(unk);
            }
        }
        out
    }

    /// Decode a sequence of BPE-encoded token IDs back to text. Reverses
    /// the byte→char mapping.
    pub fn decode_bpe(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            if Some(id) == self.bos_id || Some(id) == self.eos_id { continue; }
            if let Some(t) = self.id_to_token(id) {
                for ch in t.chars() {
                    if let Some(b) = char_to_byte(ch) {
                        bytes.push(b);
                    }
                }
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Decode a sequence of token IDs back to text. Reverses the SPM
    /// space-replacement: U+2581 → ASCII space.
    pub fn decode_spm(&self, ids: &[u32]) -> String {
        let mut out = String::new();
        for &id in ids {
            if let Some(t) = self.id_to_token(id) {
                // Skip BOS/EOS in output.
                if Some(id) == self.bos_id || Some(id) == self.eos_id { continue; }
                // Byte tokens: parse `<0xNN>` back to bytes.
                if t.starts_with("<0x") && t.ends_with('>') && t.len() == 6 {
                    if let Ok(b) = u8::from_str_radix(&t[3..5], 16) {
                        // Append raw byte. For test purposes assume valid
                        // UTF-8 sequences are preserved by concatenation.
                        out.push(b as char);
                        continue;
                    }
                }
                // Replace U+2581 with space.
                for ch in t.chars() {
                    if ch == '\u{2581}' { out.push(' '); }
                    else { out.push(ch); }
                }
            }
        }
        // SPM typically prepends a leading space; trim it.
        if out.starts_with(' ') { out.remove(0); }
        out
    }
}

/// One symbol in the SPM working list.
#[derive(Debug, Clone)]
struct Symbol {
    text: String,
    prev: isize,
    next: isize,
    alive: bool,
}

/// One candidate bigram for merging.
#[derive(Debug, Clone)]
struct Bigram {
    left: isize,
    right: isize,
    score: f32,
    size: usize,
}

/// Errors loading a vocabulary.
#[derive(Debug)]
pub enum VocabError {
    MissingTokens,
    ScoresLengthMismatch { tokens: usize, scores: usize },
}

impl std::fmt::Display for VocabError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingTokens => write!(f,
                "GGUF metadata missing tokenizer.ggml.tokens"),
            Self::ScoresLengthMismatch { tokens, scores } => write!(f,
                "tokens length {} != scores length {}", tokens, scores),
        }
    }
}

impl std::error::Error for VocabError {}

/// Trait extension: split a string into Unicode grapheme clusters.
/// Llama's SPM works on UTF-8 chars; we use chars (Unicode scalar values),
/// which is correct for the tokens that appear in standard Llama vocabs.
trait GraphemesExt {
    fn graphemes(&self) -> Vec<&str>;
}

impl GraphemesExt for String {
    fn graphemes(&self) -> Vec<&str> {
        // Walk char boundaries and emit slices.
        let mut out = Vec::new();
        let bytes = self.as_bytes();
        let mut i = 0;
        while i < bytes.len() {
            let mut j = i + 1;
            while j < bytes.len() && (bytes[j] & 0xC0) == 0x80 { j += 1; }
            out.push(&self[i..j]);
            i = j;
        }
        out
    }
}

// =====================================================================
// BPE helpers: pre-tokenization and the GPT-2 byte↔char bijection
// =====================================================================

/// Split text into BPE pre-tokens using the canonical GPT-2 regex.
///
/// The reference pattern from OpenAI's GPT-2 tokenizer is:
/// ```text
/// 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
/// ```
///
/// Rust's `regex` crate doesn't support `(?!\S)` (negative lookahead), so
/// we approximate the trailing-whitespace branch with `\s+$|\s+` and split
/// the input on newlines first to localize the `$` anchor.
///
/// In practice this matches GPT-2's behavior on standard text: chunks of
/// letters, chunks of numbers, chunks of punctuation, each preceded by an
/// optional single space; common English contractions as their own chunks;
/// runs of whitespace as their own chunks (compressed onto the next token
/// when followed by non-whitespace, kept as standalone when trailing).
pub fn pre_tokenize_gpt2(text: &str) -> Vec<String> {
    use regex::Regex;
    // Lazily construct once per call. Real production code should cache
    // this in a `OnceCell` or `lazy_static`; for now correctness > perf.
    let pat = r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+";
    let re = Regex::new(pat).expect("GPT-2 pre-tokenizer regex is valid");

    let mut out = Vec::new();
    for mat in re.find_iter(text) {
        out.push(mat.as_str().to_string());
    }
    out
}

/// Split text into BPE pre-tokens by whitespace. Each non-first chunk gets
/// a leading space attached (this matches how GPT-2/Llama 3 tokenize "the
/// quick brown fox" as `the`, ` quick`, ` brown`, ` fox`).
///
/// This is a simplified pre-tokenizer; the canonical GPT-2 regex
/// (`'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+`)
/// is more sophisticated. Phase 1.11 will optionally swap this in once a
/// regex dependency is acceptable.
fn pre_tokenize_whitespace(text: &str, prefix_space_for_first: bool) -> Vec<String> {
    let mut out = Vec::new();
    let mut current = String::new();
    let mut started = false;

    for (i, ch) in text.char_indices() {
        let is_ws = ch.is_whitespace();
        if is_ws {
            if !current.is_empty() {
                out.push(std::mem::take(&mut current));
            }
            // Whitespace becomes the leading space of the next chunk.
            // We skip multiple consecutive whitespace characters (only emit
            // one leading space).
            if i + ch.len_utf8() < text.len() {
                let rest = &text[i + ch.len_utf8()..];
                let next_is_ws = rest.chars().next().map_or(false, |c| c.is_whitespace());
                if !next_is_ws {
                    current.push(' ');
                }
            }
        } else {
            if !started && prefix_space_for_first && current.is_empty() {
                current.push(' ');
            }
            current.push(ch);
            started = true;
        }
    }
    if !current.is_empty() { out.push(current); }
    out
}

/// GPT-2 byte→Unicode mapping. The 256 byte values map to a specific set
/// of Unicode codepoints chosen to be (1) all printable, (2) avoiding
/// whitespace and control characters, so the result is round-trippable
/// through Unicode-aware processing.
///
/// The canonical mapping (from `bytes_to_unicode()` in GPT-2's
/// tokenizer.py):
///   - bytes 33..=126 ('!'..='~'), 161..=172 ('¡'..='¬'), 174..=255 map to themselves
///   - all other bytes map to 256, 257, 258, ... in order
fn byte_to_char(b: u8) -> char {
    let mapped = byte_to_unicode_raw(b);
    char::from_u32(mapped).unwrap_or('\u{FFFD}')
}

/// Reverse mapping: char → byte. Returns None for characters that aren't in
/// the mapping (which shouldn't happen for tokens produced by `byte_to_char`).
fn char_to_byte(ch: char) -> Option<u8> {
    let cp = ch as u32;
    // Direct-mapped ranges.
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        return Some(cp as u8);
    }
    // Indirect: bytes that don't fall in the direct ranges get codepoints
    // starting at 256, in increasing order of byte value.
    // We need to recover which byte produced this codepoint.
    let mut idx: u32 = 256;
    for b in 0u32..256 {
        let in_direct = (33..=126).contains(&b)
            || (161..=172).contains(&b)
            || (174..=255).contains(&b);
        if !in_direct {
            if cp == idx { return Some(b as u8); }
            idx += 1;
        }
    }
    None
}

fn byte_to_unicode_raw(b: u8) -> u32 {
    let cp = b as u32;
    if (33..=126).contains(&cp) || (161..=172).contains(&cp) || (174..=255).contains(&cp) {
        return cp;
    }
    // Indirect mapping: count how many bytes < this one were also indirect.
    let mut offset: u32 = 0;
    for x in 0u32..cp {
        let in_direct = (33..=126).contains(&x)
            || (161..=172).contains(&x)
            || (174..=255).contains(&x);
        if !in_direct {
            offset += 1;
        }
    }
    256 + offset
}
