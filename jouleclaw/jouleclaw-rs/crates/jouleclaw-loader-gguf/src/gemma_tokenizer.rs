//! Minimal Gemma `tokenizer.json` reader.
//!
//! Faithful enough for the Gemma 4 family — and only Gemma. The
//! file's structure is:
//!
//!   * `normalizer`  = Replace `" "` → `"▁"` (single regular space → U+2581).
//!   * `pre_tokenizer` = `Split " "` `MergedWithPrevious` — after the
//!     normalizer there are no spaces left, so this is effectively
//!     a no-op single chunk.
//!   * `model.type = BPE`, `byte_fallback = true`, 262 144-entry vocab,
//!     ~515 K merges.
//!   * `post_processor` prepends `<bos>` (id 2) for the `single` template.
//!   * `decoder` = Sequence(Replace `"▁"`→" ", ByteFallback, Fuse).
//!
//! Oracle: `encode("The capital of France is", add_bos=true)` MUST
//! equal `[2, 818, 5279, 529, 7001, 563]` — the exact ids the HF
//! tokenizer produces and the verified Gemma 4 forward consumes.

use std::collections::HashMap;
use std::path::Path;

use crate::ParseError;

/// One Gemma BPE merge: two token-ids and the resulting token-id.
type MergedId = u32;

pub struct GemmaTokenizer {
    /// piece → id.
    vocab: HashMap<String, u32>,
    /// id → piece. Built once, used by [`Self::decode`].
    inv: Vec<String>,
    /// `(left_id, right_id) → merged_id`, keyed by merge rank
    /// (lower rank = higher priority); the rank lives implicitly in
    /// the `Vec<>` order via `merge_rank` for fast lookup.
    merge_to: HashMap<(u32, u32), MergedId>,
    /// `(left_id, right_id) → rank`. Smaller is preferred.
    merge_rank: HashMap<(u32, u32), u32>,
    /// id of each UTF-8 byte's `<0xHH>` piece (byte fallback).
    byte_token: [u32; 256],
    bos_id: u32,
    eos_id: u32,
}

fn tok_err(msg: impl Into<String>) -> ParseError {
    ParseError::Safetensors(format!("gemma tokenizer.json: {}", msg.into()))
}

impl GemmaTokenizer {
    /// Read and parse `tokenizer.json` from an HF snapshot directory.
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self, ParseError> {
        let txt = std::fs::read_to_string(path.as_ref()).map_err(crate::io_err)?;
        let v: serde_json::Value = serde_json::from_str(&txt)
            .map_err(|e| tok_err(format!("parse: {e}")))?;

        // ---- model.vocab ----
        let model = v.get("model").ok_or_else(|| tok_err("no .model"))?;
        if model.get("type").and_then(|x| x.as_str()) != Some("BPE") {
            return Err(tok_err("model.type != BPE"));
        }
        let voc = model
            .get("vocab")
            .and_then(|x| x.as_object())
            .ok_or_else(|| tok_err("no .model.vocab"))?;
        let mut vocab: HashMap<String, u32> = HashMap::with_capacity(voc.len());
        let mut inv: Vec<String> = vec![String::new(); voc.len()];
        for (tok, id_v) in voc {
            let id = id_v
                .as_u64()
                .ok_or_else(|| tok_err("non-int vocab id"))?
                as u32;
            if (id as usize) < inv.len() {
                inv[id as usize] = tok.clone();
            }
            vocab.insert(tok.clone(), id);
        }

        // ---- byte-fallback table: <0x00>..<0xFF> ----
        let mut byte_token = [u32::MAX; 256];
        for b in 0..=255u8 {
            let key = format!("<0x{b:02X}>");
            if let Some(&id) = vocab.get(&key) {
                byte_token[b as usize] = id;
            }
        }
        if byte_token.iter().any(|&x| x == u32::MAX) {
            return Err(tok_err("byte_fallback table incomplete"));
        }

        // ---- merges ----
        let merges = model
            .get("merges")
            .and_then(|x| x.as_array())
            .ok_or_else(|| tok_err("no .model.merges"))?;
        let mut merge_to: HashMap<(u32, u32), u32> =
            HashMap::with_capacity(merges.len());
        let mut merge_rank: HashMap<(u32, u32), u32> =
            HashMap::with_capacity(merges.len());
        for (rank, m) in merges.iter().enumerate() {
            let pair = m
                .as_array()
                .ok_or_else(|| tok_err("merge entry not [left,right]"))?;
            if pair.len() != 2 {
                return Err(tok_err("merge entry length != 2"));
            }
            let l = pair[0].as_str().ok_or_else(|| tok_err("merge left !str"))?;
            let r = pair[1].as_str().ok_or_else(|| tok_err("merge right !str"))?;
            let merged = format!("{l}{r}");
            // skip merges where any leg or the result is missing — these
            // are dead entries that real HF tokenizers also skip.
            if let (Some(&li), Some(&ri), Some(&mi)) =
                (vocab.get(l), vocab.get(r), vocab.get(&merged))
            {
                merge_to.entry((li, ri)).or_insert(mi);
                merge_rank.entry((li, ri)).or_insert(rank as u32);
            }
        }

        // ---- specials ----
        let bos_id = *vocab
            .get("<bos>")
            .ok_or_else(|| tok_err("no <bos> in vocab"))?;
        let eos_id = *vocab
            .get("<eos>")
            .ok_or_else(|| tok_err("no <eos> in vocab"))?;

        Ok(Self {
            vocab,
            inv,
            merge_to,
            merge_rank,
            byte_token,
            bos_id,
            eos_id,
        })
    }

    pub fn bos_id(&self) -> u32 {
        self.bos_id
    }
    pub fn eos_id(&self) -> u32 {
        self.eos_id
    }
    pub fn vocab_size(&self) -> usize {
        self.inv.len()
    }

    /// SentencePiece-style normalize: every ASCII space → `▁` (U+2581).
    fn normalize(s: &str) -> String {
        s.replace(' ', "\u{2581}")
    }

    /// Initial pieces: each Unicode character becomes one id if its
    /// `String` is in vocab, otherwise its UTF-8 bytes are emitted as
    /// `<0xHH>` byte-fallback ids (which never merge further — those
    /// pairs are absent from `merge_*`).
    fn initial_pieces(&self, s: &str) -> Vec<u32> {
        let mut out: Vec<u32> = Vec::with_capacity(s.len());
        let mut buf = [0u8; 4];
        for ch in s.chars() {
            let cs = ch.encode_utf8(&mut buf);
            if let Some(&id) = self.vocab.get(cs) {
                out.push(id);
            } else {
                for &b in cs.as_bytes() {
                    out.push(self.byte_token[b as usize]);
                }
            }
        }
        out
    }

    /// Greedy BPE: repeatedly merge the adjacent pair with the lowest
    /// rank until no applicable merge remains. Naive O(n²·passes)
    /// scan; fine for normal-length inputs.
    fn bpe(&self, mut ids: Vec<u32>) -> Vec<u32> {
        loop {
            let mut best_rank = u32::MAX;
            let mut best_i = 0usize;
            let mut best_merged = 0u32;
            let mut found = false;
            for i in 0..ids.len().saturating_sub(1) {
                if let Some(&r) = self.merge_rank.get(&(ids[i], ids[i + 1])) {
                    if r < best_rank {
                        best_rank = r;
                        best_i = i;
                        best_merged = *self.merge_to.get(&(ids[i], ids[i + 1])).unwrap();
                        found = true;
                    }
                }
            }
            if !found {
                break;
            }
            ids[best_i] = best_merged;
            ids.remove(best_i + 1);
        }
        ids
    }

    /// Encode a UTF-8 string. `add_bos=true` prepends `<bos>` per the
    /// `single` post-processing template Gemma 4 uses.
    pub fn encode(&self, text: &str, add_bos: bool) -> Vec<u32> {
        let s = Self::normalize(text);
        let pieces = self.initial_pieces(&s);
        let merged = self.bpe(pieces);
        if add_bos {
            let mut out = Vec::with_capacity(merged.len() + 1);
            out.push(self.bos_id);
            out.extend(merged);
            out
        } else {
            merged
        }
    }

    /// Decode a token-id sequence back to UTF-8 text. Implements the
    /// `Sequence(Replace "▁" → " ", ByteFallback, Fuse)` decoder:
    /// joins pieces, replaces `▁` with space, and folds consecutive
    /// `<0xHH>` byte-fallback tokens into actual UTF-8 bytes (which
    /// may straddle token boundaries — Fuse).
    pub fn decode(&self, ids: &[u32]) -> String {
        // First pass: build a Vec where each entry is either a
        // string piece (with `▁` already replaced) or a raw byte
        // from a `<0xHH>` token. Then flush adjacent bytes through
        // UTF-8 decoding (lossy on invalid sequences).
        enum Frag {
            Bytes(Vec<u8>),
            Text(String),
        }
        let mut frags: Vec<Frag> = Vec::new();
        for &id in ids {
            let piece = self
                .inv
                .get(id as usize)
                .map(|s| s.as_str())
                .unwrap_or("");
            // <0xHH> byte token?
            if piece.len() == 6
                && piece.starts_with("<0x")
                && piece.ends_with('>')
            {
                if let Ok(b) = u8::from_str_radix(&piece[3..5], 16) {
                    if let Some(Frag::Bytes(buf)) = frags.last_mut() {
                        buf.push(b);
                    } else {
                        frags.push(Frag::Bytes(vec![b]));
                    }
                    continue;
                }
            }
            // skip special added tokens like <bos>/<eos>/<pad>
            if piece.starts_with('<') && piece.ends_with('>') {
                continue;
            }
            let t = piece.replace('\u{2581}', " ");
            if let Some(Frag::Text(s)) = frags.last_mut() {
                s.push_str(&t);
            } else {
                frags.push(Frag::Text(t));
            }
        }
        let mut out = String::new();
        for f in frags {
            match f {
                Frag::Text(s) => out.push_str(&s),
                Frag::Bytes(b) => out.push_str(&String::from_utf8_lossy(&b)),
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const GEMMA4_E2B: &str =
        "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B";

    #[test]
    fn real_gemma4_tokenizer_matches_hf() {
        let p = std::path::Path::new(GEMMA4_E2B).join("tokenizer.json");
        if !p.exists() {
            eprintln!("skip: gemma-4-E2B not downloaded");
            return;
        }
        let tk = GemmaTokenizer::from_file(&p).expect("load tokenizer");
        // HF reference (from oracle dump): the Gemma 4 forward was
        // verified end-to-end against these exact ids.
        let ids = tk.encode("The capital of France is", true);
        assert_eq!(ids, vec![2u32, 818, 5279, 529, 7001, 563]);
        assert_eq!(tk.bos_id(), 2);
        assert_eq!(tk.eos_id(), 1);

        // Round-trip: decoding the HF greedy continuation should
        // reproduce the expected text.
        let cont: Vec<u32> = vec![9079, 236761, 108, 818, 5279, 529, 7001, 563];
        let txt = tk.decode(&cont);
        assert_eq!(txt, " Paris.\n\nThe capital of France is");
    }
}
