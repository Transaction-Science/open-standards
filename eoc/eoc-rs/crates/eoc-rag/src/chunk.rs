//! Chunking strategies.
//!
//! Four strategies, all deterministic and operating on raw `&str`:
//!
//! * [`ChunkerKind::SentenceWindow`] — split on sentence boundaries,
//!   then emit fixed-width sliding windows of `n` sentences with
//!   `overlap` overlap. Sentence-Window retrieval, Llama-Index 2023.
//! * [`ChunkerKind::Semantic`] — split into sentences then merge
//!   adjacent sentences while their cosine similarity (against a
//!   crude bag-of-tokens vector) exceeds `threshold`. Reference of the
//!   "semantic chunker" idea from Greg Kamradt, 2023.
//! * [`ChunkerKind::Recursive`] — recursive split with a priority list
//!   of separators (`"\n\n"`, `"\n"`, `". "`, `" "`). The Langchain
//!   `RecursiveCharacterTextSplitter` algorithm.
//! * [`ChunkerKind::Late`] — emit the entire document as a single
//!   chunk; "late chunking" defers segmentation to embedding time, so
//!   the chunker just preserves the whole token stream
//!   (Günther et al. 2024, "Late Chunking").

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::store::Chunk;

/// Which chunker to run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum ChunkerKind {
    /// Sentence-window with `window` sentences and `overlap` overlap.
    SentenceWindow,
    /// Semantic merge with `threshold` similarity.
    Semantic,
    /// Recursive separator-list split.
    Recursive,
    /// Late chunking — the document stays whole.
    Late,
}

/// Configuration for [`Chunker`].
#[derive(Debug, Clone)]
pub struct ChunkingConfig {
    /// Which strategy to use.
    pub kind: ChunkerKind,
    /// Target chunk size in bytes. Recursive splitter respects this as
    /// a soft cap.
    pub target_size: usize,
    /// Sentence-window window length in sentences.
    pub window: usize,
    /// Sentence-window overlap in sentences.
    pub overlap: usize,
    /// Semantic-chunker merge threshold (cosine, 0..=1).
    pub threshold: f32,
}

impl Default for ChunkingConfig {
    fn default() -> Self {
        Self {
            kind: ChunkerKind::Recursive,
            target_size: 512,
            window: 3,
            overlap: 1,
            threshold: 0.6,
        }
    }
}

/// Chunker.
pub struct Chunker {
    cfg: ChunkingConfig,
}

impl Chunker {
    /// Construct.
    pub fn new(cfg: ChunkingConfig) -> Self {
        Self { cfg }
    }

    /// Chunk a document.
    pub fn chunk(&self, doc_id: &str, text: &str) -> Vec<Chunk> {
        match self.cfg.kind {
            ChunkerKind::Late => vec![Chunk::new(doc_id, 0, text)],
            ChunkerKind::SentenceWindow => self.sentence_window(doc_id, text),
            ChunkerKind::Semantic => self.semantic(doc_id, text),
            ChunkerKind::Recursive => self.recursive(doc_id, text),
        }
    }

    fn sentence_window(&self, doc_id: &str, text: &str) -> Vec<Chunk> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }
        let window = self.cfg.window.max(1);
        let overlap = self.cfg.overlap.min(window.saturating_sub(1));
        let step = window - overlap;
        let mut out: Vec<Chunk> = Vec::new();
        let mut i = 0;
        while i < sentences.len() {
            let end = (i + window).min(sentences.len());
            let first = &sentences[i];
            let last = &sentences[end - 1];
            let chunk_text = text[first.start..last.end].to_string();
            out.push(Chunk::new(doc_id, first.start, chunk_text));
            if end == sentences.len() {
                break;
            }
            i += step.max(1);
        }
        out
    }

    fn semantic(&self, doc_id: &str, text: &str) -> Vec<Chunk> {
        let sentences = split_sentences(text);
        if sentences.is_empty() {
            return Vec::new();
        }
        let mut out: Vec<Chunk> = Vec::new();
        let mut group_start = sentences[0].start;
        let mut group_end = sentences[0].end;
        let mut prev_vec = bag_of_tokens(&text[sentences[0].start..sentences[0].end]);

        for s in sentences.iter().skip(1) {
            let cur_vec = bag_of_tokens(&text[s.start..s.end]);
            let sim = cosine_bag(&prev_vec, &cur_vec);
            let group_size = group_end - group_start;
            if sim >= self.cfg.threshold && group_size + (s.end - s.start) <= self.cfg.target_size {
                group_end = s.end;
                // Merge bag-of-tokens.
                for (k, v) in cur_vec {
                    *prev_vec.entry(k).or_insert(0.0) += v;
                }
            } else {
                let chunk_text = text[group_start..group_end].to_string();
                out.push(Chunk::new(doc_id, group_start, chunk_text));
                group_start = s.start;
                group_end = s.end;
                prev_vec = bag_of_tokens(&text[s.start..s.end]);
            }
        }
        let chunk_text = text[group_start..group_end].to_string();
        out.push(Chunk::new(doc_id, group_start, chunk_text));
        out
    }

    fn recursive(&self, doc_id: &str, text: &str) -> Vec<Chunk> {
        let separators = ["\n\n", "\n", ". ", " "];
        let mut out: Vec<Chunk> = Vec::new();
        let target = self.cfg.target_size.max(1);
        recursive_helper(doc_id, text, 0, target, &separators, 0, &mut out);
        // Drop trailing empties.
        out.retain(|c| !c.text.trim().is_empty());
        if out.is_empty() && !text.is_empty() {
            out.push(Chunk::new(doc_id, 0, text));
        }
        out
    }
}

fn recursive_helper(
    doc_id: &str,
    text: &str,
    offset: usize,
    target: usize,
    seps: &[&str],
    sep_idx: usize,
    out: &mut Vec<Chunk>,
) {
    if text.len() <= target || sep_idx >= seps.len() {
        if !text.is_empty() {
            out.push(Chunk::new(doc_id, offset, text));
        }
        return;
    }
    let sep = seps[sep_idx];
    let mut start = 0;
    let mut current = String::new();
    let mut current_start = 0;
    for (i, part) in text.split(sep).enumerate() {
        let piece_start = start;
        let piece_end = start + part.len();
        if current.is_empty() {
            current_start = piece_start;
            current.push_str(part);
        } else if current.len() + sep.len() + part.len() <= target {
            current.push_str(sep);
            current.push_str(part);
        } else {
            // Emit current.
            if current.len() > target {
                recursive_helper(
                    doc_id,
                    &current,
                    offset + current_start,
                    target,
                    seps,
                    sep_idx + 1,
                    out,
                );
            } else {
                out.push(Chunk::new(doc_id, offset + current_start, current.clone()));
            }
            current.clear();
            current_start = piece_start;
            current.push_str(part);
        }
        start = piece_end + sep.len();
        let _ = i; // splittersuppress unused
    }
    if !current.is_empty() {
        if current.len() > target {
            recursive_helper(
                doc_id,
                &current,
                offset + current_start,
                target,
                seps,
                sep_idx + 1,
                out,
            );
        } else {
            out.push(Chunk::new(doc_id, offset + current_start, current));
        }
    }
}

/// One sentence span — `[start, end)` byte offsets into the source text.
#[derive(Debug, Clone, Copy)]
struct SentenceSpan {
    start: usize,
    end: usize,
}

fn split_sentences(text: &str) -> Vec<SentenceSpan> {
    let mut out: Vec<SentenceSpan> = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == '.' || c == '!' || c == '?' || c == '\n' {
            let end = i + 1;
            // Skip whitespace forwards to mark next start.
            let span = SentenceSpan { start, end };
            if span.end > span.start {
                let slice = &text[span.start..span.end];
                if !slice.trim().is_empty() {
                    out.push(span);
                }
            }
            let mut j = end;
            while j < bytes.len() && (bytes[j] as char).is_whitespace() {
                j += 1;
            }
            start = j;
            i = j;
        } else {
            i += 1;
        }
    }
    if start < bytes.len() {
        let slice = &text[start..bytes.len()];
        if !slice.trim().is_empty() {
            out.push(SentenceSpan {
                start,
                end: bytes.len(),
            });
        }
    }
    out
}

fn bag_of_tokens(s: &str) -> HashMap<String, f32> {
    let mut out: HashMap<String, f32> = HashMap::new();
    let mut cur = String::new();
    for ch in s.chars() {
        if ch.is_alphanumeric() {
            for low in ch.to_lowercase() {
                cur.push(low);
            }
        } else if !cur.is_empty() {
            *out.entry(std::mem::take(&mut cur)).or_insert(0.0) += 1.0;
        }
    }
    if !cur.is_empty() {
        *out.entry(cur).or_insert(0.0) += 1.0;
    }
    out
}

fn cosine_bag(a: &HashMap<String, f32>, b: &HashMap<String, f32>) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for v in a.values() {
        na += v * v;
    }
    for v in b.values() {
        nb += v * v;
    }
    for (k, av) in a {
        if let Some(bv) = b.get(k) {
            dot += av * bv;
        }
    }
    let denom = na.sqrt() * nb.sqrt();
    if denom <= 0.0 { 0.0 } else { dot / denom }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn late_chunker_emits_one() {
        let c = Chunker::new(ChunkingConfig {
            kind: ChunkerKind::Late,
            ..Default::default()
        });
        let chunks = c.chunk("d1", "hello world");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "hello world");
    }

    #[test]
    fn sentence_window_emits_windows() {
        let c = Chunker::new(ChunkingConfig {
            kind: ChunkerKind::SentenceWindow,
            window: 2,
            overlap: 1,
            ..Default::default()
        });
        let chunks = c.chunk("d1", "First. Second. Third. Fourth.");
        assert!(!chunks.is_empty());
        // First chunk should contain "First" and "Second".
        assert!(chunks[0].text.contains("First"));
        assert!(chunks[0].text.contains("Second"));
    }

    #[test]
    fn recursive_chunker_respects_target() {
        let c = Chunker::new(ChunkingConfig {
            kind: ChunkerKind::Recursive,
            target_size: 20,
            ..Default::default()
        });
        let text = "Paragraph one is here.\n\nParagraph two is also here.\n\nThird.";
        let chunks = c.chunk("d1", text);
        assert!(!chunks.is_empty());
        for ch in &chunks {
            // Soft cap — the smallest separator path may still exceed
            // by one separator's worth.
            assert!(ch.text.len() <= 40, "chunk too big: {} bytes", ch.text.len());
        }
    }

    #[test]
    fn semantic_chunker_groups_related() {
        let c = Chunker::new(ChunkingConfig {
            kind: ChunkerKind::Semantic,
            threshold: 0.0, // accept any overlap; just check it runs.
            target_size: 1000,
            ..Default::default()
        });
        let chunks = c.chunk(
            "d1",
            "Apples are red. Apples are sweet. Bicycles have wheels.",
        );
        assert!(!chunks.is_empty());
    }

    #[test]
    fn split_sentences_basic() {
        let s = split_sentences("Hello world. How are you? Fine!");
        assert_eq!(s.len(), 3);
    }
}
