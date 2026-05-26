//! BPE tokenizer tests (Llama 3 / Mistral / Qwen / GPT-style).
//!
//! Each test builds a small vocabulary + merge table and verifies the
//! algorithm picks the correct rank-ordered merges.

use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::read_gguf;
use std::io::Cursor;

/// Build a Vocab with both vocabulary and merges by going through the
/// synthetic GGUF round-trip.
fn vocab_with_merges(
    tokens: Vec<(&str, f32)>,
    merges: Vec<(&str, &str)>,
    bos: Option<u32>,
    eos: Option<u32>,
    unk: Option<u32>,
) -> Vocab {
    let owned_vocab: Vec<(String, f32)> = tokens.into_iter()
        .map(|(t, s)| (t.to_string(), s)).collect();
    let owned_merges: Vec<(String, String)> = merges.into_iter()
        .map(|(l, r)| (l.to_string(), r.to_string())).collect();
    let cfg = SyntheticConfig {
        vocab_size: owned_vocab.len(),
        embedding_length: 8,
        block_count: 1,
        feed_forward_length: 16,
        head_count: 1,
        head_count_kv: 1,
        rms_eps: 1e-6,
        seed: 1,
        vocab: Some(owned_vocab),
        merges: Some(owned_merges),
        bos_id: bos,
        eos_id: eos,
        unk_id: unk, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).expect("parse gguf");
    Vocab::from_gguf(&model).expect("load vocab")
}

/// Merges are loaded from GGUF and exposed via the rank lookup.
#[test]
fn merges_load_from_gguf() {
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("ab", 0.0), ("bc", 0.0)],
        vec![("a", "b"), ("b", "c")],
        None, None, None,
    );
    let merges = v.bpe_merges.as_ref().expect("merges should be loaded");
    assert_eq!(merges.merges.len(), 2);
    assert_eq!(merges.rank("a", "b"), Some(0));
    assert_eq!(merges.rank("b", "c"), Some(1));
    assert_eq!(merges.rank("a", "c"), None);
}

/// With no merges in the table, BPE emits per-byte tokens.
#[test]
fn bpe_with_empty_merge_table_emits_per_byte() {
    // Vocabulary covers single-character tokens for "abc" via byte mapping.
    // 'a' = 0x61, 'b' = 0x62, 'c' = 0x63 — all in the direct-mapped range
    // (33..=126), so the byte-to-char produces 'a', 'b', 'c'.
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("c", 0.0)],
        vec![],  // empty merges
        None, None, None,
    );
    // No prefix space; we want exact "abc" -> [a, b, c].
    let ids = v.encode_bpe("abc", false, false);
    assert_eq!(ids, vec![0, 1, 2],
        "with no merges, should emit each byte as a token");
}

/// Single merge applies: vocabulary has "a", "b", "ab"; merge table contains
/// ("a", "b"). Input "ab" should produce a single token for "ab".
#[test]
fn bpe_single_merge_combines_pair() {
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("ab", 0.0)],
        vec![("a", "b")],
        None, None, None,
    );
    let ids = v.encode_bpe("ab", false, false);
    assert_eq!(ids, vec![2], "should merge a+b into single 'ab' token (id 2)");
}

/// Lower rank wins over higher rank.
///
/// Setup: vocab has a, b, c, ab, bc; merges are ("b", "c") at rank 0
/// (highest priority) and ("a", "b") at rank 1.
/// Input "abc": both pairs are mergeable, but ("b", "c") has rank 0,
/// so it merges first. Result: "a" + "bc".
#[test]
fn bpe_lower_rank_wins() {
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("ab", 0.0), ("bc", 0.0)],
        vec![("b", "c"), ("a", "b")],  // rank 0 = (b, c)
        None, None, None,
    );
    let ids = v.encode_bpe("abc", false, false);
    // rank 0 (b,c) merges first: a, bc. No more merges.
    // Expected tokens: a (id 0), bc (id 4).
    assert_eq!(ids, vec![0, 4]);
}

/// Multi-step merges: rank 0 = (a, b), rank 1 = (ab, c). Input "abc":
/// step 1 merges (a, b) → ab; step 2 merges (ab, c) → abc.
#[test]
fn bpe_iterative_merge_through_two_steps() {
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("c", 0.0),
             ("ab", 0.0), ("abc", 0.0)],
        vec![("a", "b"), ("ab", "c")],
        None, None, None,
    );
    let ids = v.encode_bpe("abc", false, false);
    assert_eq!(ids, vec![4], "should merge a+b→ab then ab+c→abc");
}

/// BOS prepending in BPE.
#[test]
fn bpe_prepends_bos_when_requested() {
    let v = vocab_with_merges(
        vec![("<s>", 0.0), ("a", 0.0)],
        vec![],
        Some(0), None, None,
    );
    let with_bos = v.encode_bpe("a", true, false);
    let without_bos = v.encode_bpe("a", false, false);
    assert_eq!(with_bos.first(), Some(&0));
    assert_eq!(&with_bos[1..], &without_bos[..]);
}

/// BPE round-trip: encode then decode should recover the original text.
#[test]
fn bpe_round_trips_simple_input() {
    // Vocabulary covers 'h', 'e', 'l', 'o' as single-byte tokens; merges
    // create "ll" and "ello" so "hello" tokenizes to ["h", "ello"].
    let v = vocab_with_merges(
        vec![
            ("h", 0.0),    // 0
            ("e", 0.0),    // 1
            ("l", 0.0),    // 2
            ("o", 0.0),    // 3
            ("ll", 0.0),   // 4
            ("ello", 0.0), // 5
        ],
        vec![("l", "l"), ("e", "ll"), ("ell", "o")],
        None, None, None,
    );
    let ids = v.encode_bpe("hello", false, false);
    let decoded = v.decode_bpe(&ids);
    assert_eq!(decoded, "hello",
        "BPE round-trip failed: tokens={:?}, decoded='{}'", ids, decoded);
}

/// BPE encoding is deterministic.
#[test]
fn bpe_is_deterministic() {
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("ab", 0.0)],
        vec![("a", "b")],
        None, None, None,
    );
    let ids1 = v.encode_bpe("abcabc", false, false);
    let ids2 = v.encode_bpe("abcabc", false, false);
    let ids3 = v.encode_bpe("abcabc", false, false);
    assert_eq!(ids1, ids2);
    assert_eq!(ids2, ids3);
}

/// Whitespace pre-tokenization splits input. "a b c" should produce three
/// chunks: "a", " b", " c" (each non-first gets a leading space).
#[test]
fn bpe_pre_tokenizes_on_whitespace() {
    // Build a vocabulary covering 'a', 'b', 'c', and " a", " b", " c"
    // (with the space-as-Ġ encoding from GPT-2 byte mapping).
    // Space is byte 0x20 = 32, which is NOT in the direct-mapped ranges
    // (33..=126), so it maps to a high-Unicode codepoint. We need to use
    // the actual character produced by byte_to_char(0x20).
    //
    // For 0x20 = 32: not in direct ranges. Codepoint = 256 + (count of
    // bytes < 32 that are also indirect). Bytes 0..=32 are all NOT in
    // direct ranges (33..=126 starts at 33). So 0x20 = 32 → 256 + 32 = 288 = 'Ġ'.
    let space_marker = char::from_u32(288).unwrap().to_string();

    let mut tokens = vec![("a".to_string(), 0.0), ("b".to_string(), 0.0), ("c".to_string(), 0.0)];
    tokens.push((format!("{}b", space_marker), 0.0));  // " b"
    tokens.push((format!("{}c", space_marker), 0.0));  // " c"
    tokens.push((space_marker.clone(), 0.0));          // " " alone

    let owned_vocab: Vec<(String, f32)> = tokens.iter().cloned().collect();
    let pairs: Vec<(&str, f32)> = owned_vocab.iter()
        .map(|(s, sc)| (s.as_str(), *sc)).collect();

    // Merges: combine the space marker with each letter.
    let m1 = format!("{}b", space_marker);
    let m2 = format!("{}c", space_marker);
    let merges_owned: Vec<(String, String)> = vec![
        (space_marker.clone(), "b".to_string()),
        (space_marker.clone(), "c".to_string()),
    ];
    let merge_pairs: Vec<(&str, &str)> = merges_owned.iter()
        .map(|(l, r)| (l.as_str(), r.as_str())).collect();
    let _ = (&m1, &m2);  // referenced only for documentation

    let v = vocab_with_merges(pairs, merge_pairs, None, None, None);
    let ids = v.encode_bpe("a b c", false, false);

    // Expected chunks: "a", " b", " c". Each chunk is then BPE-merged.
    // Tokens: "a"=0, " b"=3, " c"=4.
    assert_eq!(ids, vec![0, 3, 4],
        "expected tokens for chunks [a], [ b], [ c], got {:?}", ids);
}

/// Input with no whitespace tokenizes as a single chunk.
#[test]
fn bpe_single_chunk_no_whitespace() {
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("abc", 0.0)],
        vec![("a", "b"), ("ab", "c")],
        None, None, None,
    );
    let ids = v.encode_bpe("abc", false, false);
    assert_eq!(ids, vec![3]);
}

/// Empty input produces empty output (or just BOS if requested).
#[test]
fn bpe_empty_input() {
    let v = vocab_with_merges(
        vec![("<s>", 0.0)],
        vec![],
        Some(0), None, None,
    );
    let no_bos = v.encode_bpe("", false, false);
    assert!(no_bos.is_empty());
    let with_bos = v.encode_bpe("", true, false);
    assert_eq!(with_bos, vec![0]);
}
