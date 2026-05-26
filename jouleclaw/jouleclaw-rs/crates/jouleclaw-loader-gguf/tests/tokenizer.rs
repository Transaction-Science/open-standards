//! Tokenizer tests: SPM (SentencePiece BPE) algorithm.
//!
//! Each test builds a small vocabulary by hand, embeds it in a synthetic
//! GGUF, parses the GGUF, and verifies the tokenizer produces the expected
//! token IDs.

use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::read_gguf;
use std::io::Cursor;

/// Build a Vocab from a list of (token, score) pairs by going through the
/// synthetic GGUF round-trip. This exercises the same code path real models use.
fn vocab_from_tokens(
    pairs: Vec<(&str, f32)>,
    bos: Option<u32>,
    eos: Option<u32>,
    unk: Option<u32>,
) -> Vocab {
    let owned: Vec<(String, f32)> = pairs.into_iter()
        .map(|(t, s)| (t.to_string(), s)).collect();
    let cfg = SyntheticConfig {
        vocab_size: owned.len(),
        embedding_length: 8,
        block_count: 1,
        feed_forward_length: 16,
        head_count: 1,
        head_count_kv: 1,
        rms_eps: 1e-6,
        seed: 1,
        vocab: Some(owned),
        merges: None,
        bos_id: bos,
        eos_id: eos,
        unk_id: unk, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).expect("parse synthetic gguf with vocab");
    Vocab::from_gguf(&model).expect("load vocab from gguf")
}

/// Vocabulary loads correctly from GGUF metadata.
#[test]
fn vocab_loads_from_gguf() {
    let v = vocab_from_tokens(
        vec![("<unk>", 0.0), ("<s>", 0.0), ("</s>", 0.0), ("a", -1.0), ("b", -2.0)],
        Some(1), Some(2), Some(0),
    );
    assert_eq!(v.len(), 5);
    assert_eq!(v.id_to_token(3), Some("a"));
    assert_eq!(v.id_to_token(4), Some("b"));
    assert_eq!(v.bos_id, Some(1));
    assert_eq!(v.eos_id, Some(2));
    assert_eq!(v.unk_id, Some(0));
    assert_eq!(v.scores[3], -1.0);
}

/// Encode with no merges available: each prepared char becomes its own token.
/// Vocabulary contains exactly `▁`, `a`, `b`, `c`, `<unk>`.
/// Input "abc" → prepared "▁abc" → tokens ["▁", "a", "b", "c"].
#[test]
fn encode_with_no_merges_emits_per_char_tokens() {
    let v = vocab_from_tokens(
        vec![
            ("<unk>", 0.0),    // 0
            ("\u{2581}", 0.0), // 1: prefix space
            ("a", -1.0),       // 2
            ("b", -2.0),       // 3
            ("c", -3.0),       // 4
        ],
        None, None, Some(0),
    );

    let ids = v.encode_spm("abc", false);
    assert_eq!(ids, vec![1, 2, 3, 4]);
}

/// Encode where a merged token is preferred: vocabulary has both "a" and "ab",
/// with "ab" having higher score. The tokenizer should emit "ab" instead of
/// two separate tokens.
#[test]
fn encode_prefers_higher_scoring_merge() {
    let v = vocab_from_tokens(
        vec![
            ("<unk>", 0.0),       // 0
            ("\u{2581}", 0.0),    // 1
            ("a", -10.0),         // 2: low priority
            ("b", -10.0),         // 3
            ("ab", -1.0),         // 4: high priority (preferred merge)
        ],
        None, None, Some(0),
    );

    let ids = v.encode_spm("ab", false);
    // After merge: ▁, ab → tokens 1, 4.
    assert_eq!(ids, vec![1, 4],
        "expected merge of a+b -> 'ab' since it has higher score");
}

/// Multi-step merging: vocabulary contains "ab", "bc", "abc". Score order:
/// "abc" > "ab" > "bc". Input "abc" should become a single token because
/// the algorithm finds the merged "ab"+"c" path through "abc".
///
/// The SPM algorithm only merges adjacent symbols; it can't directly
/// produce "abc" from individual chars. It would first merge "ab" or "bc"
/// (whichever has higher score), then merge the result with the remaining
/// char if a vocabulary entry exists for that combined string.
#[test]
fn encode_iterative_merge_to_three_chars() {
    let v = vocab_from_tokens(
        vec![
            ("<unk>", 0.0),       // 0
            ("\u{2581}", 0.0),    // 1
            ("a", -10.0),         // 2
            ("b", -10.0),         // 3
            ("c", -10.0),         // 4
            ("ab", -2.0),         // 5
            ("bc", -3.0),         // 6: lower than "ab"
            ("abc", -1.0),        // 7: highest
        ],
        None, None, Some(0),
    );

    let ids = v.encode_spm("abc", false);
    // Step 1: bigrams (a,b)=score(-2), (b,c)=score(-3). Pick (a,b) → "ab".
    // Step 2: bigram (ab,c) → "abc" with score(-1). Pick it → single "abc" token.
    // Final emit: ▁, abc → tokens 1, 7.
    assert_eq!(ids, vec![1, 7],
        "should iteratively merge a+b then ab+c into single 'abc' token");
}

/// Spaces in input are converted to U+2581. "a b" → "▁a▁b".
#[test]
fn encode_converts_spaces() {
    let v = vocab_from_tokens(
        vec![
            ("<unk>", 0.0),       // 0
            ("\u{2581}", 0.0),    // 1
            ("a", -1.0),          // 2
            ("b", -2.0),          // 3
            ("\u{2581}a", -0.5),  // 4: ▁a as a merged token
            ("\u{2581}b", -0.5),  // 5: ▁b as a merged token
        ],
        None, None, Some(0),
    );

    let ids = v.encode_spm("a b", false);
    // Prepared: ▁a▁b. Bigrams: (▁,a)=-0.5, (a,▁)=none, (▁,b)=-0.5.
    // Merge (▁,a) → ▁a (token 4). Then (▁a,▁) = none, (▁,b) = -0.5.
    // Merge (▁,b) → ▁b (token 5). Done.
    // Final tokens: ▁a, ▁b → 4, 5.
    assert_eq!(ids, vec![4, 5]);
}

/// BOS prepending works.
#[test]
fn encode_prepends_bos_when_requested() {
    let v = vocab_from_tokens(
        vec![
            ("<unk>", 0.0),
            ("<s>", 0.0),
            ("\u{2581}", 0.0),
            ("a", -1.0),
        ],
        Some(1), None, Some(0),
    );

    let with_bos = v.encode_spm("a", true);
    let without_bos = v.encode_spm("a", false);

    assert_eq!(with_bos[0], 1, "first token should be BOS");
    assert_eq!(&with_bos[1..], &without_bos[..]);
}

/// Byte fallback: a character not in the vocabulary becomes its UTF-8 bytes
/// emitted as `<0xNN>` tokens.
#[test]
fn encode_falls_back_to_bytes_for_unknown_chars() {
    // Build a vocab with only ASCII letters + byte tokens for non-ASCII.
    // Test char "©" = U+00A9 = UTF-8 bytes 0xC2 0xA9.
    let mut tokens = vec![
        ("<unk>".to_string(), 0.0),
        ("\u{2581}".to_string(), 0.0),
        ("a".to_string(), -1.0),
    ];
    // Add all 256 byte tokens so any UTF-8 byte can be emitted.
    for b in 0..256u32 {
        tokens.push((format!("<0x{:02X}>", b), -100.0));
    }
    let owned: Vec<(String, f32)> = tokens.iter().cloned().collect();
    let pairs: Vec<(&str, f32)> = owned.iter()
        .map(|(s, sc)| (s.as_str(), *sc)).collect();

    let v = vocab_from_tokens(pairs, None, None, Some(0));

    let ids = v.encode_spm("a©", false);
    // Expected: ▁ (1), a (2), <0xC2> (?), <0xA9> (?).
    // Find the IDs of <0xC2> and <0xA9>.
    let c2_id = v.token_to_id.get("<0xC2>").copied().unwrap();
    let a9_id = v.token_to_id.get("<0xA9>").copied().unwrap();
    assert_eq!(ids, vec![1, 2, c2_id, a9_id],
        "non-ASCII char without dedicated token should fall back to UTF-8 byte tokens");
}

/// Decoding reverses the U+2581 substitution.
#[test]
fn decode_replaces_spm_marker_with_space() {
    let v = vocab_from_tokens(
        vec![
            ("<unk>", 0.0),       // 0
            ("\u{2581}hello", -1.0),  // 1
            ("\u{2581}world", -1.0),  // 2
        ],
        None, None, Some(0),
    );

    let text = v.decode_spm(&[1, 2]);
    assert_eq!(text, "hello world",
        "SPM ▁ markers should decode to spaces, with leading space trimmed");
}

/// Encoding is deterministic: same input always produces same token IDs.
#[test]
fn encode_is_deterministic() {
    let v = vocab_from_tokens(
        vec![
            ("<unk>", 0.0),
            ("\u{2581}", 0.0),
            ("a", -1.0), ("b", -1.0), ("c", -1.0),
            ("ab", -0.5), ("bc", -0.5),
        ],
        None, None, Some(0),
    );

    let ids1 = v.encode_spm("abc", false);
    let ids2 = v.encode_spm("abc", false);
    let ids3 = v.encode_spm("abc", false);
    assert_eq!(ids1, ids2);
    assert_eq!(ids2, ids3);
}

/// Tie-break in merge selection: when two bigrams have equal score, the
/// leftmost one is chosen. This matches SPM's reference behavior.
#[test]
fn encode_tiebreak_prefers_leftmost() {
    let v = vocab_from_tokens(
        vec![
            ("<unk>", 0.0),       // 0
            ("\u{2581}", 0.0),    // 1
            ("a", -10.0),         // 2
            ("b", -10.0),         // 3
            ("aa", -1.0),         // 4: same score as bb
            ("bb", -1.0),         // 5
        ],
        None, None, Some(0),
    );

    // Input "aabb": prepared "▁aabb". Bigrams: (▁,a)=none, (a,a)=-1, (a,b)=none, (b,b)=-1.
    // Both (a,a) and (b,b) have score -1. Leftmost wins: (a,a) merges first → "aa".
    // After merge: ▁,aa,b,b. New bigrams: (▁,aa)=none, (aa,b)=none, (b,b)=-1.
    // Then (b,b) merges → "bb". Final: ▁, aa, bb → 1, 4, 5.
    let ids = v.encode_spm("aabb", false);
    assert_eq!(ids, vec![1, 4, 5]);
}
