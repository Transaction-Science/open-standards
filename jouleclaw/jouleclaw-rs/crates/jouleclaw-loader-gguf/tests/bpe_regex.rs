//! Tests for `encode_bpe_regex` — the regex pre-tokenizer version of BPE.
//!
//! These tests focus on cases where the regex pre-tokenizer produces
//! different (more correct) results than the simple whitespace
//! pre-tokenizer used by `encode_bpe`.

use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::tokenizer::{pre_tokenize_gpt2, Vocab};
use jouleclaw_loader_gguf::read_gguf;
use std::io::Cursor;

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
        embedding_length: 8, block_count: 1,
        feed_forward_length: 16, head_count: 1, head_count_kv: 1,
        rms_eps: 1e-6, seed: 1,
        vocab: Some(owned_vocab),
        merges: Some(owned_merges),
        bos_id: bos, eos_id: eos, unk_id: unk, chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).expect("parse gguf");
    Vocab::from_gguf(&model).expect("load vocab")
}

/// Contractions split correctly. "don't" should produce chunks "don" and "'t",
/// each of which is then BPE-encoded.
#[test]
fn regex_bpe_splits_contractions() {
    // The pre-tokenizer chunks "don't" into ["don", "'t"], so we need
    // vocabulary entries for "don" and "'t".
    let v = vocab_with_merges(
        vec![
            ("d", 0.0), ("o", 0.0), ("n", 0.0), ("t", 0.0),
            ("'", 0.0),
            ("don", 0.0),
            ("'t", 0.0),
        ],
        vec![("d", "o"), ("do", "n"), ("'", "t")],
        None, None, None,
    );

    let chunks = pre_tokenize_gpt2("don't");
    assert_eq!(chunks, vec!["don", "'t"]);

    let ids = v.encode_bpe_regex("don't", false);
    // After merging in each chunk:
    //   chunk "don": d+o=do, do+n=don → "don" (id 5)
    //   chunk "'t":  '+t='t → "'t" (id 6)
    assert_eq!(ids, vec![5, 6]);
}

/// Punctuation gets its own chunks distinct from adjacent words.
#[test]
fn regex_bpe_separates_punctuation() {
    let v = vocab_with_merges(
        vec![("a", 0.0), (",", 0.0), (" ", 0.0), (" b", 0.0), (".", 0.0)],
        vec![],
        None, None, None,
    );

    let chunks = pre_tokenize_gpt2("a, b.");
    // Expected: "a", ",", " b", "."
    assert_eq!(chunks, vec!["a", ",", " b", "."]);
}

/// Whitespace-based and regex-based encoders agree on simple inputs
/// (single word, no punctuation, no contractions).
#[test]
fn regex_bpe_matches_whitespace_bpe_on_simple_input() {
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("c", 0.0), ("ab", 0.0), ("abc", 0.0)],
        vec![("a", "b"), ("ab", "c")],
        None, None, None,
    );

    let regex_ids = v.encode_bpe_regex("abc", false);
    let whitespace_ids = v.encode_bpe("abc", false, false);

    // Both should produce a single "abc" token via the two merges.
    assert_eq!(regex_ids, vec![4]);
    assert_eq!(whitespace_ids, vec![4]);
}

/// BOS prepending works with the regex encoder.
#[test]
fn regex_bpe_prepends_bos() {
    let v = vocab_with_merges(
        vec![("<s>", 0.0), ("a", 0.0)],
        vec![],
        Some(0), None, None,
    );
    let with_bos = v.encode_bpe_regex("a", true);
    let without_bos = v.encode_bpe_regex("a", false);
    assert_eq!(with_bos.first(), Some(&0));
    assert_eq!(&with_bos[1..], &without_bos[..]);
}

/// Round-trip: encode then decode should preserve the original ASCII text.
#[test]
fn regex_bpe_round_trips() {
    // Vocabulary covers each letter individually plus the space marker
    // and a few merged forms.
    // Space byte 0x20 maps to char codepoint 288 = 'Ġ' under the GPT-2
    // byte-to-char mapping.
    let space_mark = char::from_u32(288).unwrap().to_string();

    let mut tokens: Vec<(String, f32)> = "abcdefghijklmnopqrstuvwxyz"
        .chars().map(|c| (c.to_string(), 0.0)).collect();
    tokens.push((space_mark.clone(), 0.0));
    // Add " w", " h" forms for "hello world" round-trip
    tokens.push((format!("{}w", space_mark), 0.0));
    let owned = tokens.iter().cloned().collect::<Vec<_>>();
    let pairs: Vec<(&str, f32)> = owned.iter()
        .map(|(s, sc)| (s.as_str(), *sc)).collect();
    let v = vocab_with_merges(pairs, vec![(&space_mark, "w")], None, None, None);

    let ids = v.encode_bpe_regex("hello world", false);
    let decoded = v.decode_bpe(&ids);
    assert_eq!(decoded, "hello world",
        "regex BPE should round-trip ASCII text exactly");
}

/// Determinism.
#[test]
fn regex_bpe_is_deterministic() {
    let v = vocab_with_merges(
        vec![("a", 0.0), ("b", 0.0), ("c", 0.0)],
        vec![],
        None, None, None,
    );
    let ids1 = v.encode_bpe_regex("abc", false);
    let ids2 = v.encode_bpe_regex("abc", false);
    let ids3 = v.encode_bpe_regex("abc", false);
    assert_eq!(ids1, ids2);
    assert_eq!(ids2, ids3);
}

/// Empty input handled gracefully (with or without BOS).
#[test]
fn regex_bpe_empty_input() {
    let v = vocab_with_merges(
        vec![("<s>", 0.0)],
        vec![],
        Some(0), None, None,
    );
    let no_bos = v.encode_bpe_regex("", false);
    assert!(no_bos.is_empty());
    let with_bos = v.encode_bpe_regex("", true);
    assert_eq!(with_bos, vec![0]);
}
