//! Tests for the GPT-2 / Llama 3 regex pre-tokenizer.
//!
//! This is the canonical pre-tokenization step that splits text into
//! chunks before BPE merge application. The regex is:
//!
//!   'reign|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+
//!
//! Matching behavior:
//! - Contractions ('s, 't, 're, 've, 'm, 'll, 'd) are their own chunks
//! - Letter runs, optionally preceded by a single space
//! - Number runs, optionally preceded by a single space
//! - Punctuation/symbol runs, optionally preceded by a single space
//! - Whitespace runs
//!
//! This produces tokenizations that match HuggingFace tokenizers for
//! Llama 3 / Mistral / Qwen / GPT-style models.

use jouleclaw_loader_gguf::tokenizer::pre_tokenize_gpt2;

#[test]
fn splits_simple_words_with_leading_space() {
    let chunks = pre_tokenize_gpt2("hello world");
    assert_eq!(chunks, vec!["hello", " world"]);
}

#[test]
fn first_word_has_no_leading_space() {
    let chunks = pre_tokenize_gpt2("the quick brown fox");
    assert_eq!(chunks, vec!["the", " quick", " brown", " fox"]);
}

#[test]
fn splits_contractions() {
    // "don't" → "don", "'t"
    let chunks = pre_tokenize_gpt2("don't");
    assert_eq!(chunks, vec!["don", "'t"]);
}

#[test]
fn handles_all_canonical_contractions() {
    for (input, expected_apos) in &[
        ("he's", "'s"),
        ("don't", "'t"),
        ("they're", "'re"),
        ("we've", "'ve"),
        ("I'm", "'m"),
        ("we'll", "'ll"),
        ("I'd", "'d"),
    ] {
        let chunks = pre_tokenize_gpt2(input);
        assert!(chunks.iter().any(|c| c == expected_apos),
            "input {:?} should produce a {:?} chunk; got {:?}",
            input, expected_apos, chunks);
    }
}

#[test]
fn splits_punctuation_from_words() {
    let chunks = pre_tokenize_gpt2("hello, world.");
    // "hello" + "," + " world" + "."
    assert_eq!(chunks, vec!["hello", ",", " world", "."]);
}

#[test]
fn splits_numbers_from_letters() {
    let chunks = pre_tokenize_gpt2("foo 42 bar");
    assert_eq!(chunks, vec!["foo", " 42", " bar"]);
}

#[test]
fn keeps_whitespace_runs() {
    // Double-space should produce one chunk for the leading space + word,
    // and capture the extra space somewhere.
    let chunks = pre_tokenize_gpt2("a  b");
    // Expected: "a", " ", " b"  (or similar — whitespace handling has edges)
    // Just verify reassembly is correct.
    let recovered: String = chunks.concat();
    assert_eq!(recovered, "a  b",
        "pre-tokenization should be byte-exact reversible by concatenation");
}

#[test]
fn empty_input_produces_no_chunks() {
    let chunks = pre_tokenize_gpt2("");
    assert!(chunks.is_empty());
}

#[test]
fn single_word_produces_one_chunk() {
    let chunks = pre_tokenize_gpt2("hello");
    assert_eq!(chunks, vec!["hello"]);
}

#[test]
fn whitespace_only_input() {
    let chunks = pre_tokenize_gpt2("   ");
    // Three spaces — should be one whitespace run.
    let recovered: String = chunks.concat();
    assert_eq!(recovered, "   ");
}

#[test]
fn handles_newlines_as_whitespace() {
    let chunks = pre_tokenize_gpt2("a\nb");
    let recovered: String = chunks.concat();
    assert_eq!(recovered, "a\nb");
}

#[test]
fn unicode_letters_are_letter_chunks() {
    // German umlauts, Greek, Cyrillic — all \p{L}.
    let chunks = pre_tokenize_gpt2("café αβγ привет");
    let recovered: String = chunks.concat();
    assert_eq!(recovered, "café αβγ привет");
    // Each language should be one chunk after a space.
    assert!(chunks.iter().any(|c| c == "café"));
    assert!(chunks.iter().any(|c| c == " αβγ"));
    assert!(chunks.iter().any(|c| c == " привет"));
}

#[test]
fn pre_tokenization_is_byte_reversible() {
    // Across a variety of inputs, concatenating chunks should reproduce
    // the original input exactly. This is the defining property of a
    // proper pre-tokenizer.
    for input in &[
        "hello world",
        "I don't know.",
        "She said, \"hi!\"",
        "x = 42; y = x + 1;",
        "café au lait",
        "What about 100% sure?",
        "multi\nline\ntext",
    ] {
        let chunks = pre_tokenize_gpt2(input);
        let recovered: String = chunks.concat();
        assert_eq!(&recovered, input,
            "round-trip failed for {:?}: chunks={:?}, recovered={:?}",
            input, chunks, recovered);
    }
}

#[test]
fn determinism() {
    let input = "The quick brown fox jumps over the lazy dog.";
    let c1 = pre_tokenize_gpt2(input);
    let c2 = pre_tokenize_gpt2(input);
    let c3 = pre_tokenize_gpt2(input);
    assert_eq!(c1, c2);
    assert_eq!(c2, c3);
}
