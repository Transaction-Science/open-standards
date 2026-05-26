//! End-to-end demo: text in → token IDs → graph → logits → top-k token IDs out.
//!
//! Builds a synthetic Llama-style model with an embedded vocabulary,
//! tokenizes a string, runs the forward pass through the full pipeline
//! (RoPE, scaling, causal mask, multi-head), and decodes the argmax token
//! per position back to text.
//!
//! The vocabulary and weights are random, so the output is gibberish, but
//! the *pipeline* is fully exercised: every primitive runs, the tokenizer
//! produces token IDs from real characters, the forward pass produces
//! logits with the correct shape, and the output decodes back to a string.

use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use jouleclaw_loader_gguf::llama::build_llama_graph;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::io::Cursor;

/// Build a minimal vocabulary that can tokenize ASCII letters and spaces.
/// Vocab layout:
///   0..3:    <unk>, <s>, </s>
///   3:       ▁ (space marker, score 0)
///   4..30:   a..z (score -1.0 each)
///   30..62:  ▁a..▁z (score -0.5 each, prefer "word-start" forms)
fn build_test_vocab() -> Vec<(String, f32)> {
    let mut v = vec![
        ("<unk>".to_string(), 0.0),
        ("<s>".to_string(), 0.0),
        ("</s>".to_string(), 0.0),
        ("\u{2581}".to_string(), 0.0),
    ];
    for c in 'a'..='z' {
        v.push((c.to_string(), -1.0));
    }
    for c in 'a'..='z' {
        v.push((format!("\u{2581}{}", c), -0.5));
    }
    // Pad with byte tokens so non-ASCII inputs can still be encoded.
    for b in 0..256u32 {
        let key = format!("<0x{:02X}>", b);
        // Avoid duplicates if any of these tokens collide with already-added.
        // None of the above tokens look like `<0xNN>`, so we're safe.
        v.push((key, -10.0));
    }
    v
}

#[test]
fn text_in_tokens_in_logits_out_decode_back() {
    let vocab_pairs = build_test_vocab();
    let vocab_size = vocab_pairs.len();

    let cfg = SyntheticConfig {
        vocab_size,
        embedding_length: 16,
        block_count: 2,
        feed_forward_length: 32,
        head_count: 4,         // d_head = 4
        head_count_kv: 4,
        rms_eps: 1e-6,
        seed: 42,
        vocab: Some(vocab_pairs),
        merges: None,
        bos_id: Some(1),
        eos_id: Some(2),
        unk_id: Some(0), chat_template: None,
    };

    // Synthesize and parse.
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).expect("parse synthetic gguf");
    println!("Synthetic GGUF: {} bytes, {} tensors", model.data().len(), model.tensors.len());

    // Load the vocabulary.
    let vocab = Vocab::from_gguf(&model).expect("load vocab");
    println!("Vocab: {} tokens, model='{}'", vocab.len(), vocab.model_name);

    // Tokenize the input text.
    let input_text = "hello world";
    let token_ids = vocab.encode_spm(input_text, true);  // with BOS
    println!("Tokenized '{}' to {} tokens: {:?}",
        input_text, token_ids.len(), token_ids);

    // Decode back to verify the tokenizer round-trips.
    let recovered = vocab.decode_spm(&token_ids);
    println!("Decoded back: '{}'", recovered);
    assert_eq!(recovered, input_text,
        "tokenizer must round-trip ASCII text exactly");

    // Build the inference graph for the actual sequence length.
    let seq_len = token_ids.len();
    let llama = build_llama_graph(&model, seq_len).expect("build graph");
    println!("Graph: {} nodes for seq_len={}", llama.graph.nodes.len(), seq_len);

    // Compile and execute.
    let runtime = Runtime::boot();
    let compiled = compile(llama.graph, &runtime.kernels).expect("compile");
    println!("Compiled: {} plan entries", compiled.plan.len());

    // Bind the token IDs as the graph input.
    let id_bytes: Vec<u8> = token_ids.iter()
        .flat_map(|&id| (id as i32).to_le_bytes()).collect();
    let input_tensor = Tensor {
        meta: TensorMeta::new(Dtype::I32, &[seq_len]),
        storage: std::sync::Arc::new(TensorStorage { bytes: id_bytes, mapped: None }),
    };

    let mut inputs = HashMap::new();
    inputs.insert("token_ids".into(), input_tensor);

    let res = execute(&compiled, inputs, ExecutionOptions::default()).expect("execute");
    let logits = res.outputs.get("logits").unwrap();
    println!("Forward pass: shape={:?}, joules={:.6e} J, wall={:?}",
        logits.meta.shape,
        res.trace.joule_accounting.total_joules,
        res.trace.wall_clock);

    // Decode argmax per position back to tokens.
    assert_eq!(logits.meta.shape, vec![seq_len, vocab_size]);
    let l = logits.as_f32_vec();
    let mut argmax_ids = Vec::with_capacity(seq_len);
    for pos in 0..seq_len {
        let row = &l[pos * vocab_size..(pos + 1) * vocab_size];
        let mut best_idx = 0usize;
        let mut best = row[0];
        for i in 1..row.len() {
            if row[i] > best {
                best = row[i];
                best_idx = i;
            }
        }
        argmax_ids.push(best_idx as u32);
    }
    println!("Argmax token IDs per position: {:?}", argmax_ids);

    let argmax_text = vocab.decode_spm(&argmax_ids);
    println!("Decoded argmax tokens: '{}'", argmax_text);

    // Sanity: every argmax ID is in vocabulary range.
    for id in &argmax_ids {
        assert!((*id as usize) < vocab_size);
    }
    // The output is gibberish (random weights) but the pipeline ran end-to-end.
    println!("\nOK: text → tokens → forward pass → argmax → text round-trip works.");
}

/// Just the tokenization round-trip on a non-trivial string with multiple words.
#[test]
fn tokenizer_round_trips_ascii_with_spaces() {
    let cfg = SyntheticConfig {
        vocab_size: build_test_vocab().len(),
        embedding_length: 8,
        block_count: 1,
        feed_forward_length: 16,
        head_count: 1,
        head_count_kv: 1,
        rms_eps: 1e-6,
        seed: 1,
        vocab: Some(build_test_vocab()),
        merges: None,
        bos_id: Some(1),
        eos_id: Some(2),
        unk_id: Some(0), chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    let model = read_gguf(Cursor::new(bytes)).unwrap();
    let vocab = Vocab::from_gguf(&model).unwrap();

    for text in &[
        "a",
        "abc",
        "hello",
        "the quick brown fox",
        "joule is a runtime",
    ] {
        let ids = vocab.encode_spm(text, false);
        let recovered = vocab.decode_spm(&ids);
        assert_eq!(&recovered, text,
            "round-trip failed for '{}': got '{}'", text, recovered);
    }
}
