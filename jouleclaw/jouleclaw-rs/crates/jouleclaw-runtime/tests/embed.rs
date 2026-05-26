//! Tests for the embedding API.

use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::{embed, cosine_similarity, EmbedConfig, Pooling, TokenizerKind};
use std::io::Cursor;

fn build_test_vocab() -> Vec<(String, f32)> {
    let mut v = vec![
        ("<unk>".to_string(), 0.0),
        ("<s>".to_string(), 0.0),
        ("</s>".to_string(), 0.0),
        ("\u{2581}".to_string(), 0.0),
    ];
    for c in 'a'..='z' { v.push((c.to_string(), -1.0)); }
    for c in 'a'..='z' { v.push((format!("\u{2581}{}", c), -0.5)); }
    for b in 0..256u32 { v.push((format!("<0x{:02X}>", b), -10.0)); }
    v
}

fn build_test_model(head_count: usize, head_count_kv: usize) -> jouleclaw_loader_gguf::GgufModel {
    let vocab = build_test_vocab();
    let cfg = SyntheticConfig {
        vocab_size: vocab.len(),
        embedding_length: 16,
        block_count: 2,
        feed_forward_length: 32,
        head_count, head_count_kv,
        rms_eps: 1e-6,
        seed: 200,
        vocab: Some(vocab),
        merges: None,
        bos_id: Some(1), eos_id: Some(2), unk_id: Some(0), chat_template: None,
    };
    let bytes = synthesize_llama_gguf(&cfg);
    read_gguf(Cursor::new(bytes)).unwrap()
}

/// Mean pooling produces a [d_model] vector that is L2-normalized.
#[test]
fn mean_pooling_produces_d_model_vector() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = EmbedConfig::default();
    let result = embed(&model, &vocab, "hello world", &cfg).unwrap();
    assert_eq!(result.vector.len(), 16);
    assert_eq!(result.d_model, 16);

    // L2 norm should be 1.0 (within FP tolerance).
    let norm_sq: f32 = result.vector.iter().map(|x| x * x).sum();
    assert!((norm_sq - 1.0).abs() < 1e-5,
        "L2-normalized vector should have norm 1; got norm_sq = {}", norm_sq);
}

/// Last-token pooling produces a different vector than mean pooling.
#[test]
fn last_token_pooling_differs_from_mean() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();

    let mut cfg = EmbedConfig::default();
    let mean = embed(&model, &vocab, "abc def ghi", &cfg).unwrap();
    cfg.pooling = Pooling::LastToken;
    let last = embed(&model, &vocab, "abc def ghi", &cfg).unwrap();

    assert_eq!(mean.vector.len(), last.vector.len());
    let cos = cosine_similarity(&mean.vector, &last.vector);
    assert!(cos < 0.999,
        "mean and last-token pooling should produce different vectors, \
         got cos similarity = {}", cos);
}

/// No-pooling returns the raw per-token hidden states.
#[test]
fn no_pooling_returns_per_token_states() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = EmbedConfig {
        pooling: Pooling::None,
        l2_normalize: false,
        ..EmbedConfig::default()
    };
    let result = embed(&model, &vocab, "abc", &cfg).unwrap();
    // Vector should be seq * d_model elements.
    assert_eq!(result.vector.len(), result.token_count * result.d_model);
    assert!(result.token_count > 0);
}

/// Embeddings are deterministic given identical input.
#[test]
fn embedding_is_deterministic() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = EmbedConfig::default();
    let e1 = embed(&model, &vocab, "deterministic test", &cfg).unwrap();
    let e2 = embed(&model, &vocab, "deterministic test", &cfg).unwrap();
    let e3 = embed(&model, &vocab, "deterministic test", &cfg).unwrap();
    assert_eq!(e1.vector, e2.vector);
    assert_eq!(e2.vector, e3.vector);
}

/// Self-similarity is exactly 1.0 for L2-normalized vectors.
#[test]
fn self_similarity_is_one() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = EmbedConfig::default();
    let result = embed(&model, &vocab, "test", &cfg).unwrap();
    let cos = cosine_similarity(&result.vector, &result.vector);
    assert!((cos - 1.0).abs() < 1e-5,
        "self-similarity should be 1.0, got {}", cos);
}

/// Different texts produce different embeddings; cosine similarity should
/// be in [-1, 1] and not stuck at 1.
#[test]
fn different_texts_produce_different_embeddings() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = EmbedConfig::default();
    let e1 = embed(&model, &vocab, "abc", &cfg).unwrap();
    let e2 = embed(&model, &vocab, "xyz", &cfg).unwrap();
    let cos = cosine_similarity(&e1.vector, &e2.vector);
    assert!(cos >= -1.0 && cos <= 1.0);
    assert!(cos < 0.999,
        "different texts should produce distinguishable embeddings, \
         got cos = {}", cos);
    // Embedding values themselves should differ.
    assert_ne!(e1.vector, e2.vector);
}

/// Embeddings work with GQA models.
#[test]
fn embed_works_with_gqa() {
    let model = build_test_model(4, 2);  // GQA
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = EmbedConfig::default();
    let result = embed(&model, &vocab, "hello", &cfg).unwrap();
    assert_eq!(result.vector.len(), 16);
    let norm_sq: f32 = result.vector.iter().map(|x| x * x).sum();
    assert!((norm_sq - 1.0).abs() < 1e-5);
}

/// L2 normalization can be disabled.
#[test]
fn l2_normalize_can_be_disabled() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let mut cfg = EmbedConfig::default();
    cfg.l2_normalize = false;
    let result = embed(&model, &vocab, "abc", &cfg).unwrap();
    let norm_sq: f32 = result.vector.iter().map(|x| x * x).sum();
    // Unnormalized norm should generally NOT be 1.0.
    assert!((norm_sq - 1.0).abs() > 1e-5 || norm_sq < 1e-10,
        "unnormalized embedding norm shouldn't accidentally be 1.0; got norm_sq = {}",
        norm_sq);
}

/// Empty prompt with add_bos handled gracefully (either errors or returns
/// something sensible; we just don't crash).
#[test]
fn empty_prompt_handled_gracefully() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = EmbedConfig {
        add_bos: false,
        ..EmbedConfig::default()
    };
    // Don't crash; either error or return is acceptable.
    let _ = embed(&model, &vocab, "", &cfg);
}

/// Cosine similarity properties: symmetric, in [-1, 1], 1.0 on identical.
#[test]
fn cosine_similarity_properties() {
    let a = vec![0.6f32, 0.8];
    let b = vec![0.8f32, 0.6];
    let c = vec![-0.6f32, -0.8];  // anti-parallel to a

    assert!((cosine_similarity(&a, &a) - 1.0).abs() < 1e-6);
    assert!((cosine_similarity(&a, &b) - 0.96).abs() < 1e-3);
    assert!((cosine_similarity(&a, &c) - (-1.0)).abs() < 1e-6);

    // Symmetric.
    assert_eq!(cosine_similarity(&a, &b), cosine_similarity(&b, &a));
}

/// Embed encoder graph is cheaper than full generate (no lm_head matmul).
/// We verify this indirectly: the encoder graph runs successfully and
/// returns the post-norm hidden states, demonstrating that the lm_head
/// is skipped.
#[test]
fn encoder_graph_skips_lm_head() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();

    let cfg = EmbedConfig {
        pooling: Pooling::None,
        l2_normalize: false,
        ..EmbedConfig::default()
    };
    let result = embed(&model, &vocab, "test prompt here", &cfg).unwrap();

    // The hidden state size is d_model, not vocab_size. If the lm_head
    // had been applied, the size would be vocab_size = 312, not d_model = 16.
    assert_eq!(result.d_model, 16);
    assert_eq!(result.vector.len(), result.token_count * 16);
}

/// Demo: side-by-side similarity comparison.
#[test]
fn embedding_similarity_demo() {
    let model = build_test_model(4, 4);
    let vocab = Vocab::from_gguf(&model).unwrap();
    let cfg = EmbedConfig::default();

    let texts = [
        "the cat sat on the mat",
        "a cat is sitting on a mat",
        "quantum mechanics describes nature",
    ];
    let embeddings: Vec<_> = texts.iter().map(|t| {
        embed(&model, &vocab, t, &cfg).unwrap().vector
    }).collect();

    println!("\n=== Embedding similarity demo ===");
    println!("Model: 2-layer synthetic Llama, d_model=16, random weights.");
    println!("(Real models trained on text would show meaningful similarity ");
    println!(" patterns; with random weights the similarities are arbitrary.)\n");
    println!("Pooling: mean, L2-normalized");
    println!("Texts:");
    for (i, t) in texts.iter().enumerate() {
        println!("  [{}] {:?}", i, t);
    }
    println!("\nPairwise cosine similarities:");
    for i in 0..texts.len() {
        for j in 0..texts.len() {
            let cos = cosine_similarity(&embeddings[i], &embeddings[j]);
            print!("  cos([{}], [{}]) = {:+.4}", i, j, cos);
        }
        println!();
    }
    println!("\nEmbedding dim: {}  (vs vocab_size: {} — encoder graph skips lm_head)",
        embeddings[0].len(), vocab.len());
}
