//! End-to-end MRL oracle for **jinaai/jina-embeddings-v5-omni-nano**.
//!
//! Validates everything that fired for the first time on this model:
//!
//! 1. **`eurobert`** — fifth supported architecture (encoder, not
//!    decoder). Different metadata prefix, different output contract.
//! 2. **Bidirectional attention** — `<arch>.attention.causal = False`
//!    drives the non-causal branch of `build_block`, which uses
//!    `g.softmax` instead of `g.softmax_causal`.
//! 3. **`build_llama_encoder_graph`** terminates at `hidden_states`
//!    (no LM head), exposing the post-final-norm activations.
//! 4. **`GgufTextEmbedder`** wraps that with BPE tokenisation, last-
//!    token pooling (LFM-style `pooling_type=3`), optional Matryoshka
//!    truncation, and L2 normalisation. Lands the previously-empty
//!    real-encoder seat in `crates/mrl`.
//!
//! Oracle gate: encode two related sentences + one unrelated, check
//! that the related pair has higher cosine similarity than either to
//! the unrelated one. This is the substrate-level "the embedding is
//! doing something meaningful" smoke test — much stronger than
//! "logits finite". For Matryoshka the same ordering must hold at a
//! truncated dimension too.

use jouleclaw_mrl::GgufTextEmbedder;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_JINA_V5")
        .unwrap_or_else(|_|
            "../../models/jina-v5-omni-nano-retrieval-q4_k_m.gguf".to_string())
}

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
    // Inputs are already L2-normalised by GgufTextEmbedder.
}

#[test]
#[ignore]
fn real_jina_v5_embeds_meaningfully() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    let emb = GgufTextEmbedder::from_gguf(&path).expect("from_gguf");
    eprintln!("  loaded in {:?}", t0.elapsed());
    eprintln!("  arch={} full_dim={} pooling={:?}",
        emb.arch(), emb.full_dim(), emb.pooling());
    assert_eq!(emb.arch(), "eurobert");
    let d = emb.full_dim();

    // Three sentences: the first two are about Paris, the third is about
    // food. A meaningful encoder should make cos(s1,s2) > cos(s1,s3).
    let s1 = "The capital of France is Paris.";
    let s2 = "Paris is the largest city in France.";
    let s3 = "Pizza was invented in Naples, Italy.";

    let t1 = Instant::now();
    let v1 = emb.encode(s1, None).expect("encode s1");
    let v2 = emb.encode(s2, None).expect("encode s2");
    let v3 = emb.encode(s3, None).expect("encode s3");
    eprintln!("  encoded 3 sentences in {:?} (full d={})", t1.elapsed(), d);
    assert_eq!(v1.len(), d);
    assert_eq!(v2.len(), d);
    assert_eq!(v3.len(), d);

    // Sanity: every vector is finite + approximately unit-norm.
    for (lbl, v) in [("s1", &v1), ("s2", &v2), ("s3", &v3)] {
        let n2: f32 = v.iter().map(|x| x * x).sum();
        assert!(v.iter().all(|x| x.is_finite()), "{} not finite", lbl);
        assert!((n2 - 1.0).abs() < 1e-3, "{} not unit-norm: n²={}", lbl, n2);
    }

    let sim_12 = cosine(&v1, &v2);
    let sim_13 = cosine(&v1, &v3);
    let sim_23 = cosine(&v2, &v3);
    eprintln!("  cos(Paris1, Paris2)  = {:.4}", sim_12);
    eprintln!("  cos(Paris1, pizza)   = {:.4}", sim_13);
    eprintln!("  cos(Paris2, pizza)   = {:.4}", sim_23);
    assert!(sim_12 > sim_13,
        "ORACLE FAILED: related Paris pair must out-similar the food sentence \
         (got {:.4} ≤ {:.4})", sim_12, sim_13);
    assert!(sim_12 > sim_23,
        "ORACLE FAILED: related Paris pair must out-similar Paris-vs-food \
         (got {:.4} ≤ {:.4})", sim_12, sim_23);

    // Matryoshka: truncate to 64 dims. Same ordering must hold (any
    // prefix is a valid lower-fidelity embedding).
    let v1t = emb.encode(s1, Some(64)).unwrap();
    let v2t = emb.encode(s2, Some(64)).unwrap();
    let v3t = emb.encode(s3, Some(64)).unwrap();
    let sim_12_t = cosine(&v1t, &v2t);
    let sim_13_t = cosine(&v1t, &v3t);
    eprintln!("  matryoshka d=64: cos(P1,P2)={:.4}  cos(P1,pizza)={:.4}",
        sim_12_t, sim_13_t);
    assert!(sim_12_t > sim_13_t,
        "MATRYOSHKA ORACLE FAILED at d=64: {:.4} ≤ {:.4}", sim_12_t, sim_13_t);

    eprintln!("VERDICT: jina-embeddings-v5-omni-nano (eurobert, bidirectional, \
        last-pool, Q4_K_M) embeds meaningfully at full dim {} and at d=64.", d);
}
