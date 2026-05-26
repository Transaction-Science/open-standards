//! Bit-parity test: monolithic vs sequential decode against real
//! Bonsai-1.7B weights.
//!
//! Both paths run the same per-layer arithmetic (each layer uses
//! `build_decode_step_graph_inplace_const_block`). The monolithic
//! path packs all 24 layers into one compiled graph; the sequential
//! path runs 24 separate `execute()` calls with `x` flowing through
//! Rust. Numerical equivalence should be exact — same f32 ops in
//! the same order per layer, no cross-layer reductions.
//!
//! If the logits diverge by more than fp32 noise, the sequential
//! graph composition has a bug that needs fixing before we wire it
//! to int8 mode.
//!
//! `#[ignore]` — needs Bonsai-1.7B on disk.
//!
//! Run:
//!   cargo test --release -p prism --test sequential_parity
//!     -- --ignored --nocapture

use jouleclaw_loader_gguf::kv_cache_inplace::{InPlaceKvCache, KvQuant, ShortConvStateCache};
use jouleclaw_loader_gguf::read_gguf_file;
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::generate::{
    run_inplace_step_cached, run_inplace_step_sequential, DecodeStepCache,
};
use jouleclaw_runtime::Runtime;
use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};

fn model_path() -> String {
    std::env::var("JOULE_BONSAI")
        .unwrap_or_else(|_| "../../models/ternary-bonsai-1.7b-q2_0.gguf".to_string())
}

fn tokens_to_tensor(tokens: &[u32]) -> Tensor {
    let mut bytes = Vec::with_capacity(tokens.len() * 4);
    for &t in tokens {
        bytes.extend_from_slice(&(t as i32).to_le_bytes());
    }
    Tensor {
        meta: TensorMeta::new(Dtype::I32, &[tokens.len()]),
        storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
    }
}

#[test]
#[ignore]
fn sequential_logits_match_monolithic_on_bonsai_prefill() {
    let path = model_path();
    eprintln!("loading {path}");
    let model = read_gguf_file(&path).expect("load model");
    let vocab = Vocab::from_gguf(&model).expect("vocab");

    // A short prompt that exercises prefill + head — enough to see
    // any cross-layer numerical drift.
    let prompt = "The capital of France is";
    let tokens = vocab.encode_bpe_regex(prompt, true);
    eprintln!("prompt tokens ({}): {tokens:?}", tokens.len());
    let new_seq = tokens.len();

    let max_seq = 128usize;

    // ── Monolithic path ──
    let mut cache_mono = InPlaceKvCache::for_model(&model, max_seq).expect("cache mono");
    let mut shortconv_mono = ShortConvStateCache::for_model(&model).expect("shortconv mono");
    let mut step_cache_mono = DecodeStepCache::new();
    let runtime = Runtime::boot();
    let res_mono = run_inplace_step_cached(
        &model, &runtime, &mut cache_mono, &mut shortconv_mono, &mut step_cache_mono,
        tokens_to_tensor(&tokens), new_seq,
    ).expect("mono step");
    eprintln!(
        "monolithic: logits.len()={} joules={:.4} mJ",
        res_mono.logits.len(), res_mono.joules * 1e3);

    // ── Sequential path ──
    let mut cache_seq = InPlaceKvCache::for_model(&model, max_seq).expect("cache seq");
    let mut step_cache_seq = DecodeStepCache::new();
    let res_seq = run_inplace_step_sequential(
        &model, &runtime, &mut cache_seq, &mut step_cache_seq,
        tokens_to_tensor(&tokens), new_seq,
    ).expect("seq step");
    eprintln!(
        "sequential: logits.len()={} joules={:.4} mJ",
        res_seq.logits.len(), res_seq.joules * 1e3);

    // ── Compare logits ──
    assert_eq!(res_mono.logits.len(), res_seq.logits.len(),
        "logits length mismatch");

    let mut max_abs_diff = 0.0_f32;
    let mut argmax_mismatches = 0usize;
    let vocab_size = res_mono.logits.len() / new_seq;
    for pos in 0..new_seq {
        let mono_row = &res_mono.logits[pos * vocab_size..(pos + 1) * vocab_size];
        let seq_row = &res_seq.logits[pos * vocab_size..(pos + 1) * vocab_size];
        for (a, b) in mono_row.iter().zip(seq_row.iter()) {
            let d = (a - b).abs();
            if d > max_abs_diff { max_abs_diff = d; }
        }
        // Argmax should be identical — that's what governs greedy decode.
        let mono_argmax = mono_row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
        let seq_argmax = seq_row.iter().enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap()).unwrap().0;
        if mono_argmax != seq_argmax {
            argmax_mismatches += 1;
        }
    }
    eprintln!(
        "logits diff: max_abs={max_abs_diff:.6}  argmax_mismatches={argmax_mismatches}/{new_seq}");

    // Bit-identical is the goal; allow a tiny fp32 floor for any
    // executor-level ordering effects (kernel scheduling).
    assert!(max_abs_diff < 1e-3,
        "logits diverged: max_abs={max_abs_diff}");
    assert_eq!(argmax_mismatches, 0,
        "argmax differs at {argmax_mismatches} positions — greedy decode would diverge");

    eprintln!("VERDICT: sequential path produces logits bit-identical \
              (to fp32 noise) with the monolithic path on a real \
              Bonsai-1.7B prefill. The substrate is correct; ready \
              to wire KvQuant::Int8 to the sequential executor.");
}
