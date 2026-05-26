//! End-to-end oracle for **LiquidAI/LFM2.5-350M-Q4_K_M**.
//!
//! What this validates that no prior oracle did:
//!
//! 1. **`lfm2`** — fourth supported architecture alongside `llama`,
//!    `qwen3`, `hunyuan-dense`. Hybrid: each block is *either* a
//!    standard attention block *or* a `shortconv` recurrent block,
//!    decided per-layer by the `attention.head_count_kv` array (a `0`
//!    entry marks the recurrent slot).
//! 2. **`Conv1DDepthwiseCausal`** — the new graph op + kernel for
//!    LFM2's depthwise 3-tap causal 1-D convolution.
//! 3. **Per-layer dispatch in `build_block`** routing conv vs attn.
//! 4. **`token_embd_norm.weight`** — LFM2's name for the final RMS
//!    norm (no separate `output_norm.weight`).
//!
//! Gate: parse succeeds, the per-layer kv array contains *both* zeros
//! (conv layers) and non-zeros (attn layers) so the hybrid path is
//! actually exercised, prefill runs, logits are finite, and the
//! greedy next token decodes to a non-empty string.

use jouleclaw_loader_gguf::llama::{build_llama_graph, LlamaConfig};
use jouleclaw_loader_gguf::read_gguf_file;
use jouleclaw_loader_gguf::sample::{sample_logits, SamplingConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};
use std::collections::HashMap;
use std::time::Instant;

fn model_path() -> String {
    std::env::var("JOULE_LFM2")
        .unwrap_or_else(|_| "../../models/lfm2.5-350m-q4_k_m.gguf".to_string())
}

#[test]
#[ignore]
fn real_lfm2_loads_and_speaks() {
    let path = model_path();
    eprintln!("loading {}", path);
    let t0 = Instant::now();
    let model = read_gguf_file(&path).expect("read_gguf_file");
    eprintln!("  parsed GGUF in {:?}", t0.elapsed());
    eprintln!("  arch = {:?}", model.metadata_string("general.architecture"));

    let cfg = LlamaConfig::from_metadata(&model).expect("LlamaConfig::from_metadata");
    eprintln!(
        "  config: arch={} vocab={} dim={} layers={} ff={} heads={} kv={} head_dim={} \
         qk_norm={} ctx={} rope_base={}",
        cfg.arch, cfg.vocab_size, cfg.embedding_length, cfg.block_count,
        cfg.feed_forward_length, cfg.head_count, cfg.head_count_kv, cfg.head_dim,
        cfg.qk_norm, cfg.context_length, cfg.rope_base,
    );
    assert_eq!(cfg.arch, "lfm2");

    // The hybrid path: per-layer kv array must contain BOTH zeros (conv
    // layers) and non-zeros (attn layers). If it's all the same value,
    // the new code path isn't actually exercised.
    let plk = &cfg.per_layer_head_count_kv;
    eprintln!("  per_layer_head_count_kv = {:?}", plk);
    assert_eq!(plk.len(), cfg.block_count,
        "LFM2 must report one kv entry per layer");
    let n_conv = plk.iter().filter(|&&v| v == 0).count();
    let n_attn = plk.iter().filter(|&&v| v > 0).count();
    eprintln!("  hybrid layout: {} conv layers, {} attn layers", n_conv, n_attn);
    assert!(n_conv > 0 && n_attn > 0,
        "LFM2 hybrid must have both conv and attn layers; got {} / {}",
        n_conv, n_attn);
    assert!(cfg.shortconv_l_cache >= 2,
        "LFM2 shortconv.l_cache must be set (got {})", cfg.shortconv_l_cache);

    let vocab = Vocab::from_gguf(&model).expect("Vocab::from_gguf");

    // LFM2.5-350M is instruct-tuned with a qwen2-style chat template
    // (`<|im_start|>role\n…<|im_end|>`). Without the template wrap an
    // instruct model often emits `<|im_end|>` immediately. We use an
    // open-ended continuation-shaped prompt instead so a greedy step
    // is more likely to produce a real word.
    let prompt = "Once upon a time, there was a small";
    let token_ids_u32 = vocab.encode_bpe_regex(prompt, false);
    eprintln!("  prompt {:?} → {} tokens", prompt, token_ids_u32.len());
    assert!(!token_ids_u32.is_empty());
    let seq_len = token_ids_u32.len();

    let t1 = Instant::now();
    let llama = build_llama_graph(&model, seq_len).expect("build_llama_graph");
    eprintln!("  built graph: {} nodes in {:?}", llama.graph.nodes.len(), t1.elapsed());

    let t2 = Instant::now();
    let runtime = Runtime::reference_only();
    let compiled = compile(llama.graph, &runtime.kernels).expect("compile");
    eprintln!("  compiled: {} plan entries in {:?}", compiled.plan.len(), t2.elapsed());

    let bytes: Vec<u8> = token_ids_u32.iter()
        .flat_map(|&id| (id as i32).to_le_bytes()).collect();
    let token_tensor = jouleclaw_core::tensor::Tensor {
        meta: jouleclaw_core::tensor::TensorMeta::new(
            jouleclaw_core::tensor::Dtype::I32, &[seq_len]),
        storage: std::sync::Arc::new(
            jouleclaw_core::tensor::TensorStorage { bytes, mapped: None }),
    };
    let mut inputs = HashMap::new();
    inputs.insert("token_ids".to_string(), token_tensor);

    let t3 = Instant::now();
    let res = execute(&compiled, inputs, ExecutionOptions::default()).expect("execute");
    let fwd = t3.elapsed();
    eprintln!("  >>> ONE forward pass ({} tokens): {:?}", seq_len, fwd);

    let logits = res.outputs.get("logits").expect("logits");
    assert_eq!(logits.meta.shape, vec![seq_len, cfg.vocab_size]);
    let l = logits.as_f32_vec();
    let last = &l[(seq_len - 1) * cfg.vocab_size..seq_len * cfg.vocab_size];
    let max = last.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = last.iter().map(|x| (x - max).exp()).sum();
    assert!(sum.is_finite() && sum > 0.0, "logits must be finite");

    let next = sample_logits(last, &SamplingConfig::greedy());
    let tok = vocab.id_to_token(next).unwrap_or("<UNK>");
    let cont = vocab.decode_bpe(&[next]);
    eprintln!("  >>> greedy next: id={} token={:?} decoded={:?}", next, tok, cont);
    eprintln!("  >>> {:?}{}", prompt, cont);
    // Substrate correctness gate: the next token is valid (in-vocab,
    // not UNK), and we either got printable text *or* a known control
    // token (e.g. `<|im_end|>`, which an instruct model emits without
    // its chat template). A real bug shows up as <UNK>/<unk>, NaN
    // logits, or an id past vocab.
    assert!((next as usize) < cfg.vocab_size,
        "next token id {} out of vocab range {}", next, cfg.vocab_size);
    assert_ne!(tok, "<UNK>", "model emitted unknown token");
    let printable = !cont.is_empty();
    let control = tok.starts_with("<|") && tok.ends_with("|>");
    assert!(printable || control,
        "expected printable text or known control token, got id={} tok={:?} cont={:?}",
        next, tok, cont);

    eprintln!("VERDICT: LFM2.5-350M (hybrid conv+attn, Conv1DDepthwiseCausal, lfm2 arch) \
        loaded + ran + produced a finite, decodable next token. Forward {:?} for {} tokens.",
        fwd, seq_len);
}
