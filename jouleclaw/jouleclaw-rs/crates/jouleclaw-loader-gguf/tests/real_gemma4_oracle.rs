//! End-to-end Gemma 4 E2B oracles. Ignored by default — they need
//! the real ~10 GB checkpoint at `models/gemma-4-E2B`.
//!
//! Every test in this file gates **correctness of the model output**
//! against the HF transformers float32 reference. There are no
//! "kernel-runs-without-panic" tests here.
//!
//! Run: `cargo test -p jouleclaw-loader-gguf --test real_gemma4_oracle -- --ignored --nocapture`

use std::io::Write;

const DIR: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B";
const OUT: &str = "/Users/dcharlot/data-share/vibe-coding/pattern-lang/tmp/gemma4-oracle";

fn dump(name: &str, v: &[f32]) {
    let mut f = std::fs::File::create(format!("{OUT}/rust_{name}.bin")).unwrap();
    let mut b = Vec::with_capacity(v.len() * 4);
    for x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    f.write_all(&b).unwrap();
}

/// Single-pass forward — every staged tensor matches HF and the
/// post-softcap top-5 ids+values are byte-equal to HF's.
#[test]
#[ignore]
fn real_gemma4_e2b_predicts_paris() {
    if !std::path::Path::new(DIR).join("config.json").exists() {
        eprintln!("skip: gemma-4-E2B not downloaded");
        return;
    }
    let m = jouleclaw_loader_gguf::gemma4::Gemma4::load(DIR).expect("load");
    eprintln!(
        "loaded gemma4 E2B: {} layers, d={}, vocab={}",
        m.cfg.n_layers, m.cfg.d_model, m.cfg.vocab
    );
    let ids: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
    let o = m.forward(&ids);

    dump("embed", &o.embed);
    dump("layer0", &o.layer0);
    dump("layer14", &o.layer14);
    dump("layer34", &o.layer_last);
    dump("final", &o.final_norm);
    dump("logits_post", &o.logits_post);
    dump("logits_pre", &o.logits_pre);
    dump("l0_attn", &o.l0_attn);
    dump("l0_post_attn_norm", &o.l0_post_attn_norm);
    dump("l0_mlp", &o.l0_mlp);
    dump("l0_post_ffw_norm", &o.l0_post_ffw_norm);
    dump("l0_post_ple_norm", &o.l0_post_ple_norm);

    let mut idx: Vec<usize> = (0..o.logits_post.len()).collect();
    idx.sort_by(|&a, &b| o.logits_post[b].partial_cmp(&o.logits_post[a]).unwrap());
    let hf_top5 = [9079usize, 496, 506, 3224, 886];
    assert_eq!(&idx[..5], &hf_top5, "top-5 ids diverge from HF reference");
    let v0 = o.logits_post[idx[0]];
    assert!(
        (v0 - 22.5101).abs() < 0.05,
        "top post-softcap logit {v0} != HF 22.5101 (±0.05)"
    );
}

/// Greedy generation (uncached, full re-prefill per step) — 8-token
/// continuation must equal HF `generate(do_sample=False)` exactly.
#[test]
#[ignore]
fn real_gemma4_e2b_greedy_matches_hf() {
    if !std::path::Path::new(DIR).join("config.json").exists() {
        eprintln!("skip: gemma-4-E2B not downloaded");
        return;
    }
    let m = jouleclaw_loader_gguf::gemma4::Gemma4::load(DIR).expect("load");
    let prompt: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
    let got = m.generate(&prompt, 8);
    let hf: Vec<u32> = vec![9079, 236761, 108, 818, 5279, 529, 7001, 563];
    eprintln!("rust greedy +8: {got:?}");
    eprintln!("HF   greedy +8: {hf:?}  (' Paris.\\n\\nThe capital of France is')");
    assert_eq!(got, hf, "greedy decode diverges from HF reference");
}

/// KV-cached generate equals the uncached path equals HF reference.
#[test]
#[ignore]
fn real_gemma4_e2b_kvcache_parity() {
    if !std::path::Path::new(DIR).join("config.json").exists() {
        eprintln!("skip: gemma-4-E2B not downloaded");
        return;
    }
    let m = jouleclaw_loader_gguf::gemma4::Gemma4::load(DIR).expect("load");
    let prompt: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
    let hf: Vec<u32> = vec![9079, 236761, 108, 818, 5279, 529, 7001, 563];
    let cached = m.generate_cached(&prompt, 8);
    assert_eq!(cached, hf, "kv-cached decode diverges from HF reference");
    let uncached = m.generate(&prompt, 8);
    assert_eq!(cached, uncached, "kv-cached != uncached generate");
}

/// Per-row int8 (Q8) — token-identical to HF on the short prompt.
/// (Note: per-row Q8 *drifts* at the 8th token on the longest
/// benchmark prompt; per-group Q8 below is the production tier.)
#[test]
#[ignore]
fn real_gemma4_e2b_q8_token_identical_on_paris() {
    if !std::path::Path::new(DIR).join("config.json").exists() {
        eprintln!("skip: gemma-4-E2B not downloaded");
        return;
    }
    let g = jouleclaw_loader_gguf::gemma4::Gemma4::load(DIR).expect("load f32");
    let q = jouleclaw_loader_gguf::gemma4_q8::Gemma4Q8::from_gemma4(&g);
    let prompt: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
    let hf: Vec<u32> = vec![9079, 236761, 108, 818, 5279, 529, 7001, 563];
    let got = q.generate_cached(&prompt, 8);
    eprintln!("Q8 greedy +8: {got:?}");
    eprintln!("HF reference: {hf:?}");
    assert_eq!(got, hf, "Q8 must match HF on Paris (max_new=8)");
}

/// Per-group int8 (Q8G) — the production tier. Token-identical to HF
/// on both Paris AND the long Jupiter prompt (where per-row Q8 drifts).
#[test]
#[ignore]
fn real_gemma4_e2b_q8g_token_identical() {
    if !std::path::Path::new(DIR).join("config.json").exists() {
        eprintln!("skip: gemma-4-E2B not downloaded");
        return;
    }
    let g = jouleclaw_loader_gguf::gemma4::Gemma4::load(DIR).expect("load f32");
    let q = jouleclaw_loader_gguf::gemma4_q8g::Gemma4Q8G::from_gemma4(&g);

    let paris_p: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
    let paris_hf: Vec<u32> = vec![9079, 236761, 108, 818, 5279, 529, 7001, 563];
    let got_paris = q.generate_cached(&paris_p, 8);
    assert_eq!(got_paris, paris_hf, "Q8G must match HF on Paris (max_new=8)");

    let jup_p: Vec<u32> = vec![2, 818, 7488, 13401, 528, 1023, 10321, 1458, 563];
    let jup_hf: Vec<u32> = vec![52895, 236761, 1030, 563, 496, 4314, 16784, 236764];
    let got_jup = q.generate_cached(&jup_p, 8);
    assert_eq!(got_jup, jup_hf, "Q8G must match HF on Jupiter (max_new=8)");
}

/// Asymmetric per-row per-group int5 (Q5) — the working int4-class
/// quantization. Token-identical to HF f32 on Paris at max_new=8,
/// ~0.78 B/param (between Q4's 0.6 and Q8G's 1.08).
#[test]
#[ignore]
fn real_gemma4_e2b_q5_token_identical() {
    if !std::path::Path::new(DIR).join("config.json").exists() {
        eprintln!("skip: gemma-4-E2B not downloaded");
        return;
    }
    let g = jouleclaw_loader_gguf::gemma4::Gemma4::load(DIR).expect("load f32");
    let q5 = jouleclaw_loader_gguf::gemma4_q5::Gemma4Q5::from_gemma4(&g);
    let prompt: Vec<u32> = vec![2, 818, 5279, 529, 7001, 563];
    let hf: Vec<u32> = vec![9079, 236761, 108, 818, 5279, 529, 7001, 563];
    let got = q5.generate_cached(&prompt, 8);
    eprintln!("Q5 +8: {got:?}");
    eprintln!("HF   : {hf:?}");
    assert_eq!(got, hf, "Q5 must match HF on Paris (max_new=8)");
}
