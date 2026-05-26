//! Gemma 4 (2026, Apache 2.0) HF safetensors → [`GgufModel`] adapter.
//!
//! The [`safetensors`](crate::safetensors) loader reads the *format*
//! faithfully but keeps HF tensor names and carries no architecture
//! metadata. This module is the Gemma-4-specific adapter that turns a
//! raw HF Gemma 4 checkpoint into a `GgufModel` shaped for the rest of
//! the substrate:
//!
//!   1. `config.json` → the `gemma.*` metadata keys consumed by
//!      [`crate::gemma4::Gemma4Config`] (dual-RoPE θ, partial-rotary,
//!      sliding-window, KV-share count, PLE dims, final softcap,
//!      per-layer attention types …).
//!   2. HF `model.language_model.*` tensor names → the GGUF
//!      `blk.{i}.*` / `token_embd` / `per_layer_*` convention
//!      [`crate::gemma4::Gemma4::generate_cached`] expects. The
//!      multimodal vision/audio towers and their projectors have no
//!      text-model analogue and are dropped (they dominate size).
//!
//! Gemma 4 / Gemma 3n use plain `normed * weight` RMSNorm (weight
//! initialised to ones — see `Gemma3nRMSNorm` in transformers); norm
//! weights pass through unfolded. The Gemma 2/3 `(1 + w)` convention
//! is not supported here.

use std::collections::HashMap;
use std::path::Path;

use crate::{GgufModel, GgufValue, ParseError, TensorInfo};

fn cfg_err(msg: impl Into<String>) -> ParseError {
    ParseError::Safetensors(format!("gemma config.json: {}", msg.into()))
}

/// Gemma 4 nests text hyperparameters under `text_config` (multimodal
/// form). Require nesting; flat config is rejected with a clear error.
fn text_root(cfg: &serde_json::Value) -> Result<&serde_json::Value, ParseError> {
    cfg.get("text_config")
        .filter(|tc| tc.get("hidden_size").is_some())
        .ok_or_else(|| cfg_err("missing or invalid `text_config` (Gemma 4 nests text hyperparams)"))
}

fn req_u64(o: &serde_json::Value, key: &str) -> Result<u64, ParseError> {
    o.get(key)
        .and_then(|v| v.as_u64())
        .ok_or_else(|| cfg_err(format!("missing or non-integer `{key}`")))
}

/// Synthesize the `gemma.*` GGUF metadata block from an HF Gemma 4
/// `config.json`.
pub fn gemma_metadata_from_config(
    cfg: &serde_json::Value,
) -> Result<HashMap<String, GgufValue>, ParseError> {
    if let Some(arch0) = cfg
        .get("architectures")
        .and_then(|a| a.as_array())
        .and_then(|a| a.first())
        .and_then(|s| s.as_str())
    {
        let lc = arch0.to_lowercase();
        if !lc.contains("gemma4") && !lc.contains("gemma_4") {
            return Err(cfg_err(format!(
                "architectures[0]=`{arch0}` is not a Gemma 4 variant"
            )));
        }
    }

    let t = text_root(cfg)?;
    let hidden = req_u64(t, "hidden_size")?;
    let heads = req_u64(t, "num_attention_heads")?;
    let kv = t
        .get("num_key_value_heads")
        .and_then(|v| v.as_u64())
        .unwrap_or(heads);
    let head_dim = t
        .get("head_dim")
        .and_then(|v| v.as_u64())
        .unwrap_or_else(|| hidden / heads.max(1));

    let mut m = HashMap::new();
    let mut put = |k: &str, v: GgufValue| {
        m.insert(format!("gemma.{k}"), v);
    };

    put("embedding_length", GgufValue::U64(hidden));
    put("block_count", GgufValue::U64(req_u64(t, "num_hidden_layers")?));
    put("feed_forward_length", GgufValue::U64(req_u64(t, "intermediate_size")?));
    put("attention.head_count", GgufValue::U64(heads));
    put("attention.head_count_kv", GgufValue::U64(kv));
    put("attention.key_length", GgufValue::U64(head_dim));
    put("vocab_size", GgufValue::U64(req_u64(t, "vocab_size")?));
    put(
        "context_length",
        GgufValue::U64(
            t.get("max_position_embeddings")
                .and_then(|v| v.as_u64())
                .unwrap_or(8192),
        ),
    );
    put(
        "attention.layer_norm_rms_epsilon",
        GgufValue::F32(
            t.get("rms_norm_eps").and_then(|v| v.as_f64()).unwrap_or(1e-6) as f32,
        ),
    );

    // Per-attention-type RoPE.
    //   sliding_attention -> { rope_theta, rope_type: default }
    //   full_attention    -> { rope_theta, rope_type: proportional,
    //                          partial_rotary_factor }
    let rp = t
        .get("rope_parameters")
        .ok_or_else(|| cfg_err("missing `rope_parameters`"))?;
    let sliding_theta = rp
        .get("sliding_attention")
        .and_then(|s| s.get("rope_theta"))
        .and_then(|v| v.as_f64())
        .ok_or_else(|| cfg_err("missing rope_parameters.sliding_attention.rope_theta"))?
        as f32;
    put("rope.freq_base", GgufValue::F32(sliding_theta));
    if let Some(full) = rp.get("full_attention") {
        if let Some(gt) = full.get("rope_theta").and_then(|v| v.as_f64()) {
            put("rope.freq_base_global", GgufValue::F32(gt as f32));
        }
        if let Some(prf) = full.get("partial_rotary_factor").and_then(|v| v.as_f64()) {
            put("rope.partial_rotary_factor", GgufValue::F32(prf as f32));
        }
    }
    if let Some(ghd) = t.get("global_head_dim").and_then(|v| v.as_u64()) {
        put("attention.global_head_dim", GgufValue::U64(ghd));
    }

    // Structural extras.
    if let Some(sw) = t.get("sliding_window").and_then(|v| v.as_u64()) {
        put("attention.sliding_window", GgufValue::U64(sw));
    }
    if let Some(n) = t.get("num_kv_shared_layers").and_then(|v| v.as_u64()) {
        put("attention.num_kv_shared_layers", GgufValue::U64(n));
    }
    if let Some(p) = t.get("hidden_size_per_layer_input").and_then(|v| v.as_u64()) {
        put("per_layer_input_length", GgufValue::U64(p));
    }
    if let Some(p) = t.get("vocab_size_per_layer_input").and_then(|v| v.as_u64()) {
        put("per_layer_vocab_size", GgufValue::U64(p));
    }
    if let Some(sc) = t.get("final_logit_softcapping").and_then(|v| v.as_f64()) {
        put("final_logit_softcapping", GgufValue::F32(sc as f32));
    }
    // Per-layer attention type ("sliding_attention" / "full_attention").
    if let Some(lt) = t.get("layer_types").and_then(|v| v.as_array()) {
        let arr: Vec<GgufValue> = lt
            .iter()
            .map(|s| GgufValue::String(s.as_str().unwrap_or("").to_string()))
            .collect();
        m.insert("gemma.attention.layer_types".to_string(), GgufValue::Array(arr));
    }

    m.insert(
        "general.architecture".to_string(),
        GgufValue::String("gemma".to_string()),
    );
    Ok(m)
}

/// Map one HF Gemma 4 tensor name to its GGUF equivalent. Returns
/// `None` for vision/audio tower tensors (dropped — text-only) and
/// for any name that doesn't fit the Gemma 4 language-model layout.
pub fn remap_gemma_tensor_name(hf: &str) -> Option<String> {
    if hf.starts_with("model.vision_tower.")
        || hf.starts_with("model.audio_tower.")
        || hf.starts_with("model.embed_vision.")
        || hf.starts_with("model.embed_audio.")
    {
        return None;
    }
    // Gemma 4 multimodal checkpoints prefix the LM under
    // `model.language_model.`; require that exact infix.
    let core = hf.strip_prefix("model.language_model.")?;

    match core {
        "embed_tokens.weight" => return Some("token_embd.weight".into()),
        "embed_tokens_per_layer.weight" => {
            return Some("per_layer_token_embd.weight".into())
        }
        "per_layer_model_projection.weight" => {
            return Some("per_layer_model_proj.weight".into())
        }
        "per_layer_projection_norm.weight" => {
            return Some("per_layer_proj_norm.weight".into())
        }
        "norm.weight" => return Some("output_norm.weight".into()),
        "lm_head.weight" => return Some("output.weight".into()),
        _ => {}
    }

    let rest = core.strip_prefix("layers.")?;
    let dot = rest.find('.')?;
    let layer: usize = rest[..dot].parse().ok()?;
    let tail = &rest[dot + 1..];

    let mapped = match tail {
        "input_layernorm.weight" => format!("blk.{layer}.attn_norm.weight"),
        "self_attn.q_proj.weight" => format!("blk.{layer}.attn_q.weight"),
        "self_attn.k_proj.weight" => format!("blk.{layer}.attn_k.weight"),
        "self_attn.v_proj.weight" => format!("blk.{layer}.attn_v.weight"),
        "self_attn.o_proj.weight" => format!("blk.{layer}.attn_output.weight"),
        "self_attn.q_norm.weight" => format!("blk.{layer}.attn_q_norm.weight"),
        "self_attn.k_norm.weight" => format!("blk.{layer}.attn_k_norm.weight"),
        "post_attention_layernorm.weight" => {
            format!("blk.{layer}.post_attention_norm.weight")
        }
        "pre_feedforward_layernorm.weight" => format!("blk.{layer}.ffn_norm.weight"),
        "post_feedforward_layernorm.weight" => {
            format!("blk.{layer}.post_ffw_norm.weight")
        }
        "mlp.gate_proj.weight" => format!("blk.{layer}.ffn_gate.weight"),
        "mlp.up_proj.weight" => format!("blk.{layer}.ffn_up.weight"),
        "mlp.down_proj.weight" => format!("blk.{layer}.ffn_down.weight"),
        "per_layer_input_gate.weight" => {
            format!("blk.{layer}.per_layer_gate.weight")
        }
        "per_layer_projection.weight" => {
            format!("blk.{layer}.per_layer_proj.weight")
        }
        "post_per_layer_input_norm.weight" => {
            format!("blk.{layer}.post_per_layer_norm.weight")
        }
        "layer_scalar" => format!("blk.{layer}.layer_scalar"),
        _ => return None,
    };
    Some(mapped)
}

/// Rebuild `model` in place: HF→GGUF tensor names, drop tensors with
/// no GGUF analogue (vision/audio towers), repack the data blob tight.
/// Norm weights pass through unfolded (Gemma 4 / Gemma3nRMSNorm uses
/// plain `normed * weight`).
pub fn adapt_gemma_model(model: &mut GgufModel) -> Result<(), ParseError> {
    let old = std::mem::take(model);
    let mut new_data: Vec<u8> = Vec::with_capacity(old.data().len());
    let mut new_tensors: Vec<TensorInfo> = Vec::with_capacity(old.tensors.len());

    for info in &old.tensors {
        let Some(gguf_name) = remap_gemma_tensor_name(&info.name) else {
            continue;
        };
        let bytes = old.tensor_bytes(info);
        let off = new_data.len() as u64;
        new_data.extend_from_slice(bytes);
        new_tensors.push(TensorInfo {
            name: gguf_name,
            shape: info.shape.clone(),
            dtype: info.dtype,
            offset: off,
        });
    }

    *model = GgufModel {
        version: old.version,
        metadata: old.metadata,
        tensors: new_tensors,
        buf: crate::GgufBuffer::Owned(new_data),
    };
    Ok(())
}

/// Locate `config.json` and the safetensors entrypoint inside an HF
/// snapshot directory, then produce an adapted [`GgufModel`]: format
/// read → HF→GGUF name remap → synthesized `gemma.*` metadata.
pub fn load_gemma_dir<P: AsRef<Path>>(dir: P) -> Result<GgufModel, ParseError> {
    let dir = dir.as_ref();

    let cfg_path = dir.join("config.json");
    let cfg_txt = std::fs::read_to_string(&cfg_path).map_err(crate::io_err)?;
    let cfg: serde_json::Value = serde_json::from_str(&cfg_txt)
        .map_err(|e| cfg_err(format!("parse: {e}")))?;
    let metadata = gemma_metadata_from_config(&cfg)?;

    let idx = dir.join("model.safetensors.index.json");
    let one = dir.join("model.safetensors");
    let mut model = if idx.exists() {
        crate::safetensors::read_safetensors_model(&idx)?
    } else if one.exists() {
        crate::safetensors::read_safetensors_file(&one)?
    } else {
        return Err(cfg_err("no model.safetensors[.index.json] in directory"));
    };

    adapt_gemma_model(&mut model)?;
    model.metadata = metadata;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_non_gemma4_arch() {
        // Gemma 3 / Llama / any non-Gemma-4 architecture is refused.
        let cfg = serde_json::json!({
            "architectures": ["LlamaForCausalLM"],
            "text_config": {
                "hidden_size": 64, "num_hidden_layers": 1,
                "num_attention_heads": 1, "intermediate_size": 64,
                "vocab_size": 10,
                "rope_parameters": {"sliding_attention": {"rope_theta": 1e4}}
            }
        });
        assert!(matches!(
            gemma_metadata_from_config(&cfg),
            Err(ParseError::Safetensors(_))
        ));
        // Bare (non-text_config) Gemma config is also refused.
        let flat = serde_json::json!({
            "architectures": ["Gemma4ForConditionalGeneration"],
            "hidden_size": 64
        });
        assert!(matches!(
            gemma_metadata_from_config(&flat),
            Err(ParseError::Safetensors(_))
        ));
    }

    #[test]
    fn tensor_name_remap_covers_block() {
        assert_eq!(
            remap_gemma_tensor_name("model.language_model.embed_tokens.weight").as_deref(),
            Some("token_embd.weight")
        );
        assert_eq!(
            remap_gemma_tensor_name("model.language_model.norm.weight").as_deref(),
            Some("output_norm.weight")
        );
        assert_eq!(
            remap_gemma_tensor_name("model.language_model.layers.3.self_attn.q_proj.weight")
                .as_deref(),
            Some("blk.3.attn_q.weight")
        );
        assert_eq!(
            remap_gemma_tensor_name("model.language_model.layers.0.mlp.down_proj.weight")
                .as_deref(),
            Some("blk.0.ffn_down.weight")
        );
        assert_eq!(
            remap_gemma_tensor_name(
                "model.language_model.layers.7.post_feedforward_layernorm.weight"
            )
            .as_deref(),
            Some("blk.7.post_ffw_norm.weight")
        );
        assert_eq!(
            remap_gemma_tensor_name(
                "model.language_model.embed_tokens_per_layer.weight"
            )
            .as_deref(),
            Some("per_layer_token_embd.weight")
        );
        assert_eq!(
            remap_gemma_tensor_name("model.language_model.layers.0.layer_scalar")
                .as_deref(),
            Some("blk.0.layer_scalar")
        );
        // Vision/audio towers are dropped.
        assert_eq!(remap_gemma_tensor_name("model.vision_tower.foo.weight"), None);
        assert_eq!(remap_gemma_tensor_name("model.audio_tower.bar"), None);
        assert_eq!(remap_gemma_tensor_name("model.embed_vision.proj"), None);
        // Anything outside `model.language_model.` (and not a tower
        // we explicitly drop) is unmapped.
        assert_eq!(remap_gemma_tensor_name("model.layers.0.foo"), None);
        assert_eq!(remap_gemma_tensor_name("model.rotary_emb.inv_freq"), None);
    }

    /// Real-checkpoint oracle: runs the actual downloaded Gemma 4 E2B
    /// `config.json` + safetensors header through the adapter. Reads
    /// only the 8-byte len + JSON header, never the 10 GB blob.
    #[test]
    fn real_gemma4_e2b_adapter_matches_checkpoint() {
        use std::io::Read;
        let dir = std::path::Path::new(
            "/Users/dcharlot/data-share/vibe-coding/pattern-lang/models/gemma-4-E2B",
        );
        let cfg_path = dir.join("config.json");
        if !cfg_path.exists() {
            eprintln!("skip: gemma-4-E2B not downloaded");
            return;
        }

        let cfg: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&cfg_path).unwrap()).unwrap();
        let m = gemma_metadata_from_config(&cfg).unwrap();
        let g = GgufModel {
            version: 0,
            metadata: m,
            tensors: vec![],
            buf: crate::GgufBuffer::Owned(vec![]),
        };
        assert_eq!(g.metadata_string("general.architecture"), Some("gemma"));
        assert_eq!(g.metadata_u64("gemma.embedding_length"), Some(1536));
        assert_eq!(g.metadata_u64("gemma.block_count"), Some(35));
        assert_eq!(g.metadata_u64("gemma.attention.head_count"), Some(8));
        assert_eq!(g.metadata_u64("gemma.attention.head_count_kv"), Some(1));
        assert_eq!(g.metadata_u64("gemma.attention.key_length"), Some(256));
        assert_eq!(g.metadata_u64("gemma.feed_forward_length"), Some(6144));
        assert_eq!(g.metadata_u64("gemma.vocab_size"), Some(262144));
        assert_eq!(g.metadata_u64("gemma.attention.sliding_window"), Some(512));
        assert_eq!(g.metadata_u64("gemma.attention.num_kv_shared_layers"), Some(20));
        assert_eq!(g.metadata_u64("gemma.per_layer_input_length"), Some(256));
        assert_eq!(g.metadata_u64("gemma.per_layer_vocab_size"), Some(262144));
        assert_eq!(g.metadata_f32("gemma.final_logit_softcapping"), Some(30.0));
        assert_eq!(g.metadata_f32("gemma.rope.freq_base"), Some(10000.0));
        assert_eq!(g.metadata_f32("gemma.rope.freq_base_global"), Some(1_000_000.0));
        assert_eq!(g.metadata_f32("gemma.rope.partial_rotary_factor"), Some(0.25));
        let lt = g
            .metadata
            .get("gemma.attention.layer_types")
            .and_then(|v| v.as_string_array())
            .unwrap();
        assert_eq!(lt.len(), 35);
        assert_eq!(lt[4], "full_attention");
        assert_eq!(lt[0], "sliding_attention");

        let mut f = std::fs::File::open(dir.join("model.safetensors")).unwrap();
        let mut len8 = [0u8; 8];
        f.read_exact(&mut len8).unwrap();
        let hlen = u64::from_le_bytes(len8) as usize;
        let mut hbuf = vec![0u8; hlen];
        f.read_exact(&mut hbuf).unwrap();
        let hdr: serde_json::Value = serde_json::from_slice(&hbuf).unwrap();
        let obj = hdr.as_object().unwrap();

        let mut text_mapped = 0usize;
        let mut dropped_modal = 0usize;
        let mut seen_layers = std::collections::BTreeSet::new();
        let mut have_ple = false;
        for name in obj.keys() {
            if name == "__metadata__" {
                continue;
            }
            match remap_gemma_tensor_name(name) {
                Some(g) => {
                    text_mapped += 1;
                    if let Some(rest) = g.strip_prefix("blk.") {
                        if let Some(d) = rest.find('.') {
                            seen_layers.insert(rest[..d].parse::<usize>().unwrap());
                        }
                    }
                    if g == "per_layer_token_embd.weight" {
                        have_ple = true;
                    }
                }
                None => {
                    if name.starts_with("model.vision_tower.")
                        || name.starts_with("model.audio_tower.")
                        || name.starts_with("model.embed_vision.")
                        || name.starts_with("model.embed_audio.")
                    {
                        dropped_modal += 1;
                    } else {
                        panic!("unmapped text tensor: {name}");
                    }
                }
            }
        }
        assert_eq!(
            seen_layers,
            (0..35).collect::<std::collections::BTreeSet<_>>()
        );
        assert!(have_ple, "PLE token embedding not mapped");
        assert!(text_mapped >= 35 * 16, "too few text tensors: {text_mapped}");
        assert!(dropped_modal > 0, "expected vision/audio tensors dropped");
    }
}
