//! Synthetic GGUF model factory for tests.
//!
//! Generates a tiny but architecturally valid Llama-style GGUF buffer with
//! deterministic random weights. Used by integration tests and as a
//! reference for the end-to-end loader path.

use crate::{GgmlType, GgufValue};

const GGUF_MAGIC: u32 = 0x46554747;

/// Configuration for a synthetic Llama-style model.
#[derive(Debug, Clone)]
pub struct SyntheticConfig {
    pub vocab_size: usize,
    pub embedding_length: usize,
    pub block_count: usize,
    pub feed_forward_length: usize,
    pub head_count: usize,
    /// KV head count. When equal to `head_count` this is standard MHA;
    /// when smaller (and `head_count % head_count_kv == 0`) this is GQA.
    /// Default: equal to `head_count` (MHA).
    pub head_count_kv: usize,
    pub rms_eps: f32,
    pub seed: u64,
    /// Optional embedded vocabulary. If `None`, no `tokenizer.ggml.*` keys
    /// are written. If `Some`, must have `vocab_size` entries.
    pub vocab: Option<Vec<(String, f32)>>,
    /// Optional BPE merge rules for `tokenizer.ggml.merges`. Each entry is
    /// a `(left, right)` pair; written as `"left right"` strings in GGUF.
    /// Index = rank (lower = higher priority).
    pub merges: Option<Vec<(String, String)>>,
    /// Optional BOS / EOS / UNK token IDs. Only meaningful when `vocab` is set.
    pub bos_id: Option<u32>,
    pub eos_id: Option<u32>,
    pub unk_id: Option<u32>,
    /// Optional Jinja-source chat_template string. Written to GGUF as
    /// `tokenizer.chat_template`. Used by `ChatTemplate::detect_from_model`.
    pub chat_template: Option<String>,
}

impl Default for SyntheticConfig {
    fn default() -> Self {
        Self {
            vocab_size: 32,
            embedding_length: 16,
            block_count: 2,
            feed_forward_length: 32,
            head_count: 1,
            head_count_kv: 1,
            rms_eps: 1e-6,
            seed: 42,
            vocab: None,
            merges: None,
            bos_id: None,
            eos_id: None,
            unk_id: None,
            chat_template: None,
        }
    }
}

/// Generate a complete GGUF byte buffer for a synthetic Llama model.
pub fn synthesize_llama_gguf(config: &SyntheticConfig) -> Vec<u8> {
    let mut rng = SeededRng::new(config.seed);

    // Build the metadata KV list.
    let mut metadata: Vec<(String, GgufValue)> = vec![
        ("general.architecture".into(), GgufValue::String("llama".into())),
        ("general.name".into(), GgufValue::String("synthetic".into())),
        ("general.alignment".into(), GgufValue::U32(32)),
        ("llama.vocab_size".into(), GgufValue::U64(config.vocab_size as u64)),
        ("llama.embedding_length".into(), GgufValue::U64(config.embedding_length as u64)),
        ("llama.block_count".into(), GgufValue::U64(config.block_count as u64)),
        ("llama.feed_forward_length".into(),
            GgufValue::U64(config.feed_forward_length as u64)),
        ("llama.attention.head_count".into(),
            GgufValue::U64(config.head_count as u64)),
        ("llama.attention.head_count_kv".into(),
            GgufValue::U64(config.head_count_kv as u64)),
        ("llama.attention.layer_norm_rms_epsilon".into(),
            GgufValue::F32(config.rms_eps)),
        ("llama.context_length".into(), GgufValue::U64(2048)),
    ];

    if let Some(vocab) = &config.vocab {
        assert_eq!(vocab.len(), config.vocab_size,
            "vocab length must equal vocab_size");
        let tokens: Vec<GgufValue> = vocab.iter()
            .map(|(t, _)| GgufValue::String(t.clone())).collect();
        let scores: Vec<GgufValue> = vocab.iter()
            .map(|(_, s)| GgufValue::F32(*s)).collect();
        metadata.push(("tokenizer.ggml.model".into(),
            GgufValue::String("llama".into())));
        metadata.push(("tokenizer.ggml.tokens".into(), GgufValue::Array(tokens)));
        metadata.push(("tokenizer.ggml.scores".into(), GgufValue::Array(scores)));
        if let Some(id) = config.bos_id {
            metadata.push(("tokenizer.ggml.bos_token_id".into(), GgufValue::U32(id)));
        }
        if let Some(id) = config.eos_id {
            metadata.push(("tokenizer.ggml.eos_token_id".into(), GgufValue::U32(id)));
        }
        if let Some(id) = config.unk_id {
            metadata.push(("tokenizer.ggml.unknown_token_id".into(), GgufValue::U32(id)));
        }
    }

    if let Some(merges) = &config.merges {
        let merge_strings: Vec<GgufValue> = merges.iter()
            .map(|(l, r)| GgufValue::String(format!("{} {}", l, r)))
            .collect();
        metadata.push(("tokenizer.ggml.merges".into(), GgufValue::Array(merge_strings)));
    }

    if let Some(tmpl) = &config.chat_template {
        metadata.push(("tokenizer.chat_template".into(), GgufValue::String(tmpl.clone())));
    }

    // Tensors. GGUF stores weights as [out_features, in_features].
    let d = config.embedding_length;
    let dff = config.feed_forward_length;
    let v = config.vocab_size;
    // Under GQA: K and V have fewer heads.
    let d_head = if config.head_count > 0 { d / config.head_count } else { d };
    let d_kv = config.head_count_kv * d_head;

    let mut tensors: Vec<(String, Vec<u64>, GgmlType, Vec<u8>)> = Vec::new();

    // Token embedding: [vocab, d_model].
    tensors.push(("token_embd.weight".into(), vec![v as u64, d as u64],
        GgmlType::F32, rng.f32_bytes(v * d)));

    // Per-block weights.
    for i in 0..config.block_count {
        // attn_norm.weight: [d]
        tensors.push((format!("blk.{}.attn_norm.weight", i),
            vec![d as u64], GgmlType::F32, rng.f32_bytes(d)));
        // Q projection: [d, d] (n_heads * d_head columns out)
        tensors.push((format!("blk.{}.attn_q.weight", i),
            vec![d as u64, d as u64], GgmlType::F32, rng.f32_bytes(d * d)));
        // K, V projections: [d_kv, d] (n_heads_kv * d_head columns out)
        for name in ["attn_k", "attn_v"] {
            tensors.push((format!("blk.{}.{}.weight", i, name),
                vec![d_kv as u64, d as u64], GgmlType::F32,
                rng.f32_bytes(d_kv * d)));
        }
        // Output projection: [d, d]
        tensors.push((format!("blk.{}.attn_output.weight", i),
            vec![d as u64, d as u64], GgmlType::F32, rng.f32_bytes(d * d)));
        // ffn_norm.weight: [d]
        tensors.push((format!("blk.{}.ffn_norm.weight", i),
            vec![d as u64], GgmlType::F32, rng.f32_bytes(d)));
        // ffn_gate.weight, ffn_up.weight: [dff, d]
        for name in ["ffn_gate", "ffn_up"] {
            tensors.push((format!("blk.{}.{}.weight", i, name),
                vec![dff as u64, d as u64], GgmlType::F32,
                rng.f32_bytes(dff * d)));
        }
        // ffn_down.weight: [d, dff]
        tensors.push((format!("blk.{}.ffn_down.weight", i),
            vec![d as u64, dff as u64], GgmlType::F32,
            rng.f32_bytes(d * dff)));
    }

    // Final norm + lm_head.
    tensors.push(("output_norm.weight".into(),
        vec![d as u64], GgmlType::F32, rng.f32_bytes(d)));
    tensors.push(("output.weight".into(),
        vec![v as u64, d as u64], GgmlType::F32, rng.f32_bytes(v * d)));

    encode_gguf(&metadata, &tensors)
}

fn encode_gguf(
    metadata: &[(String, GgufValue)],
    tensors: &[(String, Vec<u64>, GgmlType, Vec<u8>)],
) -> Vec<u8> {
    let mut out = Vec::new();

    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (k, v) in metadata {
        write_string(&mut out, k);
        write_value(&mut out, v);
    }

    let mut running: u64 = 0;
    let mut offsets = Vec::with_capacity(tensors.len());
    for (_, _, _, data) in tensors {
        offsets.push(running);
        running += data.len() as u64;
    }

    for ((name, shape, dtype, _), offset) in tensors.iter().zip(offsets.iter()) {
        write_string(&mut out, name);
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        // Emit in GGML `ne` order (fastest axis first), matching real
        // GGUF. The logical shape used everywhere else is the reverse;
        // `tensor_from_gguf` reverses it back. Keeps this fixture
        // byte-format-identical to llama.cpp output.
        for d in shape.iter().rev() { out.extend_from_slice(&d.to_le_bytes()); }
        out.extend_from_slice(&(*dtype as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    // Align to 32.
    while out.len() % 32 != 0 { out.push(0); }

    for (_, _, _, data) in tensors {
        out.extend_from_slice(data);
    }
    out
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, v: &GgufValue) {
    match v {
        GgufValue::U32(x) => {
            out.extend_from_slice(&4u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        GgufValue::F32(x) => {
            out.extend_from_slice(&6u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        GgufValue::String(s) => {
            out.extend_from_slice(&8u32.to_le_bytes());
            write_string(out, s);
        }
        GgufValue::U64(x) => {
            out.extend_from_slice(&10u32.to_le_bytes());
            out.extend_from_slice(&x.to_le_bytes());
        }
        GgufValue::Array(arr) => {
            // Array type tag (9), then element type tag, then length, then values.
            out.extend_from_slice(&9u32.to_le_bytes());
            // Determine element type from first item; assume homogeneous.
            // For empty arrays default to String type (8) — a common case
            // for empty merges or empty token lists.
            let elem_type: u32 = match arr.first() {
                Some(GgufValue::String(_)) => 8,
                Some(GgufValue::F32(_)) => 6,
                Some(GgufValue::U32(_)) => 4,
                Some(GgufValue::U64(_)) => 10,
                None => 8,  // empty array; element type is conventional
                _ => panic!("synthetic factory: unsupported array element type"),
            };
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(arr.len() as u64).to_le_bytes());
            for elem in arr {
                match elem {
                    GgufValue::String(s) => write_string(out, s),
                    GgufValue::F32(x) => out.extend_from_slice(&x.to_le_bytes()),
                    GgufValue::U32(x) => out.extend_from_slice(&x.to_le_bytes()),
                    GgufValue::U64(x) => out.extend_from_slice(&x.to_le_bytes()),
                    _ => panic!("array element type mismatch in synthetic factory"),
                }
            }
        }
        _ => unimplemented!("synthetic factory writes only the value types we use"),
    }
}

/// Linear-congruential generator for deterministic random weights.
/// Identical to the runtime example's PRNG so tests reproduce.
struct SeededRng { state: u64 }

impl SeededRng {
    fn new(seed: u64) -> Self { Self { state: seed } }

    fn next_f32(&mut self) -> f32 {
        self.state = self.state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        let bits = (self.state >> 40) as u32;
        let v = (bits as f32) * (1.0 / (1u32 << 24) as f32);
        // Center on 0; scale small to avoid fp blowup through deep nets.
        (v - 0.5) * 0.1
    }

    fn f32_bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * 4);
        for _ in 0..n {
            out.extend_from_slice(&self.next_f32().to_le_bytes());
        }
        out
    }
}
