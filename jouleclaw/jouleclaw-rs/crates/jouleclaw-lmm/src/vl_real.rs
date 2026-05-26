//! `LfmVl` — real LFM2.5-VL inference: image bytes + text prompt →
//! next-token logits over the combined multimodal sequence.
//!
//! Ties together:
//!   * [`crate::vision::VisionTower`]  — SigLIP ViT + LFM2 projector
//!     (mmproj GGUF) → `[n_img_tok, d_model]` image tokens
//!   * the LFM2 text backbone via
//!     `jouleclaw_loader_gguf::llama::build_llama_graph_from_embeds`
//!     (the embedding-input entry added for multimodal)
//!
//! The embedding stream is `[ text_embeds … ; image_tokens ; text_embeds … ]`
//! — image tokens are spliced at the placeholder position and run
//! through the same LFM2 stack as text. Single forward (prefill) for
//! the oracle; full autoregressive caption decode is a follow-on.

use std::collections::HashMap;
use std::path::Path;

use jouleclaw_core::tensor::{Dtype, Tensor, TensorMeta, TensorStorage};
use jouleclaw_loader_gguf::llama::{build_llama_graph_from_embeds, LlamaConfig};
use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::{read_gguf_file, tensor_from_gguf, GgufModel};
use jouleclaw_runtime::{compile, execute, ExecutionOptions, Runtime};

use crate::vision::{VisionError, VisionTower};

#[derive(Debug)]
pub enum VlError {
    Vision(VisionError),
    Parse(String),
    Config(String),
    Vocab,
    Run(String),
    DimMismatch { vision: usize, text: usize },
}

impl std::fmt::Display for VlError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Vision(e) => write!(f, "vision: {}", e),
            Self::Parse(s) => write!(f, "parse: {}", s),
            Self::Config(s) => write!(f, "config: {}", s),
            Self::Vocab => write!(f, "vocab load failed"),
            Self::Run(s) => write!(f, "run: {}", s),
            Self::DimMismatch { vision, text } => write!(f,
                "projector out dim {} != text embedding dim {}", vision, text),
        }
    }
}
impl std::error::Error for VlError {}

pub struct LfmVl {
    text_model: GgufModel,
    vocab: Vocab,
    config: LlamaConfig,
    vision: VisionTower,
    /// Dequantised text token-embedding table `[vocab, d]`, row-major.
    token_embd: Vec<f32>,
    d_model: usize,
}

impl LfmVl {
    pub fn from_gguf<P: AsRef<Path>, Q: AsRef<Path>>(
        text_path: P,
        mmproj_path: Q,
    ) -> Result<Self, VlError> {
        let text_model = read_gguf_file(text_path.as_ref())
            .map_err(|e| VlError::Parse(format!("{:?}", e)))?;
        let vocab = Vocab::from_gguf(&text_model).map_err(|_| VlError::Vocab)?;
        let config = LlamaConfig::from_metadata(&text_model)
            .map_err(|e| VlError::Config(format!("{}", e)))?;
        let vision = VisionTower::from_gguf(mmproj_path).map_err(VlError::Vision)?;

        let te_info = text_model.tensor_by_name("token_embd.weight")
            .ok_or_else(|| VlError::Parse("no token_embd.weight".into()))?;
        let te = tensor_from_gguf(&text_model, te_info)
            .map_err(|e| VlError::Parse(format!("token_embd: {:?}", e)))?;
        let token_embd = te.as_f32_vec();
        let d_model = config.embedding_length;

        Ok(Self { text_model, vocab, config, vision, token_embd, d_model })
    }

    pub fn arch(&self) -> &str { &self.config.arch }
    pub fn d_model(&self) -> usize { self.d_model }

    /// One text-token's embedding row from the (dequantised) table.
    fn embed_row(&self, token_id: u32) -> &[f32] {
        let i = token_id as usize * self.d_model;
        &self.token_embd[i..i + self.d_model]
    }

    /// Run image + prompt through one forward pass. Returns
    /// `(next_token_id, decoded_text, n_seq, n_img_tokens, joules)`.
    /// `prompt` text is placed *after* the image tokens (the common
    /// "<image> question" layout); a leading text segment can be added
    /// later once we wire the exact LFM2-VL chat template.
    pub fn forward_once(
        &self,
        image_bytes: &[u8],
        prompt: &str,
    ) -> Result<(u32, String, usize, usize, f64), VlError> {
        // 1. Vision tower → image tokens [n_img, d_proj].
        let rgb = self.vision.preprocess(image_bytes).map_err(VlError::Vision)?;
        let img_tokens = self.vision.encode(&rgb).map_err(VlError::Vision)?;
        let n_img = img_tokens.len();
        let proj_d = img_tokens.first().map(|v| v.len()).unwrap_or(0);
        if proj_d != self.d_model {
            return Err(VlError::DimMismatch { vision: proj_d, text: self.d_model });
        }

        // 2. Tokenise prompt (LFM2 uses gpt2 BPE; no BOS for this probe).
        let txt_tokens = self.vocab.encode_bpe_regex(prompt, false);
        let n_txt = txt_tokens.len();
        let seq = n_img + n_txt;
        if seq == 0 {
            return Err(VlError::Run("empty multimodal sequence".into()));
        }

        // 3. Build the [image ; text] embedding stream.
        let d = self.d_model;
        let mut embeds = Vec::with_capacity(seq * d);
        for v in &img_tokens {
            embeds.extend_from_slice(v);
        }
        for &t in &txt_tokens {
            embeds.extend_from_slice(self.embed_row(t));
        }

        // 4. LFM2 backbone over the precomputed embeddings.
        let graph = build_llama_graph_from_embeds(&self.text_model, seq)
            .map_err(|e| VlError::Config(format!("{}", e)))?
            .graph;
        let runtime = Runtime::reference_only();
        let compiled = compile(graph, &runtime.kernels)
            .map_err(|e| VlError::Run(format!("compile: {:?}", e)))?;

        let bytes: Vec<u8> = embeds.iter().flat_map(|v| v.to_le_bytes()).collect();
        let input = Tensor {
            meta: TensorMeta::new(Dtype::F32, &[seq, d]),
            storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
        };
        let mut inputs = HashMap::new();
        inputs.insert("input_embeds".to_string(), input);

        let res = execute(&compiled, inputs, ExecutionOptions::default())
            .map_err(|e| VlError::Run(format!("execute: {:?}", e)))?;
        let joules = res.trace.joule_accounting.total_joules;
        let logits = res.outputs.get("logits")
            .ok_or_else(|| VlError::Run("no logits".into()))?;
        let v = self.vocab.len();
        let l = logits.as_f32_vec();
        let last = &l[(seq - 1) * v..seq * v];

        // Greedy argmax over the last position.
        let mut best = 0usize;
        let mut best_v = f32::NEG_INFINITY;
        for (i, &x) in last.iter().enumerate() {
            if x.is_finite() && x > best_v { best_v = x; best = i; }
        }
        let tok = best as u32;
        let decoded = self.vocab.decode_bpe(&[tok]);
        Ok((tok, decoded, seq, n_img, joules))
    }

    /// Generate up to `max_new_tokens` text tokens for the given image +
    /// prompt. Returns the decoded caption + cumulative joules.
    ///
    /// Naive no-cache loop: every step re-runs the full prefill over
    /// `[image_tokens ; prompt_tokens ; generated_so_far]` and reads the
    /// last logit row. Correct, deterministic, and slow — for a 64-image-
    /// token + short-prompt setup it's `O(max_new × (n_img+text)^2)`.
    /// A cache-aware multimodal prefill is a separate optimisation
    /// milestone; this is the correctness floor.
    ///
    /// Stops early on EOS. Greedy argmax — no sampling temperature here
    /// (the caller can run a smarter sampler outside if it wants).
    pub fn generate(
        &self,
        image_bytes: &[u8],
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<(String, f64), VlError> {
        // 1. Vision tower → image tokens [n_img, d_proj].
        let rgb = self.vision.preprocess(image_bytes).map_err(VlError::Vision)?;
        let img_tokens = self.vision.encode(&rgb).map_err(VlError::Vision)?;
        let proj_d = img_tokens.first().map(|v| v.len()).unwrap_or(0);
        if proj_d != self.d_model {
            return Err(VlError::DimMismatch { vision: proj_d, text: self.d_model });
        }
        let n_img = img_tokens.len();

        // 2. Tokenise the user prompt.
        let prompt_tokens = self.vocab.encode_bpe_regex(prompt, false);
        let n_prompt = prompt_tokens.len();

        let d = self.d_model;
        let v = self.vocab.len();
        let runtime = Runtime::reference_only();
        let eos = self.vocab.eos_id;
        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut total_joules: f64 = 0.0;

        for _ in 0..max_new_tokens {
            let seq = n_img + n_prompt + generated.len();

            // Build the embedding stream: image first, then prompt
            // tokens, then any tokens generated so far.
            let mut embeds = Vec::with_capacity(seq * d);
            for v in &img_tokens {
                embeds.extend_from_slice(v);
            }
            for &t in &prompt_tokens {
                embeds.extend_from_slice(self.embed_row(t));
            }
            for &t in &generated {
                embeds.extend_from_slice(self.embed_row(t));
            }

            let graph = build_llama_graph_from_embeds(&self.text_model, seq)
                .map_err(|e| VlError::Config(format!("{}", e)))?
                .graph;
            let compiled = compile(graph, &runtime.kernels)
                .map_err(|e| VlError::Run(format!("compile: {:?}", e)))?;

            let bytes: Vec<u8> = embeds.iter().flat_map(|v| v.to_le_bytes()).collect();
            let input = Tensor {
                meta: TensorMeta::new(Dtype::F32, &[seq, d]),
                storage: std::sync::Arc::new(TensorStorage { bytes, mapped: None }),
            };
            let mut inputs = HashMap::new();
            inputs.insert("input_embeds".to_string(), input);

            let res = execute(&compiled, inputs, ExecutionOptions::default())
                .map_err(|e| VlError::Run(format!("execute: {:?}", e)))?;
            total_joules += res.trace.joule_accounting.total_joules;

            let logits = res.outputs.get("logits")
                .ok_or_else(|| VlError::Run("no logits".into()))?;
            let l = logits.as_f32_vec();
            let last = &l[(seq - 1) * v..seq * v];

            let mut best = 0usize;
            let mut best_v = f32::NEG_INFINITY;
            for (i, &x) in last.iter().enumerate() {
                if x.is_finite() && x > best_v { best_v = x; best = i; }
            }
            let tok = best as u32;
            if Some(tok) == eos { break; }
            generated.push(tok);
        }

        let caption = self.vocab.decode_bpe(&generated);
        Ok((caption, total_joules))
    }
}
