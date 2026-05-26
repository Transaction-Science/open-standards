//! Typed weight storage materialized from the safetensors file.
//!
//! The inventory in [`crate::loader::ModelInventory`] gives us
//! shapes + offsets but not data. This module owns the in-memory
//! float buffers the forward pass operates on.
//!
//! All weights are stored as fp32 here even though the on-disk
//! format is fp16. Computing in fp32 matches HF's reference behavior
//! (the model is upcast on load when `torch_dtype=float32`) and
//! avoids fp16 intermediate rounding in attention softmax /
//! LayerNorm. Memory cost: ~1.7 GB for DeBERTa-v3-large vs 830 MB on
//! disk. Acceptable for a CPU implementation; the storage format can
//! switch back to fp16 once we have a kernel that does fp16 ops
//! correctly.

use std::fs::File;
use std::path::Path;

use half::f16;
use memmap2::Mmap;
use safetensors::{Dtype, SafeTensors};

use crate::config::ModelConfig;
use crate::loader::{LoaderError, ModelInventory};

/// A flat row-major fp32 buffer with a known shape.
#[derive(Debug, Clone)]
pub struct FloatTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl FloatTensor {
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(0)
    }

    pub fn cols(&self) -> usize {
        self.shape.get(1).copied().unwrap_or(0)
    }

    /// Row-major row slice. Panics on out-of-bounds.
    pub fn row(&self, i: usize) -> &[f32] {
        let c = self.cols();
        &self.data[i * c..(i + 1) * c]
    }
}

/// Embedding-layer weights (word embeddings + LayerNorm).
#[derive(Debug, Clone)]
pub struct EmbeddingWeights {
    pub word_embeddings: FloatTensor,
    pub layer_norm_weight: Vec<f32>,
    pub layer_norm_bias: Vec<f32>,
}

/// One transformer layer's attention sub-block weights. Names track
/// the safetensors layout `deberta.encoder.layer.N.attention.*`.
#[derive(Debug, Clone)]
pub struct AttentionWeights {
    /// `[hidden, hidden]`.
    pub query_proj_w: FloatTensor,
    pub query_proj_b: Vec<f32>,
    pub key_proj_w: FloatTensor,
    pub key_proj_b: Vec<f32>,
    pub value_proj_w: FloatTensor,
    pub value_proj_b: Vec<f32>,
    /// `[hidden, hidden]` — applied after the per-head context concat.
    pub output_dense_w: FloatTensor,
    pub output_dense_b: Vec<f32>,
    /// LayerNorm applied to `(output_dense(context) + residual)`.
    pub output_ln_gamma: Vec<f32>,
    pub output_ln_beta: Vec<f32>,
}

/// One transformer layer's FFN sub-block weights. Names track
/// `deberta.encoder.layer.N.{intermediate,output}.*`.
#[derive(Debug, Clone)]
pub struct FfnWeights {
    /// `[ffn_dim, hidden]`.
    pub intermediate_w: FloatTensor,
    pub intermediate_b: Vec<f32>,
    /// `[hidden, ffn_dim]`.
    pub output_w: FloatTensor,
    pub output_b: Vec<f32>,
    pub output_ln_gamma: Vec<f32>,
    pub output_ln_beta: Vec<f32>,
}

/// One full encoder layer = attention + FFN.
#[derive(Debug, Clone)]
pub struct LayerWeights {
    pub attention: AttentionWeights,
    pub ffn: FfnWeights,
}

/// Encoder-level weights shared across layers: relative-position
/// embeddings + the LayerNorm that's applied to them per
/// `norm_rel_ebd = "layer_norm"`.
#[derive(Debug, Clone)]
pub struct EncoderWeights {
    /// `[position_buckets * 2, hidden]`, fp32 from fp16 storage.
    pub rel_embeddings: FloatTensor,
    pub rel_ln_gamma: Vec<f32>,
    pub rel_ln_beta: Vec<f32>,
    pub layers: Vec<LayerWeights>,
}

/// Pooler + classifier head (DebertaV2ForSequenceClassification's
/// `ContextPooler` followed by `nn.Linear(hidden, num_labels)`).
#[derive(Debug, Clone)]
pub struct ClassificationHead {
    /// `[hidden, hidden]` — pooler dense applied to CLS token.
    pub pooler_w: FloatTensor,
    pub pooler_b: Vec<f32>,
    /// `[num_labels, hidden]` — final NLI logits.
    pub classifier_w: FloatTensor,
    pub classifier_b: Vec<f32>,
}

/// Whole-model weights. Each phase fills in more fields:
///   - 4d: embeddings
///   - 4e: encoder.rel_embeddings + encoder.rel_ln + layers[0].attention
///   - 4f: all 24 layers (attention + ffn) + pooler + classifier
#[derive(Debug, Clone)]
pub struct Weights {
    pub config: ModelConfig,
    pub embeddings: EmbeddingWeights,
    /// Populated once the encoder is loaded (phase 4e+).
    pub encoder: Option<EncoderWeights>,
    /// Populated once the classification head is loaded (phase 4f+).
    pub head: Option<ClassificationHead>,
}

impl Weights {
    /// Materialize the embedding sub-layer's weights from the
    /// safetensors file at `model_dir/model.safetensors`. The
    /// inventory's already been validated by [`ModelInventory::from_dir`].
    pub fn load_embeddings_only(
        model_dir: impl AsRef<Path>,
        inventory: &ModelInventory,
    ) -> Result<Self, LoaderError> {
        let path = model_dir.as_ref().join("model.safetensors");
        let file = File::open(&path)
            .map_err(|e| LoaderError::Io(format!("open safetensors: {e}")))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| LoaderError::Io(format!("mmap: {e}")))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| LoaderError::Parse(format!("safetensors deserialize: {e}")))?;

        let word_embeddings = read_tensor_to_f32(
            &st,
            "deberta.embeddings.word_embeddings.weight",
        )?;
        let layer_norm_weight = read_tensor_to_f32(
            &st,
            "deberta.embeddings.LayerNorm.weight",
        )?
        .data;
        let layer_norm_bias = read_tensor_to_f32(
            &st,
            "deberta.embeddings.LayerNorm.bias",
        )?
        .data;

        Ok(Self {
            config: inventory.config.clone(),
            embeddings: EmbeddingWeights {
                word_embeddings,
                layer_norm_weight,
                layer_norm_bias,
            },
            encoder: None,
            head: None,
        })
    }

    /// Load embeddings + encoder shared weights + layer-0 attention.
    /// Used by the Phase 4e attention verification test. Subsequent
    /// phases extend this to load all layers + FFN + pooler +
    /// classifier.
    pub fn load_embeddings_and_layer0_attention(
        model_dir: impl AsRef<Path>,
        inventory: &ModelInventory,
    ) -> Result<Self, LoaderError> {
        let path = model_dir.as_ref().join("model.safetensors");
        let file = File::open(&path)
            .map_err(|e| LoaderError::Io(format!("open safetensors: {e}")))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| LoaderError::Io(format!("mmap: {e}")))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| LoaderError::Parse(format!("safetensors deserialize: {e}")))?;

        let embeddings = EmbeddingWeights {
            word_embeddings: read_tensor_to_f32(
                &st,
                "deberta.embeddings.word_embeddings.weight",
            )?,
            layer_norm_weight: read_tensor_to_f32(
                &st,
                "deberta.embeddings.LayerNorm.weight",
            )?
            .data,
            layer_norm_bias: read_tensor_to_f32(
                &st,
                "deberta.embeddings.LayerNorm.bias",
            )?
            .data,
        };

        let rel_embeddings = read_tensor_to_f32(
            &st,
            "deberta.encoder.rel_embeddings.weight",
        )?;
        let rel_ln_gamma = read_tensor_to_f32(
            &st,
            "deberta.encoder.LayerNorm.weight",
        )?
        .data;
        let rel_ln_beta = read_tensor_to_f32(
            &st,
            "deberta.encoder.LayerNorm.bias",
        )?
        .data;
        let layer0_attention = read_attention_weights(&st, 0)?;

        // Phase 4e doesn't need the FFN, but the LayerWeights struct
        // requires it. Materialize layer 0's FFN too so the public
        // shape is honest, even though the attention test won't use
        // those fields. Cheap relative to total memory.
        let layer0_ffn = read_ffn_weights(&st, 0)?;

        Ok(Self {
            config: inventory.config.clone(),
            embeddings,
            encoder: Some(EncoderWeights {
                rel_embeddings,
                rel_ln_gamma,
                rel_ln_beta,
                layers: vec![LayerWeights {
                    attention: layer0_attention,
                    ffn: layer0_ffn,
                }],
            }),
            head: None,
        })
    }

    /// Load every weight needed for an end-to-end forward pass:
    /// embeddings + all 24 encoder layers + pooler + classifier.
    /// ~1.7 GB in fp32 once materialized.
    pub fn load_full(
        model_dir: impl AsRef<Path>,
        inventory: &ModelInventory,
    ) -> Result<Self, LoaderError> {
        let path = model_dir.as_ref().join("model.safetensors");
        let file = File::open(&path)
            .map_err(|e| LoaderError::Io(format!("open safetensors: {e}")))?;
        let mmap = unsafe { Mmap::map(&file) }
            .map_err(|e| LoaderError::Io(format!("mmap: {e}")))?;
        let st = SafeTensors::deserialize(&mmap)
            .map_err(|e| LoaderError::Parse(format!("safetensors deserialize: {e}")))?;

        let embeddings = EmbeddingWeights {
            word_embeddings: read_tensor_to_f32(
                &st,
                "deberta.embeddings.word_embeddings.weight",
            )?,
            layer_norm_weight: read_tensor_to_f32(
                &st,
                "deberta.embeddings.LayerNorm.weight",
            )?
            .data,
            layer_norm_bias: read_tensor_to_f32(
                &st,
                "deberta.embeddings.LayerNorm.bias",
            )?
            .data,
        };

        let rel_embeddings = read_tensor_to_f32(
            &st,
            "deberta.encoder.rel_embeddings.weight",
        )?;
        let rel_ln_gamma = read_tensor_to_f32(
            &st,
            "deberta.encoder.LayerNorm.weight",
        )?
        .data;
        let rel_ln_beta = read_tensor_to_f32(
            &st,
            "deberta.encoder.LayerNorm.bias",
        )?
        .data;

        let n = inventory.config.num_hidden_layers;
        let mut layers = Vec::with_capacity(n);
        for layer_idx in 0..n {
            layers.push(LayerWeights {
                attention: read_attention_weights(&st, layer_idx)?,
                ffn: read_ffn_weights(&st, layer_idx)?,
            });
        }

        let head = ClassificationHead {
            pooler_w: read_tensor_to_f32(&st, "pooler.dense.weight")?,
            pooler_b: read_tensor_to_f32(&st, "pooler.dense.bias")?.data,
            classifier_w: read_tensor_to_f32(&st, "classifier.weight")?,
            classifier_b: read_tensor_to_f32(&st, "classifier.bias")?.data,
        };

        Ok(Self {
            config: inventory.config.clone(),
            embeddings,
            encoder: Some(EncoderWeights {
                rel_embeddings,
                rel_ln_gamma,
                rel_ln_beta,
                layers,
            }),
            head: Some(head),
        })
    }
}

fn read_attention_weights(
    st: &SafeTensors,
    layer: usize,
) -> Result<AttentionWeights, LoaderError> {
    let base = format!("deberta.encoder.layer.{layer}.attention");
    Ok(AttentionWeights {
        query_proj_w: read_tensor_to_f32(st, &format!("{base}.self.query_proj.weight"))?,
        query_proj_b: read_tensor_to_f32(st, &format!("{base}.self.query_proj.bias"))?.data,
        key_proj_w: read_tensor_to_f32(st, &format!("{base}.self.key_proj.weight"))?,
        key_proj_b: read_tensor_to_f32(st, &format!("{base}.self.key_proj.bias"))?.data,
        value_proj_w: read_tensor_to_f32(st, &format!("{base}.self.value_proj.weight"))?,
        value_proj_b: read_tensor_to_f32(st, &format!("{base}.self.value_proj.bias"))?.data,
        output_dense_w: read_tensor_to_f32(st, &format!("{base}.output.dense.weight"))?,
        output_dense_b: read_tensor_to_f32(st, &format!("{base}.output.dense.bias"))?.data,
        output_ln_gamma: read_tensor_to_f32(st, &format!("{base}.output.LayerNorm.weight"))?
            .data,
        output_ln_beta: read_tensor_to_f32(st, &format!("{base}.output.LayerNorm.bias"))?
            .data,
    })
}

fn read_ffn_weights(st: &SafeTensors, layer: usize) -> Result<FfnWeights, LoaderError> {
    let base = format!("deberta.encoder.layer.{layer}");
    Ok(FfnWeights {
        intermediate_w: read_tensor_to_f32(st, &format!("{base}.intermediate.dense.weight"))?,
        intermediate_b: read_tensor_to_f32(st, &format!("{base}.intermediate.dense.bias"))?.data,
        output_w: read_tensor_to_f32(st, &format!("{base}.output.dense.weight"))?,
        output_b: read_tensor_to_f32(st, &format!("{base}.output.dense.bias"))?.data,
        output_ln_gamma: read_tensor_to_f32(st, &format!("{base}.output.LayerNorm.weight"))?.data,
        output_ln_beta: read_tensor_to_f32(st, &format!("{base}.output.LayerNorm.bias"))?.data,
    })
}

fn read_tensor_to_f32(
    st: &SafeTensors,
    name: &str,
) -> Result<FloatTensor, LoaderError> {
    let view = st
        .tensor(name)
        .map_err(|_| LoaderError::MissingTensor(name.into()))?;
    let shape: Vec<usize> = view.shape().to_vec();
    let data = match view.dtype() {
        Dtype::F16 => bytes_to_f32_via_f16(view.data()),
        Dtype::F32 => bytes_to_f32_direct(view.data()),
        other => {
            return Err(LoaderError::UnexpectedDtype {
                name: name.into(),
                expected: Dtype::F16,
                actual: other,
            });
        }
    };
    Ok(FloatTensor { shape, data })
}

fn bytes_to_f32_via_f16(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 2 == 0, "fp16 buffer must be even-length");
    let n = bytes.len() / 2;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(2) {
        let bits = u16::from_le_bytes([chunk[0], chunk[1]]);
        out.push(f16::from_bits(bits).to_f32());
    }
    out
}

fn bytes_to_f32_direct(bytes: &[u8]) -> Vec<f32> {
    assert!(bytes.len() % 4 == 0, "fp32 buffer must be 4-byte-aligned");
    let n = bytes.len() / 4;
    let mut out = Vec::with_capacity(n);
    for chunk in bytes.chunks_exact(4) {
        out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn workspace_root() -> PathBuf {
        PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .parent()
            .and_then(|p| p.parent())
            .map(|p| p.to_path_buf())
            .expect("workspace root")
    }

    fn model_dir() -> Option<PathBuf> {
        let p = workspace_root().join("models/deberta-v3-large-mnli");
        if p.join("model.safetensors").exists() {
            Some(p)
        } else {
            None
        }
    }

    #[test]
    fn loads_embedding_weights_with_expected_shapes() {
        let Some(dir) = model_dir() else {
            eprintln!("skip: model not downloaded");
            return;
        };
        let inv = ModelInventory::from_dir(&dir).expect("inventory");
        let weights = Weights::load_embeddings_only(&dir, &inv).expect("load");
        assert_eq!(weights.embeddings.word_embeddings.shape, vec![128100, 1024]);
        assert_eq!(
            weights.embeddings.word_embeddings.data.len(),
            128100 * 1024
        );
        assert_eq!(weights.embeddings.layer_norm_weight.len(), 1024);
        assert_eq!(weights.embeddings.layer_norm_bias.len(), 1024);
    }

    #[test]
    fn f16_bytes_decode_matches_half_crate() {
        // 0x3C00 in fp16 = 1.0
        let bytes = vec![0x00, 0x3C];
        let v = bytes_to_f32_via_f16(&bytes);
        assert_eq!(v, vec![1.0]);
        // 0xBC00 in fp16 = -1.0
        let bytes = vec![0x00, 0xBC];
        let v = bytes_to_f32_via_f16(&bytes);
        assert_eq!(v, vec![-1.0]);
    }
}
