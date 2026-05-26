//! Nemotron 3 architecture support.
//!
//! Nemotron 3 Super is a hybrid Mamba-2 + MoE + Attention model:
//! - 75% of layers are Mamba-2 SSM (linear time, constant memory)
//! - 25% of layers are Attention + MoE (standard transformer + sparse experts)
//! - 120B total params, 12B active per token
//! - Latent MoE: tokens projected to smaller latent dim for routing

/// Layer type in the Nemotron hybrid architecture.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NemotronLayerType {
    /// Mamba-2 SSM layer (selective state space, linear time complexity)
    Mamba2,
    /// Attention + MoE layer (standard transformer attention with sparse expert FFN)
    AttentionMoE,
}

/// Nemotron architecture configuration.
#[derive(Debug, Clone)]
pub struct NemotronConfig {
    /// Layer schedule: which type each layer is
    pub layer_types: Vec<NemotronLayerType>,
    /// Total number of layers
    pub num_layers: usize,
    /// Number of Mamba-2 SSM layers
    pub num_mamba_layers: usize,
    /// Number of Attention+MoE layers
    pub num_attention_layers: usize,
}

impl NemotronConfig {
    /// Create config from total layers, defaulting to 75/25 split.
    /// Attention layers are evenly distributed (every 4th layer).
    pub fn from_num_layers(num_layers: usize) -> Self {
        let num_attention = num_layers / 4;
        let num_mamba = num_layers - num_attention;

        let mut layer_types = Vec::with_capacity(num_layers);
        for i in 0..num_layers {
            if (i + 1) % 4 == 0 {
                layer_types.push(NemotronLayerType::AttentionMoE);
            } else {
                layer_types.push(NemotronLayerType::Mamba2);
            }
        }

        Self {
            layer_types,
            num_layers,
            num_mamba_layers: num_mamba,
            num_attention_layers: num_attention,
        }
    }

    /// Get the layer type for a given layer index.
    pub fn layer_type(&self, layer_idx: usize) -> NemotronLayerType {
        self.layer_types.get(layer_idx).copied().unwrap_or(NemotronLayerType::Mamba2)
    }

    /// Check if a layer is an attention+MoE layer.
    pub fn is_attention_layer(&self, layer_idx: usize) -> bool {
        self.layer_type(layer_idx) == NemotronLayerType::AttentionMoE
    }
}
