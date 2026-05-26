//! HuggingFace Tokenizers wrapper.
//!
//! Uses the official HuggingFace tokenizers crate for robust tokenization.

use crate::core::{Error, Result};
use std::path::Path;
use tokenizers::Tokenizer;

/// Wrapper around HuggingFace tokenizers for easy use.
pub struct HfTokenizer {
    tokenizer: Tokenizer,
}

impl HfTokenizer {
    /// Load a tokenizer from a tokenizer.json file.
    pub fn from_file(path: &Path) -> Result<Self> {
        let tokenizer = Tokenizer::from_file(path)
            .map_err(|e| Error::io("tokenizer", format!("Failed to load tokenizer: {}", e)))?;

        Ok(Self { tokenizer })
    }

    // Feature not currently enabled in Cargo.toml
    // #[cfg(feature = "tokenizers-http")]
    // pub fn from_pretrained(model_name: &str) -> Result<Self> {
    //     let tokenizer = Tokenizer::from_pretrained(model_name, None)
    //         .map_err(|e| Error::io("tokenizer", format!("Failed to load pretrained tokenizer: {}", e)))?;
    // 
    //     Ok(Self { tokenizer })
    // }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, false)
            .map_err(|e| Error::internal(format!("Encoding failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Encode text with special tokens (BOS/EOS as configured).
    pub fn encode_with_special(&self, text: &str) -> Result<Vec<u32>> {
        let encoding = self.tokenizer.encode(text, true)
            .map_err(|e| Error::internal(format!("Encoding failed: {}", e)))?;

        Ok(encoding.get_ids().to_vec())
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, true)
            .map_err(|e| Error::internal(format!("Decoding failed: {}", e)))
    }

    /// Decode without skipping special tokens.
    pub fn decode_all(&self, ids: &[u32]) -> Result<String> {
        self.tokenizer.decode(ids, false)
            .map_err(|e| Error::internal(format!("Decoding failed: {}", e)))
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.tokenizer.get_vocab_size(true)
    }

    /// Get token ID for a string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.tokenizer.token_to_id(token)
    }

    /// Get string for a token ID.
    pub fn id_to_token(&self, id: u32) -> Option<String> {
        self.tokenizer.id_to_token(id)
    }

    /// Get the underlying tokenizer for advanced operations.
    pub fn inner(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Encode a batch of texts.
    pub fn encode_batch(&self, texts: &[&str]) -> Result<Vec<Vec<u32>>> {
        let encodings = self.tokenizer.encode_batch(texts.to_vec(), false)
            .map_err(|e| Error::internal(format!("Batch encoding failed: {}", e)))?;

        Ok(encodings.into_iter().map(|e| e.get_ids().to_vec()).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // Requires a real tokenizer.json file
    fn test_hf_tokenizer_load() {
        let path = std::path::Path::new("test_tokenizer.json");
        if path.exists() {
            let tokenizer = HfTokenizer::from_file(path).unwrap();
            assert!(tokenizer.vocab_size() > 0);
        }
    }
}
