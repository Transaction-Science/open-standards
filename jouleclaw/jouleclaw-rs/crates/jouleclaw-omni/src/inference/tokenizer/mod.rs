//! Tokenizer support for LLM inference.
//!
//! Provides fast tokenization using:
//! - BPE (Byte-Pair Encoding) for most models
//! - SentencePiece for LLaMA-style models
//! - Memory-mapped vocabulary for instant loading

pub mod bpe;
pub mod hf;

pub use bpe::BpeTokenizer;
pub use hf::HfTokenizer;

use crate::core::{Error, Result};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// A tokenizer for text encoding/decoding.
pub struct Tokenizer {
    /// Tokenizer type
    tokenizer_type: TokenizerType,
    /// Token to ID mapping
    vocab: HashMap<String, u32>,
    /// ID to token mapping
    id_to_token: Vec<String>,
    /// Merges (for BPE)
    merges: Vec<(String, String)>,
    /// Special tokens
    special_tokens: SpecialTokens,
    /// Byte fallback tokens (for SentencePiece)
    byte_fallback: Option<HashMap<u8, u32>>,
    /// The real BPE engine (GPT-2 or CLIP-BPE). When present, `encode_bpe`
    /// delegates to it instead of the simplified inline splitter, which
    /// neither byte-encodes nor applies CLIP's `</w>` merge scheme.
    bpe: Option<bpe::BpeTokenizer>,
}

/// Type of tokenizer.
#[derive(Debug, Clone, Copy)]
pub enum TokenizerType {
    /// BPE tokenizer (GPT-style)
    BPE,
    /// SentencePiece (LLaMA-style)
    SentencePiece,
    /// WordPiece (BERT-style)
    WordPiece,
}

/// Special token IDs.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    /// Beginning of sequence
    pub bos_id: u32,
    /// End of sequence
    pub eos_id: u32,
    /// Padding token
    pub pad_id: u32,
    /// Unknown token
    pub unk_id: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos_id: 1,
            eos_id: 2,
            pad_id: 0,
            unk_id: 0,
        }
    }
}

impl Tokenizer {
    /// Load a tokenizer from a path.
    pub fn load(path: &Path) -> Result<Self> {
        // Detect format from file
        let filename = path.file_name()
            .and_then(|n| n.to_str())
            .unwrap_or("");

        if filename.ends_with(".json") {
            Self::load_json(path)
        } else if filename.ends_with(".model") {
            Self::load_sentencepiece(path)
        } else {
            Err(Error::internal("unsupported tokenizer format"))
        }
    }

    /// Load a HuggingFace tokenizer.json.
    fn load_json(path: &Path) -> Result<Self> {
        let bpe = bpe::BpeTokenizer::from_file(path)?;
        // The BPE tokenizer has the real vocab and merges
        // We need to build our Tokenizer from it

        // Try to detect special tokens from the parsed data
        let bos_id = bpe.token_to_id("<s>")
            .or_else(|| bpe.token_to_id("<|begin_of_text|>"))
            .unwrap_or(1);
        let eos_id = bpe.token_to_id("</s>")
            .or_else(|| bpe.token_to_id("<|end_of_text|>"))
            .unwrap_or(2);
        let pad_id = bpe.token_to_id("<pad>").unwrap_or(0);
        let unk_id = bpe.token_to_id("<unk>").unwrap_or(0);

        // Build vocab and id_to_token from BPE encoder
        let vocab_size = bpe.vocab_size();
        let mut vocab = HashMap::new();
        let mut id_to_token = vec![String::new(); vocab_size];

        for id in 0..vocab_size as u32 {
            if let Some(token) = bpe.id_to_token(id) {
                vocab.insert(token.to_string(), id);
                if (id as usize) < id_to_token.len() {
                    id_to_token[id as usize] = token.to_string();
                }
            }
        }

        Ok(Self {
            tokenizer_type: TokenizerType::BPE,
            vocab,
            id_to_token,
            merges: Vec::new(), // Merges are handled internally by BpeTokenizer
            special_tokens: SpecialTokens {
                bos_id,
                eos_id,
                pad_id,
                unk_id,
            },
            byte_fallback: None,
            bpe: Some(bpe),
        })
    }

    /// Load a SentencePiece model.
    fn load_sentencepiece(path: &Path) -> Result<Self> {
        // SentencePiece .model files are protobuf format
        // Parse the vocabulary pieces from the binary data
        let data = std::fs::read(path)
            .map_err(|e| Error::io("read", format!("{}: {}", path.display(), e)))?;

        let mut vocab = HashMap::new();
        let mut id_to_token = Vec::new();
        let mut byte_fallback = HashMap::new();

        // Parse protobuf: field 1 (SentencePiece pieces) is a repeated message
        // Each piece has: field 1 = piece string, field 2 = score (float), field 3 = type (int)
        let mut pos = 0;
        while pos < data.len() {
            // Read protobuf tag
            let (tag, new_pos) = read_varint(&data, pos);
            if new_pos >= data.len() { break; }
            pos = new_pos;

            let field_number = tag >> 3;
            let wire_type = tag & 0x7;

            if field_number == 1 && wire_type == 2 {
                // Length-delimited: a SentencePiece piece message
                let (len, new_pos) = read_varint(&data, pos);
                pos = new_pos;
                let end = pos + len as usize;
                if end > data.len() { break; }

                // Parse inner message to get piece string
                let piece_data = &data[pos..end];
                if let Some(piece_str) = parse_piece_string(piece_data) {
                    let id = id_to_token.len() as u32;
                    vocab.insert(piece_str.clone(), id);
                    id_to_token.push(piece_str);
                }
                pos = end;
            } else if wire_type == 0 {
                // Varint
                let (_, new_pos) = read_varint(&data, pos);
                pos = new_pos;
            } else if wire_type == 1 {
                // 64-bit
                pos += 8;
            } else if wire_type == 2 {
                // Length-delimited (skip)
                let (len, new_pos) = read_varint(&data, pos);
                pos = new_pos + len as usize;
            } else if wire_type == 5 {
                // 32-bit
                pos += 4;
            } else {
                pos += 1;
            }
        }

        // Build byte fallback for tokens like <0xAB>
        for (token_str, &id) in &vocab {
            if token_str.starts_with("<0x") && token_str.ends_with('>') && token_str.len() == 6 {
                if let Ok(byte) = u8::from_str_radix(&token_str[3..5], 16) {
                    byte_fallback.insert(byte, id);
                }
            }
        }

        // Detect special tokens
        let bos_id = vocab.get("<s>").copied().unwrap_or(1);
        let eos_id = vocab.get("</s>").copied().unwrap_or(2);
        let unk_id = vocab.get("<unk>").copied().unwrap_or(0);

        Ok(Self {
            tokenizer_type: TokenizerType::SentencePiece,
            vocab,
            id_to_token,
            merges: Vec::new(),
            special_tokens: SpecialTokens {
                bos_id,
                eos_id,
                pad_id: 0,
                unk_id,
            },
            byte_fallback: if byte_fallback.is_empty() { None } else { Some(byte_fallback) },
            bpe: None,
        })
    }

    /// Create a simple whitespace tokenizer.
    pub fn simple() -> Self {
        Self {
            tokenizer_type: TokenizerType::BPE,
            vocab: HashMap::new(),
            id_to_token: Vec::new(),
            merges: Vec::new(),
            special_tokens: SpecialTokens::default(),
            byte_fallback: None,
            bpe: None,
        }
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.vocab.len()
    }

    /// Get special tokens.
    pub fn special_tokens(&self) -> &SpecialTokens {
        &self.special_tokens
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> EncodingResult {
        match self.tokenizer_type {
            TokenizerType::BPE => self.encode_bpe(text),
            TokenizerType::SentencePiece => self.encode_sentencepiece(text),
            TokenizerType::WordPiece => self.encode_wordpiece(text),
        }
    }

    /// Encode with BOS/EOS tokens.
    pub fn encode_with_special(&self, text: &str, add_bos: bool, add_eos: bool) -> EncodingResult {
        let mut result = self.encode(text);

        if add_bos {
            result.ids.insert(0, self.special_tokens.bos_id);
        }

        if add_eos {
            result.ids.push(self.special_tokens.eos_id);
        }

        result
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut result = String::new();

        for &id in ids {
            if let Some(token) = self.id_to_token.get(id as usize) {
                // Handle SentencePiece spacing
                if self.tokenizer_type == TokenizerType::SentencePiece {
                    if token.starts_with("▁") {
                        if !result.is_empty() {
                            result.push(' ');
                        }
                        result.push_str(&token[3..]); // Skip ▁
                    } else if token.starts_with("<0x") {
                        // Byte token
                        if let Some(byte) = self.parse_byte_token(token) {
                            result.push(byte as char);
                        }
                    } else if !token.starts_with("<") {
                        result.push_str(token);
                    }
                } else {
                    result.push_str(token);
                }
            }
        }

        result
    }

    /// Decode a single token.
    pub fn decode_token(&self, id: u32) -> Option<&str> {
        self.id_to_token.get(id as usize).map(|s| s.as_str())
    }

    fn encode_bpe(&self, text: &str) -> EncodingResult {
        // Delegate to the real BPE engine (correct GPT-2 byte-level OR
        // CLIP-BPE with `</w>`), not the simplified inline splitter below.
        if let Some(bpe) = &self.bpe {
            return EncodingResult { ids: bpe.encode(text) };
        }

        let mut ids = Vec::new();

        // Pre-tokenize by splitting on whitespace and punctuation boundaries
        // This is a simplified pre-tokenization; real GPT-2 uses a regex
        for word in text.split_inclusive(|c: char| c.is_whitespace() || c.is_ascii_punctuation()) {
            let trimmed = word.trim_end();
            let trailing = &word[trimmed.len()..];

            // Try to find the word directly in vocab
            if !trimmed.is_empty() {
                if let Some(&id) = self.vocab.get(trimmed) {
                    ids.push(id);
                } else {
                    // Try BPE merge
                    let bpe_tokens = self.apply_bpe(trimmed);
                    ids.extend(bpe_tokens);
                }
            }

            // Handle trailing whitespace/punctuation
            if !trailing.is_empty() {
                if let Some(&id) = self.vocab.get(trailing) {
                    ids.push(id);
                } else {
                    for byte in trailing.bytes() {
                        if let Some(&id) = self.byte_fallback.as_ref().and_then(|m| m.get(&byte)) {
                            ids.push(id);
                        } else {
                            ids.push(self.special_tokens.unk_id);
                        }
                    }
                }
            }
        }

        EncodingResult { ids }
    }

    fn apply_bpe(&self, word: &str) -> Vec<u32> {
        // Start with individual characters/bytes
        let mut tokens: Vec<String> = word.chars().map(|c| c.to_string()).collect();

        if tokens.len() <= 1 {
            // Single character - look up directly
            return tokens.iter()
                .filter_map(|t| self.vocab.get(t).copied())
                .collect();
        }

        // Iteratively merge the most frequent pair
        loop {
            let mut best_pair: Option<(usize, usize)> = None; // (index, merge_rank)
            let mut best_rank = usize::MAX;

            for i in 0..tokens.len() - 1 {
                // Check merges list for this pair
                for (rank, (a, b)) in self.merges.iter().enumerate() {
                    if *a == tokens[i] && *b == tokens[i + 1] && rank < best_rank {
                        best_rank = rank;
                        best_pair = Some((i, rank));
                        break;
                    }
                }
            }

            match best_pair {
                Some((idx, _)) => {
                    let merged = format!("{}{}", tokens[idx], tokens[idx + 1]);
                    tokens[idx] = merged;
                    tokens.remove(idx + 1);
                    if tokens.len() <= 1 { break; }
                }
                None => break,
            }
        }

        // Convert tokens to IDs
        tokens.iter()
            .map(|t| self.vocab.get(t).copied().unwrap_or(self.special_tokens.unk_id))
            .collect()
    }

    fn encode_sentencepiece(&self, text: &str) -> EncodingResult {
        let mut ids = Vec::new();

        // SentencePiece prepends ▁ (U+2581) to represent spaces/word boundaries
        // The first word also gets ▁ prefix in standard SentencePiece
        for word in text.split_whitespace() {
            // SentencePiece uses ▁ prefix for word boundaries (including the first word)
            let with_prefix = format!("▁{}", word);

            if let Some(&id) = self.vocab.get(&with_prefix) {
                ids.push(id);
            } else {
                // Try to greedily match subword tokens with ▁ prefix for the first piece
                let mut matched = false;

                // Try the word without prefix as well
                if let Some(&id) = self.vocab.get(word) {
                    ids.push(id);
                    matched = true;
                }

                if !matched {
                    // Greedy longest-match tokenization with ▁ prefix on the first subtoken
                    let chars: Vec<char> = with_prefix.chars().collect();
                    let mut pos = 0;

                    while pos < chars.len() {
                        let mut best_end = pos + 1;
                        // Try longest match first
                        for end in (pos + 1..=chars.len()).rev() {
                            let substr: String = chars[pos..end].iter().collect();
                            if self.vocab.contains_key(&substr) {
                                best_end = end;
                                break;
                            }
                        }

                        let substr: String = chars[pos..best_end].iter().collect();
                        if let Some(&id) = self.vocab.get(&substr) {
                            ids.push(id);
                        } else {
                            // Fall back to byte encoding for each character
                            for c in substr.chars() {
                                let mut char_buf = [0u8; 4];
                                let char_bytes = c.encode_utf8(&mut char_buf);
                                for byte in char_bytes.bytes() {
                                    if let Some(&id) = self.byte_fallback.as_ref().and_then(|m| m.get(&byte)) {
                                        ids.push(id);
                                    } else {
                                        ids.push(self.special_tokens.unk_id);
                                    }
                                }
                            }
                        }

                        pos = best_end;
                    }
                }
            }
        }

        EncodingResult { ids }
    }

    fn encode_wordpiece(&self, text: &str) -> EncodingResult {
        // WordPiece encoding (placeholder)
        self.encode_bpe(text)
    }

    fn parse_byte_token(&self, token: &str) -> Option<u8> {
        // Parse <0xAB> format
        if token.starts_with("<0x") && token.ends_with(">") {
            let hex = &token[3..token.len() - 1];
            u8::from_str_radix(hex, 16).ok()
        } else {
            None
        }
    }
}

impl PartialEq for TokenizerType {
    fn eq(&self, other: &Self) -> bool {
        matches!(
            (self, other),
            (TokenizerType::BPE, TokenizerType::BPE)
                | (TokenizerType::SentencePiece, TokenizerType::SentencePiece)
                | (TokenizerType::WordPiece, TokenizerType::WordPiece)
        )
    }
}

/// Result of text encoding.
#[derive(Debug, Clone)]
pub struct EncodingResult {
    /// Token IDs
    pub ids: Vec<u32>,
}

impl EncodingResult {
    /// Get token IDs.
    pub fn ids(&self) -> &[u32] {
        &self.ids
    }

    /// Get number of tokens.
    pub fn len(&self) -> usize {
        self.ids.len()
    }

    /// Check if empty.
    pub fn is_empty(&self) -> bool {
        self.ids.is_empty()
    }

    /// Pad to length.
    pub fn pad(&mut self, length: usize, pad_id: u32) {
        while self.ids.len() < length {
            self.ids.push(pad_id);
        }
    }

    /// Truncate to length.
    pub fn truncate(&mut self, length: usize) {
        self.ids.truncate(length);
    }
}

/// Batch encoder for efficient batch tokenization.
pub struct BatchEncoder {
    tokenizer: Arc<Tokenizer>,
    max_length: usize,
    padding: bool,
    truncation: bool,
}

impl BatchEncoder {
    /// Create a new batch encoder.
    pub fn new(tokenizer: Arc<Tokenizer>) -> Self {
        Self {
            tokenizer,
            max_length: 2048,
            padding: true,
            truncation: true,
        }
    }

    /// Set maximum length.
    pub fn max_length(mut self, length: usize) -> Self {
        self.max_length = length;
        self
    }

    /// Enable/disable padding.
    pub fn padding(mut self, padding: bool) -> Self {
        self.padding = padding;
        self
    }

    /// Enable/disable truncation.
    pub fn truncation(mut self, truncation: bool) -> Self {
        self.truncation = truncation;
        self
    }

    /// Encode a batch of texts.
    pub fn encode_batch(&self, texts: &[&str]) -> BatchEncodingResult {
        let mut results: Vec<EncodingResult> = texts
            .iter()
            .map(|text| self.tokenizer.encode(text))
            .collect();

        // Find max length in batch
        let max_len = results.iter().map(|r| r.len()).max().unwrap_or(0);
        let target_len = if self.truncation {
            max_len.min(self.max_length)
        } else {
            max_len
        };

        // Pad/truncate
        for result in &mut results {
            if self.truncation && result.len() > target_len {
                result.truncate(target_len);
            }
            if self.padding && result.len() < target_len {
                result.pad(target_len, self.tokenizer.special_tokens.pad_id);
            }
        }

        BatchEncodingResult {
            encodings: results,
            max_length: target_len,
        }
    }
}

/// Result of batch encoding.
#[derive(Debug)]
pub struct BatchEncodingResult {
    /// Individual encodings
    pub encodings: Vec<EncodingResult>,
    /// Maximum length in batch
    pub max_length: usize,
}

impl BatchEncodingResult {
    /// Get all IDs as a 2D vector.
    pub fn ids(&self) -> Vec<Vec<u32>> {
        self.encodings.iter().map(|e| e.ids.clone()).collect()
    }

    /// Get batch size.
    pub fn batch_size(&self) -> usize {
        self.encodings.len()
    }
}

/// Chat template for formatting conversations.
pub struct ChatTemplate {
    /// Template format
    format: ChatFormat,
    /// System message prefix
    system_prefix: String,
    /// User message prefix
    user_prefix: String,
    /// Assistant message prefix
    assistant_prefix: String,
    /// Message suffix
    suffix: String,
}

/// Chat template format.
#[derive(Debug, Clone, Copy)]
pub enum ChatFormat {
    /// LLaMA 2 format
    Llama2,
    /// ChatML format
    ChatML,
    /// Alpaca format
    Alpaca,
    /// Vicuna format
    Vicuna,
}

impl ChatTemplate {
    /// Create a LLaMA 2 chat template.
    pub fn llama2() -> Self {
        Self {
            format: ChatFormat::Llama2,
            system_prefix: "<<SYS>>\n".to_string(),
            user_prefix: "[INST] ".to_string(),
            assistant_prefix: " [/INST] ".to_string(),
            suffix: " </s>".to_string(),
        }
    }

    /// Create a ChatML template.
    pub fn chatml() -> Self {
        Self {
            format: ChatFormat::ChatML,
            system_prefix: "<|im_start|>system\n".to_string(),
            user_prefix: "<|im_start|>user\n".to_string(),
            assistant_prefix: "<|im_start|>assistant\n".to_string(),
            suffix: "<|im_end|>\n".to_string(),
        }
    }

    /// Apply template to messages.
    pub fn apply(&self, messages: &[ChatMessage]) -> String {
        let mut result = String::new();

        for msg in messages {
            match msg.role {
                ChatRole::System => {
                    result.push_str(&self.system_prefix);
                    result.push_str(&msg.content);
                    result.push_str(&self.suffix);
                }
                ChatRole::User => {
                    result.push_str(&self.user_prefix);
                    result.push_str(&msg.content);
                    result.push_str(&self.suffix);
                }
                ChatRole::Assistant => {
                    result.push_str(&self.assistant_prefix);
                    result.push_str(&msg.content);
                    result.push_str(&self.suffix);
                }
            }
        }

        // Add assistant prefix for generation
        if !result.ends_with(&self.assistant_prefix) {
            result.push_str(&self.assistant_prefix);
        }

        result
    }
}

/// A chat message.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    /// Message role
    pub role: ChatRole,
    /// Message content
    pub content: String,
}

/// Chat message role.
#[derive(Debug, Clone, Copy)]
pub enum ChatRole {
    /// System message
    System,
    /// User message
    User,
    /// Assistant message
    Assistant,
}

impl ChatMessage {
    /// Create a system message.
    pub fn system(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::System,
            content: content.into(),
        }
    }

    /// Create a user message.
    pub fn user(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::User,
            content: content.into(),
        }
    }

    /// Create an assistant message.
    pub fn assistant(content: impl Into<String>) -> Self {
        Self {
            role: ChatRole::Assistant,
            content: content.into(),
        }
    }
}

/// Read a varint from protobuf-encoded data at the given position.
/// Returns the decoded value and the new position after the varint.
fn read_varint(data: &[u8], mut pos: usize) -> (u64, usize) {
    let mut result: u64 = 0;
    let mut shift = 0;
    while pos < data.len() {
        let byte = data[pos];
        pos += 1;
        result |= ((byte & 0x7F) as u64) << shift;
        if byte & 0x80 == 0 {
            break;
        }
        shift += 7;
        if shift >= 64 { break; }
    }
    (result, pos)
}

/// Parse the piece string (field 1) from a SentencePiece piece protobuf message.
fn parse_piece_string(data: &[u8]) -> Option<String> {
    let mut pos = 0;
    while pos < data.len() {
        let (tag, new_pos) = read_varint(data, pos);
        pos = new_pos;
        let field_number = tag >> 3;
        let wire_type = tag & 0x7;

        if field_number == 1 && wire_type == 2 {
            // This is the piece string
            let (len, new_pos) = read_varint(data, pos);
            pos = new_pos;
            let end = pos + len as usize;
            if end <= data.len() {
                return String::from_utf8(data[pos..end].to_vec()).ok();
            }
            return None;
        } else if wire_type == 0 {
            let (_, new_pos) = read_varint(data, pos);
            pos = new_pos;
        } else if wire_type == 1 {
            pos += 8;
        } else if wire_type == 2 {
            let (len, new_pos) = read_varint(data, pos);
            pos = new_pos + len as usize;
        } else if wire_type == 5 {
            pos += 4;
        } else {
            break;
        }
    }
    None
}
