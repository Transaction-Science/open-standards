//! Byte-Pair Encoding (BPE) tokenizer implementation.
//!
//! This implements the BPE algorithm used by GPT-2, GPT-3, LLaMA, and many other models.
//! Supports loading from HuggingFace tokenizer.json format.

use crate::core::{Error, Result};
use std::collections::HashMap;
use std::path::Path;

/// A BPE tokenizer.
pub struct BpeTokenizer {
    /// Encoder: token string -> token id
    encoder: HashMap<String, u32>,
    /// Decoder: token id -> token string
    decoder: Vec<String>,
    /// BPE merges: (pair_rank, (token1, token2))
    merges: Vec<(String, String)>,
    /// Merge rankings: (token1, token2) -> rank
    merge_ranks: HashMap<(String, String), usize>,
    /// Byte encoder for handling arbitrary bytes
    byte_encoder: HashMap<u8, char>,
    /// Byte decoder (reverse of byte_encoder)
    byte_decoder: HashMap<char, u8>,
    /// Added tokens (special tokens that bypass BPE)
    added_tokens: HashMap<String, u32>,
    /// Pattern for splitting text (pre-tokenization)
    pattern: Option<regex::Regex>,
    /// CLIP-style BPE: lowercase + whitespace-collapse normalisation, CLIP
    /// pretokenizer regex, and an end-of-word suffix appended to the last
    /// symbol of each word before merging. Detected from tokenizer.json's
    /// `model.end_of_word_suffix` ("</w>" for OpenAI CLIP). GPT-2 BPE (the
    /// other path) uses a leading-space `Ġ` scheme and no case folding.
    eow_suffix: Option<String>,
}

impl BpeTokenizer {
    /// Create a new BPE tokenizer.
    pub fn new(
        encoder: HashMap<String, u32>,
        merges: Vec<(String, String)>,
    ) -> Self {
        // Build decoder
        let mut decoder = vec![String::new(); encoder.len()];
        for (token, &id) in &encoder {
            if (id as usize) < decoder.len() {
                decoder[id as usize] = token.clone();
            }
        }

        // Build merge ranks
        let merge_ranks: HashMap<(String, String), usize> = merges
            .iter()
            .enumerate()
            .map(|(i, (a, b))| ((a.clone(), b.clone()), i))
            .collect();

        // Build byte encoder (GPT-2 style)
        let byte_encoder = bytes_to_unicode();
        let byte_decoder: HashMap<char, u8> = byte_encoder.iter().map(|(&k, &v)| (v, k)).collect();

        Self {
            encoder,
            decoder,
            merges,
            merge_ranks,
            byte_encoder,
            byte_decoder,
            added_tokens: HashMap::new(),
            pattern: None,
            eow_suffix: None,
        }
    }

    /// Load from HuggingFace tokenizer.json format.
    pub fn from_file(path: &Path) -> Result<Self> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| Error::io("read", format!("{}: {}", path.display(), e)))?;

        Self::from_json(&content)
    }

    /// Parse from JSON string.
    pub fn from_json(json: &str) -> Result<Self> {
        // Simple JSON parsing for tokenizer.json
        // Format: { "model": { "vocab": {...}, "merges": [...] }, ... }

        let mut encoder = HashMap::new();
        let mut merges = Vec::new();
        let mut added_tokens = HashMap::new();

        // Find vocab section
        if let Some(vocab_start) = json.find("\"vocab\"") {
            if let Some(obj_start) = json[vocab_start..].find('{') {
                let start = vocab_start + obj_start;
                if let Some(obj_end) = find_matching_brace(&json[start..]) {
                    let vocab_json = &json[start..start + obj_end + 1];
                    encoder = parse_vocab(vocab_json)?;
                }
            }
        }

        // Find merges section
        if let Some(merges_start) = json.find("\"merges\"") {
            if let Some(arr_start) = json[merges_start..].find('[') {
                let start = merges_start + arr_start;
                if let Some(arr_end) = find_matching_bracket(&json[start..]) {
                    let merges_json = &json[start..start + arr_end + 1];
                    merges = parse_merges(merges_json)?;
                }
            }
        }

        // Find added_tokens section
        if let Some(added_start) = json.find("\"added_tokens\"") {
            if let Some(arr_start) = json[added_start..].find('[') {
                let start = added_start + arr_start;
                if let Some(arr_end) = find_matching_bracket(&json[start..]) {
                    let added_json = &json[start..start + arr_end + 1];
                    added_tokens = parse_added_tokens(added_json)?;
                }
            }
        }

        let mut tokenizer = Self::new(encoder, merges);
        tokenizer.added_tokens = added_tokens;

        // Detect CLIP-style BPE via `model.end_of_word_suffix`. The HF
        // CLIPTokenizerFast tokenizer.json sets it to "</w>".
        let eow = extract_json_string(json, "\"end_of_word_suffix\"");
        let is_clip = eow.as_deref().map(|s| !s.is_empty()).unwrap_or(false);

        if is_clip {
            tokenizer.eow_suffix = eow;
            // CLIP pretokenizer regex (no leading-space handling — CLIP
            // normalises whitespace and lowercases first; ByteLevel has
            // add_prefix_space=false). `[\p{N}]` is a SINGLE digit.
            if let Ok(re) = regex::Regex::new(
                r"'s|'t|'re|'ve|'m|'ll|'d|[\p{L}]+|[\p{N}]|[^\s\p{L}\p{N}]+",
            ) {
                tokenizer.pattern = Some(re);
            }
        } else {
            // GPT-2 pattern.
            if let Ok(re) = regex::Regex::new(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+") {
                tokenizer.pattern = Some(re);
            }
        }

        Ok(tokenizer)
    }

    /// Encode text to token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        // Check for added tokens first
        let mut remaining = text;
        while !remaining.is_empty() {
            let mut found_added = false;
            for (token, &id) in &self.added_tokens {
                if remaining.starts_with(token) {
                    ids.push(id);
                    remaining = &remaining[token.len()..];
                    found_added = true;
                    break;
                }
            }

            if !found_added {
                // Find next added token position
                let mut next_added_pos = remaining.len();
                for token in self.added_tokens.keys() {
                    if let Some(pos) = remaining.find(token) {
                        if pos < next_added_pos {
                            next_added_pos = pos;
                        }
                    }
                }

                // Encode text before next added token
                let chunk = &remaining[..next_added_pos];
                if !chunk.is_empty() {
                    ids.extend(self.encode_chunk(chunk));
                }
                remaining = &remaining[next_added_pos..];
            }
        }

        ids
    }

    /// Encode a chunk of text (no added tokens).
    fn encode_chunk(&self, text: &str) -> Vec<u32> {
        let mut ids = Vec::new();

        // CLIP normaliser: collapse all whitespace runs to a single space,
        // trim, lowercase (matches tokenizer.json `Sequence[NFC, Replace
        // \s+→" ", Lowercase]`; NFC is a no-op for ASCII prompts). Bind to a
        // String so the regex below borrows from normalised text.
        let normalised: String = if self.eow_suffix.is_some() {
            let collapsed: String = {
                let mut s = String::with_capacity(text.len());
                let mut prev_ws = false;
                for ch in text.chars() {
                    if ch.is_whitespace() {
                        if !prev_ws { s.push(' '); }
                        prev_ws = true;
                    } else {
                        s.push(ch);
                        prev_ws = false;
                    }
                }
                s.trim().to_string()
            };
            collapsed.to_lowercase()
        } else {
            text.to_string()
        };
        let work: &str = if self.eow_suffix.is_some() { &normalised } else { text };

        // Pre-tokenization using pattern
        let tokens: Vec<&str> = if let Some(ref pattern) = self.pattern {
            pattern.find_iter(work).map(|m| m.as_str()).collect()
        } else {
            work.split_whitespace().collect()
        };

        for token in tokens {
            // Convert to bytes then to BPE string representation
            let bpe_token: String = token
                .bytes()
                .map(|b| self.byte_encoder.get(&b).copied().unwrap_or('?'))
                .collect();

            // Apply BPE
            let bpe_tokens = self.bpe(&bpe_token);

            // Convert to IDs
            for bpe_tok in bpe_tokens {
                if let Some(&id) = self.encoder.get(&bpe_tok) {
                    ids.push(id);
                }
            }
        }

        ids
    }

    /// Apply BPE to a string.
    fn bpe(&self, token: &str) -> Vec<String> {
        if token.is_empty() {
            return Vec::new();
        }

        // Start with each character as a separate symbol. For CLIP, the
        // last symbol of every word carries the end-of-word suffix BEFORE
        // merging (OpenAI CLIP: `word = tuple(t[:-1]) + (t[-1]+'</w>',)`),
        // and the vocab/merge keys are in that suffixed form.
        let mut word: Vec<String> = token.chars().map(|c| c.to_string()).collect();
        if let Some(eow) = &self.eow_suffix {
            if let Some(last) = word.last_mut() {
                last.push_str(eow);
            }
        }

        if word.len() == 1 {
            return word;
        }

        loop {
            // Find the pair with lowest rank
            let mut best_pair: Option<(usize, String, String)> = None;
            let mut best_rank = usize::MAX;

            for i in 0..word.len() - 1 {
                let pair = (word[i].clone(), word[i + 1].clone());
                if let Some(&rank) = self.merge_ranks.get(&pair) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pair = Some((i, pair.0, pair.1));
                    }
                }
            }

            // No more merges to apply
            let Some((idx, first, second)) = best_pair else {
                break;
            };

            // Apply merge
            let merged = format!("{}{}", first, second);
            word[idx] = merged;
            word.remove(idx + 1);

            if word.len() == 1 {
                break;
            }
        }

        word
    }

    /// Decode token IDs to text.
    pub fn decode(&self, ids: &[u32]) -> String {
        let mut bytes = Vec::new();

        for &id in ids {
            if let Some(token) = self.decoder.get(id as usize) {
                // Convert BPE characters back to bytes
                for c in token.chars() {
                    if let Some(&byte) = self.byte_decoder.get(&c) {
                        bytes.push(byte);
                    }
                }
            }
        }

        String::from_utf8_lossy(&bytes).to_string()
    }

    /// Get vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.encoder.len() + self.added_tokens.len()
    }

    /// Get token ID for a token string.
    pub fn token_to_id(&self, token: &str) -> Option<u32> {
        self.encoder.get(token).copied()
            .or_else(|| self.added_tokens.get(token).copied())
    }

    /// Get token string for an ID.
    pub fn id_to_token(&self, id: u32) -> Option<&str> {
        self.decoder.get(id as usize).map(|s| s.as_str())
    }
}

/// Build the byte encoder used by GPT-2 tokenizer.
fn bytes_to_unicode() -> HashMap<u8, char> {
    let mut bs: Vec<u8> = Vec::new();

    // Printable ASCII and extended Latin
    bs.extend(b'!'..=b'~');
    bs.extend(0xa1u8..=0xac);
    bs.extend(0xaeu8..=0xff);

    let mut cs: Vec<u32> = bs.iter().map(|&b| b as u32).collect();
    let mut n = 0u32;

    for b in 0u8..=255 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }

    bs.iter()
        .zip(cs.iter())
        .map(|(&b, &c)| (b, char::from_u32(c).unwrap_or('?')))
        .collect()
}

/// Extract a JSON string value for `"key"` (best-effort flat scan). Returns
/// the unescaped value, or None if the key is absent or the value is null.
fn extract_json_string(json: &str, key: &str) -> Option<String> {
    let k = json.find(key)?;
    let after = &json[k + key.len()..];
    let colon = after.find(':')?;
    let rest = after[colon + 1..].trim_start();
    if rest.starts_with("null") {
        return None;
    }
    let q = rest.find('"')?;
    let body = &rest[q + 1..];
    let mut out = String::new();
    let mut chars = body.chars();
    while let Some(c) = chars.next() {
        match c {
            '"' => return Some(out),
            '\\' => {
                if let Some(n) = chars.next() {
                    match n {
                        'n' => out.push('\n'),
                        't' => out.push('\t'),
                        'r' => out.push('\r'),
                        other => out.push(other),
                    }
                }
            }
            other => out.push(other),
        }
    }
    None
}

/// Find the position of the matching closing brace.
fn find_matching_brace(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;

    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }

        match c {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '{' if !in_string => depth += 1,
            '}' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Find the position of the matching closing bracket.
fn find_matching_bracket(s: &str) -> Option<usize> {
    let mut depth = 0;
    let mut in_string = false;
    let mut escape = false;

    for (i, c) in s.char_indices() {
        if escape {
            escape = false;
            continue;
        }

        match c {
            '\\' if in_string => escape = true,
            '"' => in_string = !in_string,
            '[' if !in_string => depth += 1,
            ']' if !in_string => {
                depth -= 1;
                if depth == 0 {
                    return Some(i);
                }
            }
            _ => {}
        }
    }
    None
}

/// Parse vocab from JSON object.
fn parse_vocab(json: &str) -> Result<HashMap<String, u32>> {
    let mut vocab = HashMap::new();

    // Simple parser for {"token": id, ...}
    let inner = json.trim().trim_start_matches('{').trim_end_matches('}');

    let mut pos = 0;
    while pos < inner.len() {
        // Skip whitespace and commas
        while pos < inner.len() {
            let c = inner.as_bytes()[pos];
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' || c == b',' {
                pos += 1;
            } else {
                break;
            }
        }

        if pos >= inner.len() {
            break;
        }

        // Parse key
        if inner.as_bytes()[pos] != b'"' {
            pos += 1;
            continue;
        }

        let key_start = pos + 1;
        pos += 1;
        let mut escaped = false;
        while pos < inner.len() {
            if escaped {
                escaped = false;
                pos += 1;
                continue;
            }
            if inner.as_bytes()[pos] == b'\\' {
                escaped = true;
                pos += 1;
                continue;
            }
            if inner.as_bytes()[pos] == b'"' {
                break;
            }
            pos += 1;
        }
        let key = unescape_json_string(&inner[key_start..pos]);
        pos += 1;

        // Skip colon
        while pos < inner.len() && inner.as_bytes()[pos] != b':' {
            pos += 1;
        }
        pos += 1;

        // Skip whitespace
        while pos < inner.len() {
            let c = inner.as_bytes()[pos];
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' {
                pos += 1;
            } else {
                break;
            }
        }

        // Parse value (number)
        let val_start = pos;
        while pos < inner.len() {
            let c = inner.as_bytes()[pos];
            if c.is_ascii_digit() || c == b'-' {
                pos += 1;
            } else {
                break;
            }
        }

        if val_start < pos {
            if let Ok(id) = inner[val_start..pos].parse::<u32>() {
                vocab.insert(key, id);
            }
        }
    }

    Ok(vocab)
}

/// Parse merges from JSON array.
fn parse_merges(json: &str) -> Result<Vec<(String, String)>> {
    // Two formats in the wild:
    //   GPT-2 / legacy HF : ["a b", "c d", …]            (space-joined)
    //   CLIP / newer HF   : [["a","b"], ["c","d"], …]    (array of pairs)
    // The old code only handled the first (split on space) so EVERY CLIP
    // merge was silently dropped → BPE became a no-op → every char became
    // its own token → text embeddings ~orthogonal to reference. Detect the
    // array-of-pairs form by the first non-space char after the outer '['.
    let trimmed = json.trim();
    let body = trimmed.trim_start_matches('[');
    if body.trim_start().starts_with('[') {
        // Array of 2-element string arrays. Collect quoted strings in order
        // and pair them consecutively.
        let mut strings: Vec<String> = Vec::new();
        let bytes = body.as_bytes();
        let mut pos = 0;
        while pos < bytes.len() {
            if bytes[pos] == b'"' {
                let start = pos + 1;
                pos += 1;
                let mut escaped = false;
                while pos < bytes.len() {
                    if escaped { escaped = false; pos += 1; continue; }
                    match bytes[pos] {
                        b'\\' => { escaped = true; pos += 1; }
                        b'"' => break,
                        _ => pos += 1,
                    }
                }
                strings.push(unescape_json_string(&body[start..pos]));
            }
            pos += 1;
        }
        let mut merges = Vec::with_capacity(strings.len() / 2);
        let mut it = strings.into_iter();
        while let (Some(a), Some(b)) = (it.next(), it.next()) {
            merges.push((a, b));
        }
        return Ok(merges);
    }

    let mut merges = Vec::new();

    // GPT-2 form: array of "a b" strings.
    let inner = json.trim().trim_start_matches('[').trim_end_matches(']');

    let mut pos = 0;
    while pos < inner.len() {
        // Skip whitespace and commas
        while pos < inner.len() {
            let c = inner.as_bytes()[pos];
            if c == b' ' || c == b'\n' || c == b'\r' || c == b'\t' || c == b',' {
                pos += 1;
            } else {
                break;
            }
        }

        if pos >= inner.len() {
            break;
        }

        // Expect opening quote
        if inner.as_bytes()[pos] != b'"' {
            pos += 1;
            continue;
        }

        // Find the content between quotes, handling escapes
        let str_start = pos + 1;
        pos += 1;
        let mut escaped = false;

        while pos < inner.len() {
            if escaped {
                escaped = false;
                pos += 1;
                continue;
            }
            if inner.as_bytes()[pos] == b'\\' {
                escaped = true;
                pos += 1;
                continue;
            }
            if inner.as_bytes()[pos] == b'"' {
                break;
            }
            pos += 1;
        }

        if pos > str_start {
            let merge_str = unescape_json_string(&inner[str_start..pos]);
            // Split on the LAST space to handle tokens that might contain spaces
            if let Some(space_pos) = merge_str.rfind(' ') {
                let first = merge_str[..space_pos].to_string();
                let second = merge_str[space_pos + 1..].to_string();
                merges.push((first, second));
            }
        }

        pos += 1; // Skip closing quote
    }

    Ok(merges)
}

/// Parse added tokens from JSON array.
fn parse_added_tokens(json: &str) -> Result<HashMap<String, u32>> {
    let mut tokens = HashMap::new();

    // Simple parser for [{"id": 0, "content": "<s>", ...}, ...]
    // Find each object
    let mut pos = 0;
    while let Some(obj_start) = json[pos..].find('{') {
        let start = pos + obj_start;
        if let Some(obj_end) = find_matching_brace(&json[start..]) {
            let obj = &json[start..start + obj_end + 1];

            // Extract id and content
            let mut id: Option<u32> = None;
            let mut content: Option<String> = None;

            if let Some(id_pos) = obj.find("\"id\"") {
                let after = &obj[id_pos + 4..];
                let after = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
                if let Some(end) = after.find(|c: char| !c.is_ascii_digit()) {
                    if let Ok(n) = after[..end].parse::<u32>() {
                        id = Some(n);
                    }
                }
            }

            if let Some(content_pos) = obj.find("\"content\"") {
                let after = &obj[content_pos + 9..];
                let after = after.trim_start_matches(|c: char| c.is_whitespace() || c == ':');
                if after.starts_with('"') {
                    let after = &after[1..];
                    if let Some(end) = after.find('"') {
                        content = Some(unescape_json_string(&after[..end]));
                    }
                }
            }

            if let (Some(id), Some(content)) = (id, content) {
                tokens.insert(content, id);
            }

            pos = start + obj_end + 1;
        } else {
            break;
        }
    }

    Ok(tokens)
}

/// Unescape a JSON string.
fn unescape_json_string(s: &str) -> String {
    let mut result = String::new();
    let mut chars = s.chars().peekable();

    while let Some(c) = chars.next() {
        if c == '\\' {
            match chars.next() {
                Some('n') => result.push('\n'),
                Some('r') => result.push('\r'),
                Some('t') => result.push('\t'),
                Some('\\') => result.push('\\'),
                Some('"') => result.push('"'),
                Some('u') => {
                    // Unicode escape \uXXXX
                    let hex: String = chars.by_ref().take(4).collect();
                    if let Ok(code) = u32::from_str_radix(&hex, 16) {
                        if let Some(c) = char::from_u32(code) {
                            result.push(c);
                        }
                    }
                }
                Some(c) => {
                    result.push('\\');
                    result.push(c);
                }
                None => result.push('\\'),
            }
        } else {
            result.push(c);
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bytes_to_unicode() {
        let encoder = bytes_to_unicode();
        assert_eq!(encoder.len(), 256);
    }

    #[test]
    fn test_simple_bpe() {
        let mut encoder = HashMap::new();
        encoder.insert("h".to_string(), 0);
        encoder.insert("e".to_string(), 1);
        encoder.insert("l".to_string(), 2);
        encoder.insert("o".to_string(), 3);
        encoder.insert("he".to_string(), 4);
        encoder.insert("ll".to_string(), 5);
        encoder.insert("lo".to_string(), 6);
        encoder.insert("hel".to_string(), 7);
        encoder.insert("llo".to_string(), 8);
        encoder.insert("hello".to_string(), 9);

        let merges = vec![
            ("h".to_string(), "e".to_string()),
            ("l".to_string(), "l".to_string()),
            ("l".to_string(), "o".to_string()),
            ("he".to_string(), "l".to_string()),
            ("ll".to_string(), "o".to_string()),
            ("hel".to_string(), "lo".to_string()),
        ];

        let tokenizer = BpeTokenizer::new(encoder, merges);
        assert_eq!(tokenizer.vocab_size(), 10);
    }
}
