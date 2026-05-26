//! Chat template support for instruction-tuned models.
//!
//! Most modern instruction-tuned LLMs expect input formatted in a specific
//! chat template — `<|im_start|>user\n{msg}<|im_end|>\n<|im_start|>assistant\n`
//! for ChatML, `[INST] {msg} [/INST]` for Mistral, etc. Passing raw text
//! to these models bypasses their instruction tuning and produces noise.
//!
//! This module exposes:
//! - `ChatMessage` — a `(role, content)` pair.
//! - `ChatTemplate` — a small enum over common templates with a
//!   `format()` method that turns a `&[ChatMessage]` into a prompt string.
//! - `ChatTemplate::detect_from_model()` — read the GGUF
//!   `tokenizer.chat_template` field and try to match it against the
//!   known templates. Falls back to a manual choice.
//!
//! The supported templates cover the dominant production models as of
//! 2026: Llama 2, Llama 3, Mistral / Mixtral, Qwen2/3, Phi-3, and the
//! generic ChatML used by many open-weight finetunes.
//!
//! Real GGUF chat templates are Jinja2 strings. A full Jinja parser is
//! out of scope here; instead we pattern-match the loaded template
//! against known canonical forms and fall back to "Generic" when no
//! match is found.

use jouleclaw_loader_gguf::tokenizer::Vocab;
use jouleclaw_loader_gguf::GgufModel;

/// One message in a chat conversation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ChatMessage {
    pub role: ChatRole,
    pub content: String,
}

impl ChatMessage {
    pub fn system(content: impl Into<String>) -> Self {
        Self { role: ChatRole::System, content: content.into() }
    }
    pub fn user(content: impl Into<String>) -> Self {
        Self { role: ChatRole::User, content: content.into() }
    }
    pub fn assistant(content: impl Into<String>) -> Self {
        Self { role: ChatRole::Assistant, content: content.into() }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ChatRole {
    System,
    User,
    Assistant,
}

/// Supported chat template formats. Cover the dominant production models
/// as of 2026.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ChatTemplate {
    /// Llama 3 format:
    /// `<|begin_of_text|><|start_header_id|>{role}<|end_header_id|>\n\n{content}<|eot_id|>...`
    Llama3,
    /// Llama 2 format: `<s>[INST] <<SYS>>\n{sys}\n<</SYS>>\n\n{user} [/INST]`
    Llama2,
    /// Mistral / Mixtral: `<s>[INST] {user} [/INST]` (no system role natively)
    Mistral,
    /// ChatML (used by Qwen, Yi, many open-weight finetunes):
    /// `<|im_start|>{role}\n{content}<|im_end|>\n<|im_start|>assistant\n`
    ChatML,
    /// Phi-3: `<|user|>\n{content}<|end|>\n<|assistant|>\n`
    Phi3,
    /// Plain concatenation: no special tokens, roles as prefixes.
    /// Useful for base models or quick testing.
    Plain,
}

impl ChatTemplate {
    /// Try to detect the appropriate template from a model's GGUF metadata.
    /// Reads `tokenizer.chat_template` and pattern-matches against canonical
    /// fragments of the known templates.
    ///
    /// Returns `Some(template)` if a known template is detected. Returns
    /// `None` if the model has no `chat_template` metadata or the template
    /// doesn't match any known form (in which case the caller should pick
    /// one manually).
    pub fn detect_from_model(model: &GgufModel) -> Option<Self> {
        let template_str = model.metadata_string("tokenizer.chat_template")?;
        // Pattern-match against signature tokens that appear in each
        // template's Jinja source.
        if template_str.contains("<|start_header_id|>")
            || template_str.contains("<|eot_id|>")
        {
            Some(Self::Llama3)
        } else if template_str.contains("<<SYS>>")
            || template_str.contains("[INST]") && template_str.contains("<s>")
                && !template_str.contains("<|im_start|>")
        {
            // Llama 2 has <<SYS>> for system; Mistral uses [INST] but no <<SYS>>.
            if template_str.contains("<<SYS>>") { Some(Self::Llama2) }
            else { Some(Self::Mistral) }
        } else if template_str.contains("<|im_start|>")
            && template_str.contains("<|im_end|>")
        {
            Some(Self::ChatML)
        } else if template_str.contains("<|user|>")
            && template_str.contains("<|assistant|>")
        {
            Some(Self::Phi3)
        } else {
            None
        }
    }

    /// Format a sequence of chat messages into a prompt string ready for
    /// tokenization. The result includes the trailing "assistant turn
    /// open" marker so the model continues from there.
    pub fn format(&self, messages: &[ChatMessage]) -> String {
        let mut out = String::new();
        match self {
            Self::Llama3 => {
                out.push_str("<|begin_of_text|>");
                for m in messages {
                    let role = match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    };
                    out.push_str(&format!(
                        "<|start_header_id|>{}<|end_header_id|>\n\n{}<|eot_id|>",
                        role, m.content
                    ));
                }
                out.push_str("<|start_header_id|>assistant<|end_header_id|>\n\n");
            }
            Self::Llama2 => {
                let system = messages.iter()
                    .find(|m| m.role == ChatRole::System)
                    .map(|m| m.content.as_str());
                out.push_str("<s>[INST] ");
                if let Some(sys) = system {
                    out.push_str(&format!("<<SYS>>\n{}\n<</SYS>>\n\n", sys));
                }
                let mut first_user = true;
                for m in messages {
                    match m.role {
                        ChatRole::System => continue,
                        ChatRole::User => {
                            if first_user {
                                out.push_str(&m.content);
                                out.push_str(" [/INST]");
                                first_user = false;
                            } else {
                                out.push_str("<s>[INST] ");
                                out.push_str(&m.content);
                                out.push_str(" [/INST]");
                            }
                        }
                        ChatRole::Assistant => {
                            out.push(' ');
                            out.push_str(&m.content);
                            out.push_str(" </s>");
                        }
                    }
                }
            }
            Self::Mistral => {
                // Mistral has no native system role; we inline it into the
                // first user message.
                let mut pending_system: Option<String> = None;
                let mut first_user = true;
                for m in messages {
                    match m.role {
                        ChatRole::System => {
                            pending_system = Some(m.content.clone());
                        }
                        ChatRole::User => {
                            if first_user {
                                out.push_str("<s>[INST] ");
                                first_user = false;
                            } else {
                                out.push_str("[INST] ");
                            }
                            if let Some(sys) = pending_system.take() {
                                out.push_str(&sys);
                                out.push_str("\n\n");
                            }
                            out.push_str(&m.content);
                            out.push_str(" [/INST]");
                        }
                        ChatRole::Assistant => {
                            out.push(' ');
                            out.push_str(&m.content);
                            out.push_str("</s>");
                        }
                    }
                }
            }
            Self::ChatML => {
                for m in messages {
                    let role = match m.role {
                        ChatRole::System => "system",
                        ChatRole::User => "user",
                        ChatRole::Assistant => "assistant",
                    };
                    out.push_str(&format!(
                        "<|im_start|>{}\n{}<|im_end|>\n", role, m.content));
                }
                out.push_str("<|im_start|>assistant\n");
            }
            Self::Phi3 => {
                for m in messages {
                    let tag = match m.role {
                        ChatRole::System => "<|system|>",
                        ChatRole::User => "<|user|>",
                        ChatRole::Assistant => "<|assistant|>",
                    };
                    out.push_str(&format!("{}\n{}<|end|>\n", tag, m.content));
                }
                out.push_str("<|assistant|>\n");
            }
            Self::Plain => {
                for m in messages {
                    let prefix = match m.role {
                        ChatRole::System => "System: ",
                        ChatRole::User => "User: ",
                        ChatRole::Assistant => "Assistant: ",
                    };
                    out.push_str(prefix);
                    out.push_str(&m.content);
                    out.push('\n');
                }
                out.push_str("Assistant: ");
            }
        }
        out
    }

    /// Human-readable name.
    pub fn name(&self) -> &'static str {
        match self {
            Self::Llama3 => "Llama 3",
            Self::Llama2 => "Llama 2",
            Self::Mistral => "Mistral",
            Self::ChatML => "ChatML",
            Self::Phi3 => "Phi-3",
            Self::Plain => "Plain",
        }
    }

    /// Literal special-token strings this template emits. Vocabs that
    /// expose these as atomic IDs (e.g., LFM2's ChatML `<|im_start|>` = 6)
    /// must reconstruct them via `encode_with_specials` rather than plain
    /// BPE, which would shatter them into byte-level subtokens.
    pub fn specials(&self) -> &'static [&'static str] {
        match self {
            Self::Llama3 => &[
                "<|begin_of_text|>",
                "<|start_header_id|>",
                "<|end_header_id|>",
                "<|eot_id|>",
            ],
            Self::ChatML => &["<|im_start|>", "<|im_end|>"],
            Self::Phi3 => &["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"],
            Self::Llama2 | Self::Mistral | Self::Plain => &[],
        }
    }
}

/// Encode `text` into token IDs, treating each entry of `specials` as an
/// atomic token (looked up by exact match in `vocab.token_to_id`).
///
/// Use this when the prompt contains chat-template markers (`<|im_start|>`,
/// `<|user|>`, etc.) that the vocab exposes as single tokens. Plain BPE
/// would split them into subwords; this scans for occurrences of those
/// literal strings, emits the atomic ID at each match, and BPE-encodes
/// the surrounding segments.
///
/// Longest-first matching avoids prefix conflicts (e.g., `<|end_header_id|>`
/// is matched before `<|end|>`). Specials not present in the vocab fall
/// through to BPE — harmless, since the model will see the same bytes.
pub fn encode_with_specials(
    vocab: &Vocab,
    text: &str,
    specials: &[&str],
    add_bos: bool,
) -> Vec<u32> {
    // SPM models (Llama 1/2 family) replace spaces with U+2581 internally;
    // BPE models (Llama 3, Qwen3, LFM2) use the GPT-2 byte→Unicode map.
    // Picking the wrong segment encoder produces missing-space garbage
    // ("TheGQofGFrance" vs "The capital of France"), so dispatch on the
    // vocab's declared `tokenizer.ggml.model`.
    let encode_segment = |seg: &str| -> Vec<u32> {
        match vocab.model_name.as_str() {
            "llama" => vocab.encode_spm(seg, false),
            _ => vocab.encode_bpe_regex(seg, false),
        }
    };

    let mut atomic: Vec<(&str, u32)> = specials
        .iter()
        .filter_map(|s| vocab.token_to_id.get(*s).map(|&id| (*s, id)))
        .collect();
    atomic.sort_by_key(|(s, _)| std::cmp::Reverse(s.len()));

    let mut out = Vec::new();
    if add_bos {
        if let Some(id) = vocab.bos_id {
            out.push(id);
        }
    }

    if atomic.is_empty() {
        out.extend(encode_segment(text));
        return out;
    }

    let bytes = text.as_bytes();
    let mut i = 0;
    let mut segment_start = 0;
    while i < bytes.len() {
        let mut matched: Option<(&str, u32)> = None;
        for &(s, id) in &atomic {
            let sb = s.as_bytes();
            if i + sb.len() <= bytes.len() && &bytes[i..i + sb.len()] == sb {
                matched = Some((s, id));
                break;
            }
        }
        if let Some((s, id)) = matched {
            if i > segment_start {
                let seg = &text[segment_start..i];
                out.extend(encode_segment(seg));
            }
            out.push(id);
            i += s.len();
            segment_start = i;
        } else {
            i += 1;
        }
    }
    if segment_start < bytes.len() {
        let seg = &text[segment_start..];
        out.extend(encode_segment(seg));
    }
    out
}

/// Wrap a single user message in `template`, encode the result to tokens,
/// and return the token IDs ready for `Conversation::extend_tokens`. The
/// trailing "assistant turn open" marker is included so the model
/// continues from there.
pub fn encode_user_turn(
    template: ChatTemplate,
    user_text: &str,
    vocab: &Vocab,
    add_bos: bool,
) -> Vec<u32> {
    let prompt = template.format(&[ChatMessage::user(user_text)]);
    encode_with_specials(vocab, &prompt, template.specials(), add_bos)
}
