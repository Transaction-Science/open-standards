//! Tests for chat templates: unit tests for each template's format, plus
//! a GGUF round-trip showing detect_from_model() works.

use jouleclaw_loader_gguf::read_gguf;
use jouleclaw_loader_gguf::synthetic::{synthesize_llama_gguf, SyntheticConfig};
use jouleclaw_runtime::{ChatMessage, ChatRole, ChatTemplate};
use std::io::Cursor;

fn msgs() -> Vec<ChatMessage> {
    vec![
        ChatMessage::system("You are a helpful assistant."),
        ChatMessage::user("What's 2+2?"),
    ]
}

#[test]
fn chatml_format_basic() {
    let out = ChatTemplate::ChatML.format(&msgs());
    assert!(out.contains("<|im_start|>system\nYou are a helpful assistant.<|im_end|>"));
    assert!(out.contains("<|im_start|>user\nWhat's 2+2?<|im_end|>"));
    assert!(out.ends_with("<|im_start|>assistant\n"),
        "ChatML should end with the open assistant turn; got:\n{:?}", out);
}

#[test]
fn llama3_format_has_correct_markers() {
    let out = ChatTemplate::Llama3.format(&msgs());
    assert!(out.starts_with("<|begin_of_text|>"));
    assert!(out.contains("<|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|>"));
    assert!(out.contains("<|start_header_id|>user<|end_header_id|>\n\nWhat's 2+2?<|eot_id|>"));
    assert!(out.ends_with("<|start_header_id|>assistant<|end_header_id|>\n\n"));
}

#[test]
fn llama2_format_has_sys_block() {
    let out = ChatTemplate::Llama2.format(&msgs());
    assert!(out.contains("<<SYS>>\nYou are a helpful assistant.\n<</SYS>>"));
    assert!(out.contains("[INST]"));
    assert!(out.contains("[/INST]"));
    assert!(out.contains("What's 2+2?"));
}

#[test]
fn mistral_format_inlines_system_into_first_user() {
    let out = ChatTemplate::Mistral.format(&msgs());
    assert!(out.starts_with("<s>[INST] "));
    assert!(out.contains("You are a helpful assistant.\n\nWhat's 2+2?"),
        "Mistral inlines system into the first user message; got:\n{:?}", out);
    assert!(out.contains("[/INST]"));
}

#[test]
fn phi3_format_uses_role_tags() {
    let out = ChatTemplate::Phi3.format(&msgs());
    assert!(out.contains("<|system|>\nYou are a helpful assistant.<|end|>"));
    assert!(out.contains("<|user|>\nWhat's 2+2?<|end|>"));
    assert!(out.ends_with("<|assistant|>\n"));
}

#[test]
fn plain_format_is_readable() {
    let out = ChatTemplate::Plain.format(&msgs());
    assert!(out.contains("System: You are a helpful assistant."));
    assert!(out.contains("User: What's 2+2?"));
    assert!(out.ends_with("Assistant: "));
}

#[test]
fn multi_turn_chatml_round_trip() {
    let conv = vec![
        ChatMessage::system("Be brief."),
        ChatMessage::user("Hi."),
        ChatMessage::assistant("Hello!"),
        ChatMessage::user("Who are you?"),
    ];
    let out = ChatTemplate::ChatML.format(&conv);
    // All four turns present, plus open assistant.
    assert!(out.contains("<|im_start|>system\nBe brief.<|im_end|>"));
    assert!(out.contains("<|im_start|>user\nHi.<|im_end|>"));
    assert!(out.contains("<|im_start|>assistant\nHello!<|im_end|>"));
    assert!(out.contains("<|im_start|>user\nWho are you?<|im_end|>"));
    assert!(out.ends_with("<|im_start|>assistant\n"));
}

fn build_model_with_template(template: Option<&str>) -> jouleclaw_loader_gguf::GgufModel {
    let mut vocab = vec![
        ("<unk>".to_string(), 0.0),
        ("<s>".to_string(), 0.0),
        ("</s>".to_string(), 0.0),
    ];
    for c in 'a'..='z' { vocab.push((c.to_string(), -1.0)); }
    let cfg = SyntheticConfig {
        vocab_size: vocab.len(),
        embedding_length: 8,
        block_count: 1,
        feed_forward_length: 16,
        head_count: 1,
        head_count_kv: 1,
        rms_eps: 1e-6,
        seed: 1,
        vocab: Some(vocab),
        merges: None,
        bos_id: Some(1), eos_id: Some(2), unk_id: Some(0),
        chat_template: template.map(|s| s.to_string()),
    };
    let bytes = synthesize_llama_gguf(&cfg);
    read_gguf(Cursor::new(bytes)).unwrap()
}

/// detect_from_model returns None when no chat_template metadata is set.
#[test]
fn detect_returns_none_when_no_template() {
    let model = build_model_with_template(None);
    assert_eq!(ChatTemplate::detect_from_model(&model), None);
}

/// Detect Llama 3 from its signature markers.
#[test]
fn detect_llama3() {
    let template = "{% for message in messages %}<|start_header_id|>{{ message.role }}<|end_header_id|>\n\n{{ message.content }}<|eot_id|>{% endfor %}";
    let model = build_model_with_template(Some(template));
    assert_eq!(ChatTemplate::detect_from_model(&model), Some(ChatTemplate::Llama3));
}

/// Detect Llama 2 from <<SYS>> marker.
#[test]
fn detect_llama2() {
    let template = "<s>[INST] <<SYS>>\n{{ system }}\n<</SYS>>\n\n{{ user }} [/INST]";
    let model = build_model_with_template(Some(template));
    assert_eq!(ChatTemplate::detect_from_model(&model), Some(ChatTemplate::Llama2));
}

/// Detect Mistral from [INST] + <s> without <<SYS>>.
#[test]
fn detect_mistral() {
    let template = "<s>[INST] {{ message.content }} [/INST]";
    let model = build_model_with_template(Some(template));
    assert_eq!(ChatTemplate::detect_from_model(&model), Some(ChatTemplate::Mistral));
}

/// Detect ChatML from <|im_start|>/<|im_end|>.
#[test]
fn detect_chatml() {
    let template = "{% for m in messages %}<|im_start|>{{ m.role }}\n{{ m.content }}<|im_end|>\n{% endfor %}<|im_start|>assistant\n";
    let model = build_model_with_template(Some(template));
    assert_eq!(ChatTemplate::detect_from_model(&model), Some(ChatTemplate::ChatML));
}

/// Detect Phi-3 from <|user|>/<|assistant|> tags.
#[test]
fn detect_phi3() {
    let template = "<|user|>\n{{ user }}<|end|>\n<|assistant|>\n";
    let model = build_model_with_template(Some(template));
    assert_eq!(ChatTemplate::detect_from_model(&model), Some(ChatTemplate::Phi3));
}

/// Unknown template strings produce None — caller handles fallback.
#[test]
fn detect_unknown_template_returns_none() {
    let template = "completely custom format with no known markers";
    let model = build_model_with_template(Some(template));
    assert_eq!(ChatTemplate::detect_from_model(&model), None);
}

/// Demo: format the same conversation in all templates.
#[test]
fn chat_template_demo() {
    let conv = vec![
        ChatMessage::system("Be brief."),
        ChatMessage::user("What is 2+2?"),
    ];

    println!("\n=== Chat template comparison ===");
    println!("Input messages:");
    for m in &conv {
        let role = match m.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
        };
        println!("  [{}]: {:?}", role, m.content);
    }
    println!();

    for tmpl in &[
        ChatTemplate::Llama3,
        ChatTemplate::Llama2,
        ChatTemplate::Mistral,
        ChatTemplate::ChatML,
        ChatTemplate::Phi3,
        ChatTemplate::Plain,
    ] {
        let formatted = tmpl.format(&conv);
        println!("--- {} ---", tmpl.name());
        // Render escaped so newlines and special tokens are visible.
        let escaped = formatted
            .replace('\n', "\\n")
            .replace('\r', "\\r");
        println!("  {}", escaped);
    }
    println!("\nEach template ends with the model's \"open assistant turn\"");
    println!("marker — generate() continues from there.");
}
