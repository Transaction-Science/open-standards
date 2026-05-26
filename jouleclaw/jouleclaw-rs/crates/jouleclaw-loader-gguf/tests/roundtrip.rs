//! Roundtrip test for the GGUF parser.
//!
//! We synthesize a minimal GGUF byte stream (one F32 tensor + a few metadata
//! fields), parse it back, and verify everything matches.

use jouleclaw_loader_gguf::{
    read_gguf, tensor_from_gguf, GgmlType, GgufValue, ParseError,
};
use std::io::Cursor;

const GGUF_MAGIC: u32 = 0x46554747;

/// Builder for a tiny GGUF-encoded buffer.
fn build_gguf(
    metadata: &[(&str, GgufValue)],
    tensors: &[(&str, &[u64], GgmlType, Vec<u8>)],
) -> Vec<u8> {
    let mut out = Vec::new();

    out.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    out.extend_from_slice(&3u32.to_le_bytes()); // version
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    out.extend_from_slice(&(metadata.len() as u64).to_le_bytes());

    for (k, v) in metadata {
        write_string(&mut out, k);
        write_value(&mut out, v);
    }

    // Tensor info table. Compute offsets after pre-computing the data
    // section layout, taking alignment into account.
    let mut tensor_offsets: Vec<u64> = Vec::with_capacity(tensors.len());
    let mut running: u64 = 0;
    for (_, _, _, data) in tensors {
        tensor_offsets.push(running);
        running += data.len() as u64;
    }

    for ((name, shape, dtype, _), offset) in tensors.iter().zip(tensor_offsets.iter()) {
        write_string(&mut out, name);
        out.extend_from_slice(&(shape.len() as u32).to_le_bytes());
        for d in *shape { out.extend_from_slice(&d.to_le_bytes()); }
        out.extend_from_slice(&(*dtype as u32).to_le_bytes());
        out.extend_from_slice(&offset.to_le_bytes());
    }

    // Align to 32 (default GGUF alignment).
    let pos = out.len();
    let aligned = (pos + 31) & !31;
    while out.len() < aligned { out.push(0); }

    // Tensor data section.
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
        _ => unimplemented!("test helper covers only the cases we use"),
    }
}

#[test]
fn parses_minimal_gguf_with_one_f32_tensor() {
    // Build a 2x3 F32 tensor: [[1,2,3],[4,5,6]]
    let data: Vec<u8> = [1.0_f32, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter().flat_map(|v| v.to_le_bytes()).collect();

    let bytes = build_gguf(
        &[
            ("general.architecture", GgufValue::String("llama".into())),
            ("general.alignment", GgufValue::U32(32)),
            ("llama.embedding_length", GgufValue::U32(16)),
            ("llama.block_count", GgufValue::U32(2)),
        ],
        &[("test.weight", &[2, 3], GgmlType::F32, data)],
    );

    let model = read_gguf(Cursor::new(bytes)).expect("parse succeeds");

    assert_eq!(model.version, 3);
    assert_eq!(model.metadata_string("general.architecture"), Some("llama"));
    assert_eq!(model.metadata_u32("llama.embedding_length"), Some(16));
    assert_eq!(model.metadata_u32("llama.block_count"), Some(2));
    assert_eq!(model.tensors.len(), 1);

    let info = &model.tensors[0];
    assert_eq!(info.name, "test.weight");
    assert_eq!(info.shape, vec![2, 3]);
    assert_eq!(info.dtype, GgmlType::F32);

    // Convert and verify values. On-disk dims `[2, 3]` are GGML `ne`
    // order (fastest axis first); the logical shape is the reverse,
    // `[3, 2]`, matching how real GGUF weights are consumed by the
    // graph ops. Flat row-major data is unchanged.
    let tensor = tensor_from_gguf(&model, info).expect("tensor extraction");
    assert_eq!(tensor.meta.shape, vec![3, 2]);
    let v = tensor.as_f32_vec();
    assert_eq!(v, vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
}

#[test]
fn rejects_bad_magic() {
    let mut bytes = vec![0u8; 32];
    bytes[..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let err = read_gguf(Cursor::new(bytes)).err();
    assert!(matches!(err, Some(ParseError::BadMagic { .. })));
}

#[test]
fn handles_array_metadata() {
    let mut buf = Vec::new();
    buf.extend_from_slice(&GGUF_MAGIC.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&0u64.to_le_bytes());  // 0 tensors
    buf.extend_from_slice(&1u64.to_le_bytes());  // 1 metadata entry
    write_string(&mut buf, "tokenizer.ggml.tokens");
    buf.extend_from_slice(&9u32.to_le_bytes());  // type = Array
    buf.extend_from_slice(&8u32.to_le_bytes());  // element type = String
    buf.extend_from_slice(&3u64.to_le_bytes());  // length 3
    write_string(&mut buf, "<unk>");
    write_string(&mut buf, "<bos>");
    write_string(&mut buf, "<eos>");

    let model = read_gguf(Cursor::new(buf)).expect("parse");
    let arr = model.metadata.get("tokenizer.ggml.tokens").unwrap()
        .as_string_array().expect("array of strings");
    assert_eq!(arr, vec!["<unk>", "<bos>", "<eos>"]);
}
