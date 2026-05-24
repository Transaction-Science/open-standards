//! Native-sink E2E: decode every shipped `.wai` neural sample through
//! the Rust `ort`-backed runtime. Asserts the decoder produces sane
//! output (right sample count for audio, right pixel grid for image,
//! right frame count for video, and a non-trivial signal in each).
//!
//! Models are expected at `wai-web/demo/models/<name>/decoder.onnx` —
//! built via the corresponding `tools/wai_<name>_export_onnx.py` once
//! per machine (see wai-web/demo/models/README.md).
//!
//! Skipped (printed, not failed) when a given .onnx isn't present yet,
//! so a fresh clone with only some models built still runs the tests
//! that can run.

#![cfg(feature = "neural")]

use std::path::{Path, PathBuf};

use wai::container::Wai;
use wai::neural::{decode_envelope, Decoded, ModelRegistry};

fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("..").canonicalize().unwrap()
}

fn model_path(name: &str) -> PathBuf {
    repo_root().join(format!("wai-web/demo/models/{name}/decoder.onnx"))
}

fn sample_path(name: &str) -> PathBuf {
    repo_root().join(format!("wai-web/demo/samples/{name}"))
}

fn load_envelope(name: &str) -> Wai {
    let bytes = std::fs::read(sample_path(name))
        .unwrap_or_else(|e| panic!("read {name}: {e}"));
    Wai::from_bytes(&bytes).unwrap_or_else(|e| panic!("parse {name}: {e}"))
}

fn registry() -> ModelRegistry {
    ModelRegistry::new()
        .register("wai.neural.encodec32",         model_path("encodec_32khz"))
        .register("wai.neural.dac",               model_path("dac_44khz"))
        .register("wai.neural.mimi",              model_path("mimi"))
        .register("wai.neural.wavtokenizer",      model_path("wavtokenizer"))
        .register("wai.neural.bmshj2018",         model_path("bmshj2018"))
        .register("wai.neural.video_bmshj2018",   model_path("bmshj2018"))   // shares the image decoder
}

fn assert_audio_sane(cap: &str, sample_file: &str, expected_sr: u32, min_samples: usize) {
    let onnx = model_path(match cap {
        "wai.neural.encodec32"    => "encodec_32khz",
        "wai.neural.dac"          => "dac_44khz",
        "wai.neural.mimi"         => "mimi",
        "wai.neural.wavtokenizer" => "wavtokenizer",
        _ => unreachable!(),
    });
    if !onnx.exists() {
        eprintln!("skipping {cap}: {} not built (see tools/wai_<name>_export_onnx.py)",
                  onnx.display());
        return;
    }
    let env = load_envelope(sample_file);
    assert_eq!(env.manifest.model_requirement.capability, cap,
               "{sample_file} declares wrong capability");
    let dec = decode_envelope(&env, &registry())
        .unwrap_or_else(|e| panic!("decode {sample_file}: {e}"));
    match dec {
        Decoded::Audio(audio) => {
            assert_eq!(audio.sample_rate, expected_sr,
                       "{cap}: wrong sample rate");
            assert!(audio.samples.len() >= min_samples,
                    "{cap}: only {} samples, expected ≥ {}",
                    audio.samples.len(), min_samples);
            // Non-trivial signal: peak amplitude well above 0.
            let peak = audio.samples.iter().fold(0f32, |a, &b| a.max(b.abs()));
            assert!(peak > 0.01, "{cap}: peak={peak:.4} (silent decode?)");
            // No NaNs.
            assert!(audio.samples.iter().all(|s| s.is_finite()),
                    "{cap}: NaN/Inf in output");
            println!("  {cap}: {} samples @ {} Hz, peak {:.3}",
                     audio.samples.len(), audio.sample_rate, peak);
        }
        _ => panic!("{cap}: expected Decoded::Audio"),
    }
}

#[test] fn encodec32_decodes_natively() {
    assert_audio_sane("wai.neural.encodec32",  "glass.encodec.wai",       32_000, 16_000);
}
#[test] fn dac_decodes_natively() {
    assert_audio_sane("wai.neural.dac",        "glass.dac.wai",           44_100, 22_000);
}
#[test] fn mimi_decodes_natively() {
    assert_audio_sane("wai.neural.mimi",       "glass.mimi.wai",          24_000, 12_000);
}
#[test] fn wavtokenizer_decodes_natively() {
    assert_audio_sane("wai.neural.wavtokenizer", "glass.wavtokenizer.wai", 24_000, 12_000);
}

#[test]
fn bmshj2018_decodes_natively() {
    let onnx = model_path("bmshj2018");
    if !onnx.exists() {
        eprintln!("skipping bmshj2018: {} not built", onnx.display());
        return;
    }
    let env = load_envelope("kodim23.bmshj2018.wai");
    let dec = decode_envelope(&env, &registry()).expect("bmshj2018 decode");
    match dec {
        Decoded::Image(img) => {
            assert_eq!(img.width, 256);
            assert_eq!(img.height, 256);
            assert_eq!(img.rgb.len(), 256 * 256 * 3);
            // Variance: a real image has lots of pixel variation. A blank
            // decode would be near-constant.
            let mean = img.rgb.iter().map(|&v| v as f64).sum::<f64>() / img.rgb.len() as f64;
            let var  = img.rgb.iter()
                .map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / img.rgb.len() as f64;
            assert!(var > 100.0, "bmshj2018: variance {var:.0} suspiciously low");
            println!("  bmshj2018: 256x256 RGB, mean={mean:.1}, var={var:.0}");
        }
        _ => panic!("bmshj2018: expected Decoded::Image"),
    }
}

#[test]
fn video_bmshj2018_decodes_natively() {
    let onnx = model_path("bmshj2018");
    if !onnx.exists() {
        eprintln!("skipping video_bmshj2018: {} not built", onnx.display());
        return;
    }
    let env = load_envelope("test.video.wai");
    let dec = decode_envelope(&env, &registry()).expect("video decode");
    match dec {
        Decoded::Video(v) => {
            assert!(v.frames_rgb.len() > 0, "no frames decoded");
            assert_eq!(v.width, 256);
            assert_eq!(v.height, 256);
            assert!((v.fps - 12.0).abs() < 0.01, "fps {} != 12", v.fps);
            // First frame should have non-trivial variance.
            let f0 = &v.frames_rgb[0];
            let mean = f0.iter().map(|&v| v as f64).sum::<f64>() / f0.len() as f64;
            let var  = f0.iter().map(|&v| (v as f64 - mean).powi(2)).sum::<f64>() / f0.len() as f64;
            assert!(var > 100.0, "video frame 0 variance {var:.0} suspiciously low");
            println!("  video_bmshj2018: {} frames {}x{} @ {} fps",
                     v.frames_rgb.len(), v.width, v.height, v.fps);
        }
        _ => panic!("video_bmshj2018: expected Decoded::Video"),
    }
}
