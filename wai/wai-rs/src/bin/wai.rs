//! WAI CLI — encode/decode through registered zeroth-condition codecs
//! (AVIF/JPEG-XL/PNG/Opus/FLAC/AV1/zstd/XZ). Designed for scripting and
//! quick verification; production paths go through the Rust API or the
//! C FFI.

use std::env;
use std::fs;
use std::process::ExitCode;

use wai::codecs::{audio, image, text, video};
use wai::{Conditioning, Manifest, ModelRequirement, Wai};
use wai::*;

fn die(msg: impl std::fmt::Display) -> ExitCode {
    eprintln!("error: {msg}");
    ExitCode::from(2)
}

fn usage() {
    eprintln!("wai encode  <codec> <input> [-o out.wai] [-q quality]");
    eprintln!("wai decode  <in.wai> -o <output>");
    eprintln!("wai inspect <in.wai>");
    eprintln!();
    eprintln!("codecs: png jpeg avif jxl jxl-lossless opus flac av1 av1-lossless zstd xz");
    eprintln!("inputs:");
    eprintln!("  image codecs:  .rgb (with .meta containing 'H W' ASCII) or PNG/JPEG file");
    eprintln!("  audio codecs:  .f32 (raw mono float32) with .meta containing sample-rate");
    eprintln!("  video codecs:  .rgb-seq directory of frame files + .meta with 'N H W fps_num fps_den'");
    eprintln!("  text codecs:   any file");
}

fn opt<'a>(args: &'a [String], flag: &str) -> Option<&'a str> {
    args.iter().position(|a| a == flag)
        .and_then(|i| args.get(i + 1).map(|s| s.as_str()))
}

fn read_meta(path: &str) -> Result<Vec<u32>, String> {
    let s = fs::read_to_string(format!("{path}.meta"))
        .map_err(|e| format!("missing {path}.meta: {e}"))?;
    s.split_ascii_whitespace().map(|t| t.parse::<u32>().map_err(|e| e.to_string()))
        .collect()
}

fn encode(codec: &str, input: &str, out: &str, q: u8) -> Result<(), String> {
    let (cap, kind, payload): (&str, &str, Vec<u8>) = match codec {
        "png" => {
            let meta = read_meta(input)?;
            let (h, w) = (meta[0], meta[1]);
            let rgb = fs::read(input).map_err(|e| e.to_string())?;
            let p = image::png_encode(&rgb, h, w).map_err(|e| format!("{:?}", e))?;
            (CAP_IMAGE_PNG, "png", p)
        }
        "jpeg" => {
            let meta = read_meta(input)?;
            let (h, w) = (meta[0], meta[1]);
            let rgb = fs::read(input).map_err(|e| e.to_string())?;
            let p = image::jpeg_encode(&rgb, h, w, q).map_err(|e| format!("{:?}", e))?;
            (CAP_IMAGE_JPEG, "jpeg", p)
        }
        "avif" => {
            let meta = read_meta(input)?;
            let (h, w) = (meta[0], meta[1]);
            let rgb = fs::read(input).map_err(|e| e.to_string())?;
            let p = image::avif_encode(&rgb, h, w, q).map_err(|e| format!("{:?}", e))?;
            (CAP_IMAGE_AVIF, "avif", p)
        }
        "jxl-lossless" => {
            let meta = read_meta(input)?;
            let (h, w) = (meta[0], meta[1]);
            let rgb = fs::read(input).map_err(|e| e.to_string())?;
            let p = image::jxl_encode(&rgb, h, w, None).map_err(|e| format!("{:?}", e))?;
            (CAP_IMAGE_JXL, "jxl", p)
        }
        "jxl" => {
            let meta = read_meta(input)?;
            let (h, w) = (meta[0], meta[1]);
            let rgb = fs::read(input).map_err(|e| e.to_string())?;
            let p = image::jxl_encode(&rgb, h, w, Some(q as f32))
                .map_err(|e| format!("{:?}", e))?;
            (CAP_IMAGE_JXL, "jxl", p)
        }
        "opus" => {
            let meta = read_meta(input)?;
            let sr = meta[0];
            let bytes = fs::read(input).map_err(|e| e.to_string())?;
            let samples: Vec<f32> = bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let kbps = (q as i32 * 4 + 32) * 1000;  // q∈[1..100] → ~36..432 kbps
            let p = audio::opus_encode(&samples, sr, kbps)
                .map_err(|e| format!("{:?}", e))?;
            (CAP_AUDIO_OPUS, "opus", p)
        }
        "flac" => {
            let meta = read_meta(input)?;
            let sr = meta[0];
            let bytes = fs::read(input).map_err(|e| e.to_string())?;
            let samples: Vec<f32> = bytes.chunks_exact(4)
                .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
                .collect();
            let p = audio::flac_encode(&samples, sr)
                .map_err(|e| format!("{:?}", e))?;
            (CAP_AUDIO_FLAC, "flac", p)
        }
        "zstd" => {
            let data = fs::read(input).map_err(|e| e.to_string())?;
            let p = text::zstd_encode(&data, (q as i32).clamp(1, 22))
                .map_err(|e| format!("{:?}", e))?;
            (CAP_TEXT_ZSTD, "zstd", p)
        }
        "xz" => {
            let data = fs::read(input).map_err(|e| e.to_string())?;
            let p = text::xz_encode(&data, (q as u32).clamp(0, 9))
                .map_err(|e| format!("{:?}", e))?;
            (CAP_TEXT_XZ, "xz", p)
        }
        "av1" | "av1-lossless" => {
            // Video: input is a packed-RGB blob with .meta containing
            //   "N H W fps_num fps_den"
            let meta = read_meta(input)?;
            let (n, h, w, fps_num, fps_den) = (meta[0], meta[1], meta[2],
                                               meta[3], meta[4]);
            let raw = fs::read(input).map_err(|e| e.to_string())?;
            let per_frame = (h * w * 3) as usize;
            if raw.len() != (n as usize) * per_frame {
                return Err(format!("video raw len {} != n*h*w*3 = {}",
                                    raw.len(), (n as usize) * per_frame));
            }
            let frames: Vec<Vec<u8>> = (0..n as usize)
                .map(|i| raw[i * per_frame..(i + 1) * per_frame].to_vec())
                .collect();
            let lossless = codec == "av1-lossless";
            let p = video::av1_encode(&frames, h, w, fps_num, fps_den,
                                      lossless, q)
                .map_err(|e| format!("{:?}", e))?;
            let cap = if lossless { CAP_VIDEO_AV1_LOSSLESS } else { CAP_VIDEO_AV1 };
            (cap, "av1", p)
        }
        c => return Err(format!("unknown codec '{c}'")),
    };

    let media = match cap {
        c if c.starts_with("wai.image.") => "image",
        c if c.starts_with("wai.audio.") => "audio",
        c if c.starts_with("wai.video.") => "video",
        c if c.starts_with("wai.text.") => "text",
        _ => "binary",
    };
    let manifest = Manifest {
        wai: "1.0".into(),
        media: media.into(),
        intent: "replicate".into(),
        model_requirement: ModelRequirement {
            capability: cap.into(),
            fallback: None,
        },
        conditioning: Conditioning { kind: kind.into() },
        target: serde_json::Value::Null,
    };
    let n = Wai::new(manifest, payload).write(out).map_err(|e| e.to_string())?;
    println!("encoded → {out}  ({} bytes, capability {})", n, cap);
    Ok(())
}

fn decode(input: &str, out: &str) -> Result<(), String> {
    let wai = Wai::read(input).map_err(|e| e.to_string())?;
    let cap = wai.capability();
    match cap {
        c if c == CAP_IMAGE_PNG => {
            let (rgb, h, w) = image::png_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            fs::write(out, &rgb).map_err(|e| e.to_string())?;
            fs::write(format!("{out}.meta"), format!("{h} {w}\n")).map_err(|e| e.to_string())?;
        }
        c if c == CAP_IMAGE_JPEG => {
            let (rgb, h, w) = image::jpeg_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            fs::write(out, &rgb).map_err(|e| e.to_string())?;
            fs::write(format!("{out}.meta"), format!("{h} {w}\n")).map_err(|e| e.to_string())?;
        }
        c if c == CAP_IMAGE_AVIF => {
            let (rgb, h, w) = image::avif_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            fs::write(out, &rgb).map_err(|e| e.to_string())?;
            fs::write(format!("{out}.meta"), format!("{h} {w}\n")).map_err(|e| e.to_string())?;
        }
        c if c == CAP_IMAGE_JXL => {
            let (rgb, h, w) = image::jxl_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            fs::write(out, &rgb).map_err(|e| e.to_string())?;
            fs::write(format!("{out}.meta"), format!("{h} {w}\n")).map_err(|e| e.to_string())?;
        }
        c if c == CAP_AUDIO_OPUS => {
            let (samples, sr) = audio::opus_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            let bytes: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
            fs::write(out, &bytes).map_err(|e| e.to_string())?;
            fs::write(format!("{out}.meta"), format!("{sr}\n")).map_err(|e| e.to_string())?;
        }
        c if c == CAP_AUDIO_FLAC => {
            let (samples, sr) = audio::flac_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            let bytes: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
            fs::write(out, &bytes).map_err(|e| e.to_string())?;
            fs::write(format!("{out}.meta"), format!("{sr}\n")).map_err(|e| e.to_string())?;
        }
        c if c == CAP_TEXT_ZSTD => {
            let d = text::zstd_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            fs::write(out, &d).map_err(|e| e.to_string())?;
        }
        c if c == CAP_TEXT_XZ => {
            let d = text::xz_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            fs::write(out, &d).map_err(|e| e.to_string())?;
        }
        c if c == CAP_VIDEO_AV1 || c == CAP_VIDEO_AV1_LOSSLESS => {
            let (frames, h, w, fps_num, fps_den) =
                video::av1_decode(&wai.payload).map_err(|e| format!("{:?}", e))?;
            let n = frames.len() as u32;
            let mut packed = Vec::with_capacity(frames.len() * (h * w * 3) as usize);
            for f in &frames { packed.extend_from_slice(f); }
            fs::write(out, &packed).map_err(|e| e.to_string())?;
            fs::write(format!("{out}.meta"),
                      format!("{n} {h} {w} {fps_num} {fps_den}\n"))
                .map_err(|e| e.to_string())?;
        }
        other => return Err(format!("unsupported capability for CLI decode: {other}")),
    }
    println!("decoded → {out}");
    Ok(())
}

fn inspect(path: &str) -> Result<(), String> {
    let bytes = std::fs::read(path).map_err(|e| e.to_string())?;
    let version = wai::container::detect_version(&bytes).map_err(|e| e.to_string())?;
    let known: std::collections::HashSet<&str> =
        wai::sink_capabilities().iter().copied().collect();
    let size = bytes.len();
    println!("wai inspect — {path}");
    println!("  envelope size:     {size:>10} B");
    println!("  format version:    {}",
             if version == 1 { "WAI1 (single payload)" }
             else { "WAI2 (multi-rendition)" });

    if version == 1 {
        let w = wai::container::Wai::from_bytes(&bytes).map_err(|e| e.to_string())?;
        let cap = w.capability();
        let fb = w.fallback();
        let cap_ok = known.contains(cap);
        let fb_ok = fb.map(|f| known.contains(f));
        println!("  payload size:      {:>10} B", w.payload.len());
        println!("  media:             {}", w.media());
        println!("  intent:            {}", w.intent());
        println!("  conditioning.kind: {}", w.kind());
        println!();
        println!("  capability:        {cap}");
        println!("    resolution:      {}",
                 if cap_ok { "\u{2713} this sink advertises it" }
                 else { "\u{2717} NOT installed at this sink" });
        if let Some(f) = fb {
            println!("  fallback:          {f}");
            println!("    resolution:      {}",
                     if fb_ok.unwrap() { "\u{2713} this sink advertises it" }
                     else { "\u{2717} NOT installed either" });
        }
        let verdict = match (cap_ok, fb_ok) {
            (true,  _)           => "DECODABLE via primary capability",
            (false, Some(true))  => "DECODABLE via declared fallback (informational only in v1.0)",
            (false, Some(false)) => "INERT here \u{2014} neither capability installed",
            (false, None)        => "INERT here \u{2014} capability missing, no fallback declared",
        };
        println!("  verdict:           {verdict}");
    } else {
        // WAI2 inspection
        let m = wai::container::WaiMulti::from_bytes(&bytes).map_err(|e| e.to_string())?;
        println!("  media:             {}", m.media());
        println!("  intent:            {}", m.intent());
        println!("  renditions:        {}", m.renditions.len());
        println!();
        println!("  rendition table (deployer-preferred order):");
        println!("    {:>3} {:>10}  {:<28} {}", "idx", "bytes", "capability", "kind");
        let mut picked = None;
        for (i, (meta, payload)) in
            m.manifest.renditions.iter().zip(m.renditions.iter()).enumerate()
        {
            let avail = known.contains(meta.capability.as_str());
            let mark = if avail { "\u{2713}" } else { "\u{2717}" };
            if avail && picked.is_none() { picked = Some(i); }
            println!("    [{i}] {:>10}  {:<28} {}  {}",
                     payload.len(), meta.capability, meta.kind, mark);
        }
        println!();
        match picked {
            Some(i) => println!("  default pick:      [{i}] {} \u{2014} first rendition this sink can decode",
                                m.manifest.renditions[i].capability),
            None => println!("  default pick:      none \u{2014} no rendition's capability installed here"),
        }
    }

    println!();
    println!("  this sink advertises {} capabilities:", known.len());
    let mut ks: Vec<&&str> = known.iter().collect(); ks.sort();
    for k in ks { println!("    - {k}"); }
    Ok(())
}

fn main() -> ExitCode {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.is_empty() {
        usage();
        return ExitCode::from(2);
    }
    let r = match args[0].as_str() {
        "encode" if args.len() >= 3 => {
            let out = opt(&args[3..], "-o").map(String::from).unwrap_or("out.wai".into());
            let q: u8 = opt(&args[3..], "-q").and_then(|s| s.parse().ok()).unwrap_or(75);
            encode(&args[1], &args[2], &out, q)
        }
        "decode" if args.len() >= 4 => decode(&args[1], &args[3]),
        "inspect" if args.len() >= 2 => inspect(&args[1]),
        _ => { usage(); return ExitCode::from(2); }
    };
    match r {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => die(e),
    }
}
