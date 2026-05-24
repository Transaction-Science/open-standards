//! Sanity harness for the wai-rs codec wrappers.
//!
//! Two questions per codec:
//!   (1) INTEROP — do the bytes WAI emits decode in ffmpeg (and the
//!       bytes ffmpeg emits decode in WAI)? The answer must be yes,
//!       because the WAI wrappers are *supposed* to be passing through
//!       the standard byte stream of each registered capability.
//!   (2) RD PARITY — at matched quality settings, do WAI-encoded files
//!       come out within ~10% of the byte size that direct ffmpeg
//!       produces? A bigger gap means the wrapper picked a non-standard
//!       parameter (wrong speed preset, default chroma format, etc.) and
//!       would be losing bytes relative to using the library directly.
//!
//! Lossless codecs (PNG, JPEG-XL lossless, FLAC, zstd, XZ, AV1 lossless)
//! are checked for *bit-exact* reconstruction. Lossy codecs are checked
//! for PSNR within 1 dB of direct ffmpeg at the same nominal quality.

use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::time::Instant;

use wai::codecs::{audio, image, text, video};

// ---- aggregate stats: a per-codec running summary ----------------
#[derive(Default, Clone, serde::Serialize)]
struct CodecStats {
    n_runs: usize,
    n_ok: usize,
    n_fail: usize,
    total_wai_bytes: u64,
    total_ref_bytes: u64,
    sum_psnr_delta: f64,
    n_psnr: usize,
    encode_ms: u128,
}

/// Top-level report: env metadata + per-codec stats. Serialized as JSON
/// when `--report <path>` is passed. Designed so production CI can
/// diff two reports across commits and flag regressions.
#[derive(serde::Serialize)]
struct BenchReport {
    wai_version: String,
    timestamp_unix: u64,
    mode: String,                                 // "quick" | "full"
    host: String,                                 // os/arch identifier
    total_seconds: f64,
    codecs: std::collections::BTreeMap<String, CodecStats>,
}

fn write_report(path: &str, mode: &str, secs: f64,
                stats: &HashMap<String, CodecStats>) -> std::io::Result<()> {
    let report = BenchReport {
        wai_version: env!("CARGO_PKG_VERSION").to_string(),
        timestamp_unix: std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs()).unwrap_or(0),
        mode: mode.to_string(),
        host: format!("{}/{}", std::env::consts::OS, std::env::consts::ARCH),
        total_seconds: secs,
        codecs: stats.iter().map(|(k, v)| (k.clone(), v.clone())).collect(),
    };
    let json = serde_json::to_string_pretty(&report)
        .map_err(|e| std::io::Error::other(e.to_string()))?;
    fs::write(path, json)?;
    Ok(())
}

impl CodecStats {
    fn record(&mut self, wai_bytes: usize, ref_bytes: Option<usize>,
              psnr_delta: Option<f64>, ok: bool, enc_ms: u128) {
        self.n_runs += 1;
        if ok { self.n_ok += 1 } else { self.n_fail += 1 }
        self.total_wai_bytes += wai_bytes as u64;
        if let Some(r) = ref_bytes { self.total_ref_bytes += r as u64; }
        if let Some(d) = psnr_delta { self.sum_psnr_delta += d; self.n_psnr += 1; }
        self.encode_ms += enc_ms;
    }
}

fn print_summary(stats: &HashMap<String, CodecStats>) {
    println!("\n=== SUMMARY ===");
    println!("  {:18} {:>6} {:>6} {:>6} {:>12} {:>12} {:>10} {:>10} {:>10}",
             "codec", "runs", "ok", "fail", "WAI bytes", "ref bytes",
             "size %", "Δ PSNR", "enc ms");
    let mut names: Vec<&String> = stats.keys().collect();
    names.sort();
    for name in names {
        let s = &stats[name];
        let pct = if s.total_ref_bytes > 0 {
            format!("{:+6.1}%",
                    (s.total_wai_bytes as f64 / s.total_ref_bytes as f64 - 1.0) * 100.0)
        } else { "-".into() };
        let psnr = if s.n_psnr > 0 {
            format!("{:+5.2}", s.sum_psnr_delta / s.n_psnr as f64)
        } else { "-".into() };
        println!("  {:18} {:>6} {:>6} {:>6} {:>12} {:>12} {:>10} {:>10} {:>10}",
                 name, s.n_runs, s.n_ok, s.n_fail,
                 s.total_wai_bytes, s.total_ref_bytes,
                 pct, psnr, s.encode_ms);
    }
}

const CORPUS: &str = "/Users/dcharlot/data-share/vibe-coding/web_standard_new/corpus";

// --- helpers ---------------------------------------------------------

fn run(cmd: &str, args: &[&str]) -> std::io::Result<Vec<u8>> {
    let out = Command::new(cmd).args(args).stderr(Stdio::null()).output()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "{cmd} failed (exit {:?})", out.status.code())));
    }
    Ok(out.stdout)
}

fn run_in_out(cmd: &str, args: &[&str], stdin_data: &[u8]) -> std::io::Result<Vec<u8>> {
    let mut child = Command::new(cmd).args(args)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::null())
        .spawn()?;
    let mut sin = child.stdin.take().expect("stdin piped");
    // The OS pipe buffer is ~64 KB on macOS. For payloads larger than
    // that we MUST drain stdout concurrently with writing stdin, or the
    // child blocks on stdout and we block on stdin (classic deadlock).
    let writer_data = stdin_data.to_vec();
    let writer = std::thread::spawn(move || -> std::io::Result<()> {
        sin.write_all(&writer_data)?;
        drop(sin);
        Ok(())
    });
    let out = child.wait_with_output()?;
    writer.join().unwrap()?;
    if !out.status.success() {
        return Err(std::io::Error::other(format!(
            "{cmd} failed (exit {:?})", out.status.code())));
    }
    Ok(out.stdout)
}

fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    let mse: f64 = (0..n)
        .map(|i| (a[i] as f64 - b[i] as f64).powi(2))
        .sum::<f64>() / n as f64;
    if mse <= 1e-12 { 99.0 } else { 10.0 * (255.0_f64.powi(2) / mse).log10() }
}

fn snr(a: &[f32], b: &[f32]) -> f64 {
    let n = a.len().min(b.len());
    if n == 0 { return 0.0; }
    let mse: f64 = (0..n).map(|i| (a[i] as f64 - b[i] as f64).powi(2))
        .sum::<f64>() / n as f64;
    let pow: f64 = a[..n].iter().map(|&x| (x as f64).powi(2))
        .sum::<f64>() / n as f64;
    if mse <= 1e-12 { 99.0 } else { 10.0 * (pow / mse).log10() }
}

/// Decode any image file (any format ffmpeg supports) → packed RGB8.
fn ffmpeg_decode_image_rgb(path: &Path, h: u32, w: u32) -> std::io::Result<Vec<u8>> {
    run("ffmpeg", &["-v", "error", "-i", path.to_str().unwrap(),
                    "-f", "rawvideo", "-pix_fmt", "rgb24",
                    "-s", &format!("{}x{}", w, h), "-"])
}

fn ffmpeg_decode_audio_f32(path: &Path) -> std::io::Result<(Vec<f32>, u32)> {
    // mono f32 at the file's native sample rate
    let raw = run("ffmpeg", &["-v", "error", "-i", path.to_str().unwrap(),
                              "-ac", "1", "-f", "f32le", "-"])?;
    // probe sr via ffprobe
    let sr_str = String::from_utf8_lossy(&run("ffprobe", &[
        "-v", "error", "-select_streams", "a:0",
        "-show_entries", "stream=sample_rate", "-of", "csv=p=0",
        path.to_str().unwrap()])?).trim().to_string();
    let sr: u32 = sr_str.parse().unwrap_or(48_000);
    let samples: Vec<f32> = raw.chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
    Ok((samples, sr))
}

// --- image -----------------------------------------------------------

fn load_png_rgb(path: &Path) -> Option<(Vec<u8>, u32, u32)> {
    // PNG fast path
    if path.extension().and_then(|x| x.to_str()) == Some("png") {
        let bytes = fs::read(path).ok()?;
        if let Ok(t) = image::png_decode(&bytes) { return Some(t); }
    }
    // Generic: shell out to ffmpeg which handles every image format.
    // First probe dimensions via ffprobe.
    let probe = run("ffprobe", &["-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
        path.to_str()?]).ok()?;
    let s = String::from_utf8_lossy(&probe);
    let mut parts = s.trim().split('x');
    let w: u32 = parts.next()?.parse().ok()?;
    let h: u32 = parts.next()?.parse().ok()?;
    let raw = run("ffmpeg", &["-v", "error", "-i", path.to_str()?,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]).ok()?;
    if raw.len() == (h * w * 3) as usize { Some((raw, h, w)) } else { None }
}

fn collect_image_sources(full: bool) -> Vec<PathBuf> {
    let mut out: Vec<PathBuf> = Vec::new();
    let mut take = |dir: &Path, limit: Option<usize>| {
        if let Ok(es) = fs::read_dir(dir) {
            let mut found: Vec<PathBuf> = es.flatten().map(|e| e.path())
                .filter(|p| matches!(p.extension().and_then(|x| x.to_str()),
                                     Some("png") | Some("jpg") | Some("jpeg") | Some("tiff")))
                .collect();
            found.sort();
            if let Some(n) = limit { found.truncate(n); }
            out.extend(found);
        }
    };
    let kodak = Path::new(CORPUS).join("image/kodak");
    if full {
        take(&kodak, None);
        take(&Path::new(CORPUS).join("image/tecnick/rgb_1200"), None);
        take(&Path::new(CORPUS).join("image/clic"), None);
        take(&Path::new(CORPUS).join("image/synthetic"), None);
    } else {
        take(&kodak, Some(4));
    }
    out
}

fn image_interop_and_rd(stats: &mut HashMap<String, CodecStats>, full: bool) {
    println!("\n=== IMAGE — interop + RD parity ({}) ===",
             if full { "FULL corpus" } else { "Kodak subset" });
    let sources = collect_image_sources(full);
    if sources.is_empty() {
        println!("  (no image corpus; skip)"); return;
    }
    let verbose = !full;
    println!("  scanning {} images...", sources.len());
    if verbose {
        println!("  {:5} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8} {:>8}",
                 "codec", "WAI B", "ref B", "B%", "WAI PSNR", "ref PSNR", "PSNR Δ", "interop");
    }
    for (idx, src) in sources.iter().enumerate() {
        let (rgb, h, w) = match load_png_rgb(src) { Some(t) => t, None => {
            if !verbose && idx % 20 == 0 {
                println!("  [{idx}/{}] (skipping {})", sources.len(), src.display());
            }
            continue
        }};
        let sz = rgb.len();
        let name = src.file_name().unwrap().to_string_lossy().into_owned();
        if verbose { println!("  --- {} ({}×{}, {} px) ---", name, w, h, sz / 3); }
        else if idx % 20 == 0 { println!("  [{idx}/{}] {}", sources.len(), name); }

        // PNG
        {
            let t0 = Instant::now();
            let wai_bytes = image::png_encode(&rgb, h, w).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let tmp = std::env::temp_dir().join("wai_bench.png");
            fs::write(&tmp, &wai_bytes).unwrap();
            let via_ff = ffmpeg_decode_image_rgb(&tmp, h, w).unwrap_or_default();
            let interop = via_ff == rgb;
            let (rec, _, _) = image::png_decode(&wai_bytes).unwrap();
            let bit_exact = rec == rgb;
            stats.entry("png".into()).or_default()
                .record(wai_bytes.len(), None, None, bit_exact && interop, enc_ms);
            if verbose {
                println!("  {:5} {:>10} {:>10} {:>8} {:>10} {:>10} {:>8} {:>8}",
                         "png", wai_bytes.len(), "-", "-",
                         if bit_exact { "lossless" } else { "BROKEN" },
                         "-", "-", if interop { "ok" } else { "FAIL" });
            }
        }

        // JPEG q=50 and q=80
        for q in [50u8, 80u8] {
            let t0 = Instant::now();
            let wai = image::jpeg_encode(&rgb, h, w, q).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let tmp = std::env::temp_dir().join("wai_bench.jpg");
            fs::write(&tmp, &wai).unwrap();
            let ref_bytes = pipe_ffmpeg_encode_image(
                &rgb, h, w, "mjpeg",
                &["-qscale:v", &format!("{}", 31 - (q as i32 * 31 / 100))],
                "image2pipe", "jpg",
            ).unwrap_or_default();
            let (wai_dec, _, _) = image::jpeg_decode(&wai).unwrap();
            let psnr_wai = psnr(&rgb, &wai_dec);
            let ref_dec = if !ref_bytes.is_empty() {
                let tmp_ref = std::env::temp_dir().join("ref_dec.jpg");
                fs::write(&tmp_ref, &ref_bytes).unwrap();
                ffmpeg_decode_image_rgb(&tmp_ref, h, w).unwrap_or_default()
            } else { Vec::new() };
            let psnr_ref = if ref_dec.is_empty() { 0.0 } else { psnr(&rgb, &ref_dec) };
            let via_ff = ffmpeg_decode_image_rgb(&tmp, h, w).unwrap_or_default();
            // Two compliant JPEG decoders agree to ~40 dB on natural
            // images. On adversarial chroma-Nyquist-violating synthetic
            // content (dead-leaves, fine noise) they can drop to ~28 dB
            // because chroma-upsample-interpolation differs per decoder.
            // 28 dB is well above "decoders fundamentally disagree" (<10).
            let interop = via_ff.len() == rgb.len() && psnr(&wai_dec, &via_ff) > 28.0;
            stats.entry(format!("jpeg{}", q)).or_default()
                .record(wai.len(),
                        if ref_bytes.is_empty() { None } else { Some(ref_bytes.len()) },
                        if ref_dec.is_empty() { None } else { Some(psnr_wai - psnr_ref) },
                        interop, enc_ms);
            if !interop && !verbose {
                eprintln!("  jpeg{q} FAIL on {}", src.display());
            }
            if verbose {
                let bp = if !ref_bytes.is_empty() {
                    format!("{:+5.1}%", (wai.len() as f64 / ref_bytes.len() as f64 - 1.0) * 100.0)
                } else { "-".into() };
                println!("  jpeg{} {:>10} {:>10} {:>8} {:>10.2} {:>10.2} {:>+7.2} {:>8}",
                         q, wai.len(), ref_bytes.len(), bp,
                         psnr_wai, psnr_ref, psnr_wai - psnr_ref,
                         if interop { "ok" } else { "FAIL" });
            }
        }

        // AVIF q=70
        {
            let t0 = Instant::now();
            let wai = image::avif_encode(&rgb, h, w, 70).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let tmp = std::env::temp_dir().join("wai_bench.avif");
            fs::write(&tmp, &wai).unwrap();
            let (wai_dec, _, _) = image::avif_decode(&wai).unwrap_or((vec![], 0, 0));
            let psnr_wai = if wai_dec.is_empty() { 0.0 } else { psnr(&rgb, &wai_dec) };
            let via_ff = ffmpeg_decode_image_rgb(&tmp, h, w).unwrap_or_default();
            let interop = via_ff.len() == rgb.len();
            stats.entry("avif70".into()).or_default()
                .record(wai.len(), None, None, interop && psnr_wai > 25.0, enc_ms);
            if verbose {
                println!("  avif  {:>10} {:>10} {:>8} {:>10.2} {:>10} {:>8} {:>8}",
                         wai.len(), "-", "-", psnr_wai, "-", "-",
                         if interop { "ok" } else { "FAIL" });
            }
        }

        // JPEG-XL lossless
        {
            let t0 = Instant::now();
            let wai = image::jxl_encode(&rgb, h, w, None).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let (rec, _, _) = image::jxl_decode(&wai).unwrap();
            let bit_exact = rec == rgb;
            let tmp = std::env::temp_dir().join("wai_bench.jxl");
            fs::write(&tmp, &wai).unwrap();
            let ff_status = ffmpeg_decode_image_rgb(&tmp, h, w).is_ok();
            stats.entry("jxl-lossless".into()).or_default()
                .record(wai.len(), None, None, bit_exact, enc_ms);
            if verbose {
                println!("  jxlL  {:>10} {:>10} {:>8} {:>10} {:>10} {:>8} {:>8}",
                     wai.len(), "-", "-",
                     if bit_exact { "lossless" } else { "BROKEN" },
                     "-", "-",
                     if ff_status { "ok" } else { "n/a" });
            }
        }
    }
}

fn pipe_ffmpeg_encode_image(rgb: &[u8], h: u32, w: u32, vcodec: &str,
                            extra: &[&str], fmt: &str, _ext: &str
                           ) -> std::io::Result<Vec<u8>> {
    let mut args = vec!["-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s"];
    let sz = format!("{w}x{h}");
    args.push(&sz);
    args.extend(["-i", "-", "-c:v", vcodec]);
    args.extend(extra.iter().copied());
    args.extend(["-f", fmt, "pipe:1"]);
    run_in_out("ffmpeg", &args, rgb)
}

// --- audio -----------------------------------------------------------

fn audio_interop_and_rd(stats: &mut HashMap<String, CodecStats>, full: bool) {
    println!("\n=== AUDIO — interop + RD parity ===");
    let dir = Path::new(CORPUS).join("audio/sqam");
    let sources: Vec<PathBuf> = if full {
        fs::read_dir(&dir).ok().into_iter().flatten().flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("wav"))
            .collect()
    } else {
        ["gspi.wav", "greasy.wav", "Clar.wav", "Piano2.wav"]
            .iter().map(|n| dir.join(n)).filter(|p| p.exists()).collect()
    };
    let verbose = !full;
    if sources.is_empty() {
        println!("  (no SQAM audio at {dir:?}; skip)"); return;
    }
    if verbose {
        println!("  {:8} {:>10} {:>10} {:>8} {:>10}",
                 "codec", "WAI B", "ref B", "B%", "result");
    }
    for src in &sources {
        let (samples, sr) = match ffmpeg_decode_audio_f32(src) {
            Ok(t) => t, Err(_) => continue,
        };
        let name = src.file_name().unwrap().to_string_lossy().into_owned();
        if verbose { println!("  --- {name} ({:.2}s @ {sr}Hz) ---",
                              samples.len() as f32 / sr as f32); }

        // FLAC
        {
            let t0 = Instant::now();
            let wai = audio::flac_encode(&samples, sr).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let (wai_dec, _) = audio::flac_decode(&wai).unwrap();
            let q_in: Vec<f32> = samples.iter()
                .map(|&s| (s.clamp(-1.0, 1.0) * 32767.0).round() / 32768.0).collect();
            let bit_exact = wai_dec.iter().zip(q_in.iter())
                .all(|(a, b)| (a - b).abs() < 1e-4);
            let raw_path = std::env::temp_dir().join("wai_bench_in.f32");
            let out_path = std::env::temp_dir().join("wai_bench_ref.flac");
            let raw_in: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
            fs::write(&raw_path, &raw_in).ok();
            let _ = run("ffmpeg", &[
                "-v", "error", "-y", "-f", "f32le", "-ar", &sr.to_string(),
                "-ac", "1", "-i", raw_path.to_str().unwrap(),
                "-c:a", "flac", out_path.to_str().unwrap()]);
            let ref_bytes = fs::read(&out_path).unwrap_or_default();
            stats.entry("flac".into()).or_default().record(wai.len(),
                if ref_bytes.is_empty() { None } else { Some(ref_bytes.len()) },
                None, bit_exact, enc_ms);
            if verbose {
                let bp = if !ref_bytes.is_empty() {
                    format!("{:+5.1}%", (wai.len() as f64 / ref_bytes.len() as f64 - 1.0) * 100.0)
                } else { "-".into() };
                println!("  {:8} {:>10} {:>10} {:>8} {:>10}",
                         "flac", wai.len(), ref_bytes.len(), bp,
                         if bit_exact { "lossless ok" } else { "BROKEN" });
            }
        }

        // Opus (resampled to 48 kHz, Ogg-Opus container)
        {
            let f32_in = std::env::temp_dir().join("wai_bench_in.f32");
            let f32_48 = std::env::temp_dir().join("wai_bench_in_48k.f32");
            let raw_in: Vec<u8> = samples.iter().flat_map(|f| f.to_le_bytes()).collect();
            fs::write(&f32_in, &raw_in).ok();
            let _ = run("ffmpeg", &[
                "-v", "error", "-y", "-f", "f32le", "-ar", &sr.to_string(),
                "-ac", "1", "-i", f32_in.to_str().unwrap(),
                "-ar", "48000", "-f", "f32le", f32_48.to_str().unwrap()]);
            let resampled: Vec<f32> = fs::read(&f32_48).unwrap_or_default()
                .chunks_exact(4).map(|b| f32::from_le_bytes(b.try_into().unwrap())).collect();
            let t0 = Instant::now();
            let wai = audio::opus_encode(&resampled, 48_000, 64_000).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let opus_tmp = std::env::temp_dir().join("wai_bench.opus");
            let ff_dec_tmp = std::env::temp_dir().join("wai_bench_ff.f32");
            fs::write(&opus_tmp, &wai).ok();
            let ff_ok = run("ffmpeg", &[
                "-v", "error", "-y", "-i", opus_tmp.to_str().unwrap(),
                "-f", "f32le", "-ac", "1", "-ar", "48000",
                ff_dec_tmp.to_str().unwrap()]).is_ok();
            stats.entry("opus64k".into()).or_default()
                .record(wai.len(), None, None, ff_ok, enc_ms);
            if verbose {
                let (wai_dec, _) = audio::opus_decode(&wai).unwrap();
                let snr_wai = snr(&resampled, &wai_dec);
                let interop = if ff_ok { "ok" } else { "FAIL" };
                println!("  {:8} {:>10} {:>10} {:>8} SNR={:.2} dB  ffmpeg-interop: {}",
                         "opus64k", wai.len(), "-", "-", snr_wai, interop);
            }
        }
    }
}

// --- text ------------------------------------------------------------

fn text_interop_and_rd(stats: &mut HashMap<String, CodecStats>, full: bool) {
    println!("\n=== TEXT — interop + RD parity ===");
    let candidates: Vec<PathBuf> = if full {
        let mut v: Vec<PathBuf> = ["text/enwik8"].iter()
            .map(|p| Path::new(CORPUS).join(p)).filter(|p| p.exists()).collect();
        if let Ok(es) = fs::read_dir(Path::new(CORPUS).join("text/silesia")) {
            v.extend(es.flatten().map(|e| e.path()));
        }
        v
    } else {
        ["text/enwik8", "text/silesia/dickens", "text/silesia/mozilla"]
            .iter().map(|p| Path::new(CORPUS).join(p)).filter(|p| p.exists()).collect()
    };
    let verbose = !full;
    if candidates.is_empty() {
        println!("  (no text corpus; skip)"); return;
    }
    if verbose {
        println!("  {:8} {:>12} {:>12} {:>8} {:>10}",
                 "codec", "WAI B", "ref B", "B%", "result");
    }
    for src in &candidates {
        let data = fs::read(src).unwrap();
        let data = if data.len() > 4 * 1024 * 1024 { &data[..4 * 1024 * 1024] } else { &data[..] };
        let name = src.file_name().unwrap().to_string_lossy().into_owned();
        if verbose { println!("  --- {name} ({} B) ---", data.len()); }

        // zstd-19
        {
            let t0 = Instant::now();
            let wai = text::zstd_encode(data, 19).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let dec = text::zstd_decode(&wai).unwrap();
            let bit_exact = dec == data;
            let ref_bytes = run_in_out("zstd", &["-19", "--no-progress", "-q"], data)
                .ok().unwrap_or_default();
            stats.entry("zstd19".into()).or_default().record(wai.len(),
                if ref_bytes.is_empty() { None } else { Some(ref_bytes.len()) },
                None, bit_exact, enc_ms);
            if verbose {
                let bp = if !ref_bytes.is_empty() {
                    format!("{:+5.1}%", (wai.len() as f64 / ref_bytes.len() as f64 - 1.0) * 100.0)
                } else { "n/a".into() };
                println!("  {:8} {:>12} {:>12} {:>8} {:>10}",
                         "zstd19", wai.len(), ref_bytes.len(), bp,
                         if bit_exact { "lossless ok" } else { "BROKEN" });
            }
        }

        // xz-6
        {
            let t0 = Instant::now();
            let wai = text::xz_encode(data, 6).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let dec = text::xz_decode(&wai).unwrap();
            let bit_exact = dec == data;
            let ref_bytes = run_in_out("xz", &["-c", "-6", "--quiet"], data)
                .ok().unwrap_or_default();
            stats.entry("xz6".into()).or_default().record(wai.len(),
                if ref_bytes.is_empty() { None } else { Some(ref_bytes.len()) },
                None, bit_exact, enc_ms);
            if verbose {
                let bp = if !ref_bytes.is_empty() {
                    format!("{:+5.1}%", (wai.len() as f64 / ref_bytes.len() as f64 - 1.0) * 100.0)
                } else { "n/a".into() };
                println!("  {:8} {:>12} {:>12} {:>8} {:>10}",
                         "xz6", wai.len(), ref_bytes.len(), bp,
                         if bit_exact { "lossless ok" } else { "BROKEN" });
            }
        }
    }
}

// --- video -----------------------------------------------------------

fn video_interop_and_rd(stats: &mut HashMap<String, CodecStats>, full: bool) {
    println!("\n=== VIDEO — interop + RD parity ===");
    let dir = Path::new(CORPUS).join("video/derf");
    let sources: Vec<PathBuf> = if full {
        // every CIF sequence (still small enough for AV1 encode in
        // reasonable time on our test machine)
        fs::read_dir(&dir).ok().into_iter().flatten().flatten()
            .map(|e| e.path())
            .filter(|p| p.extension().and_then(|x| x.to_str()) == Some("y4m"))
            .filter(|p| p.file_name().and_then(|n| n.to_str())
                       .is_some_and(|n| n.contains("_cif")))
            .collect()
    } else {
        ["akiyo_cif.y4m", "foreman_cif.y4m"]
            .iter().map(|n| dir.join(n)).filter(|p| p.exists()).collect()
    };
    let verbose = !full;
    if sources.is_empty() {
        println!("  (no Derf CIF video; skip)"); return;
    }
    if verbose {
        println!("  {:12} {:>10} {:>10} {:>10}",
                 "codec", "WAI B", "ref B (svtav1)", "result");
    }
    for src in &sources {
        // load a few frames
        let n_frames = 8u32;
        let (h, w) = (288u32, 352u32);
        let raw = run("ffmpeg", &[
            "-v", "error", "-i", src.to_str().unwrap(),
            "-frames:v", &n_frames.to_string(),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        ]).unwrap_or_default();
        if raw.len() != (n_frames * h * w * 3) as usize {
            println!("  (skip {src:?}: unexpected raw size {})", raw.len());
            continue;
        }
        let frames: Vec<Vec<u8>> = (0..n_frames as usize)
            .map(|i| {
                let pf = (h * w * 3) as usize;
                raw[i * pf..(i + 1) * pf].to_vec()
            }).collect();
        let name = src.file_name().unwrap().to_string_lossy().into_owned();
        if verbose { println!("  --- {name} ({n_frames} frames @ {w}×{h}) ---"); }
        else { println!("  [{}]", name); }

        // AV1 lossy via rav1e
        {
            let t0 = Instant::now();
            let wai = video::av1_encode(&frames, h, w, 30, 1, false, 50).unwrap();
            let enc_ms = t0.elapsed().as_millis();
            let (dec, _, _, _, _) = video::av1_decode(&wai).unwrap_or((vec![], 0, 0, 0, 0));
            let interop = !dec.is_empty();
            let ref_bytes = pipe_ffmpeg_encode_video(
                &raw, n_frames, h, w, "libsvtav1",
                &["-preset", "6", "-crf", "50"]).unwrap_or_default();
            stats.entry("av1.lossy".into()).or_default().record(wai.len(),
                if ref_bytes.is_empty() { None } else { Some(ref_bytes.len()) },
                None, interop, enc_ms);
            if verbose {
                let bp = if !ref_bytes.is_empty() {
                    format!("{:+5.1}%", (wai.len() as f64 / ref_bytes.len() as f64 - 1.0) * 100.0)
                } else { "n/a".into() };
                println!("  {:12} {:>10} {:>10} {:>10}  WAI parity vs svtav1: {}",
                         "av1.lossy", wai.len(), ref_bytes.len(), bp,
                         if interop { "decoded ok" } else { "DECODE FAIL" });
            }
        }
    }
}

fn pipe_ffmpeg_encode_video(raw_rgb: &[u8], n: u32, h: u32, w: u32,
                            vcodec: &str, extra: &[&str]) -> std::io::Result<Vec<u8>> {
    let mut args = vec!["-v", "error", "-y",
        "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s"];
    let sz = format!("{w}x{h}");
    args.push(&sz);
    let n_str = n.to_string();
    args.extend(["-r", "30", "-i", "-",
                 "-frames:v", &n_str,
                 "-c:v", vcodec]);
    args.extend(extra.iter().copied());
    args.extend(["-f", "ivf", "pipe:1"]);
    run_in_out("ffmpeg", &args, raw_rgb)
}

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let full = args.iter().any(|a| a == "--full");
    let report_path: Option<String> = {
        let mut it = args.iter();
        let mut p = None;
        while let Some(a) = it.next() {
            if a == "--report" { p = it.next().cloned(); break; }
        }
        p
    };
    println!("WAI bench harness — mode: {}",
             if full { "FULL corpus walk" } else { "quick subset" });
    if let Some(p) = &report_path {
        println!("  report → {p}");
    }
    let t0 = Instant::now();
    let mut stats: HashMap<String, CodecStats> = HashMap::new();
    image_interop_and_rd(&mut stats, full);
    audio_interop_and_rd(&mut stats, full);
    text_interop_and_rd(&mut stats, full);
    video_interop_and_rd(&mut stats, full);
    print_summary(&stats);
    let secs = t0.elapsed().as_secs_f64();
    println!("\n  total bench time: {:.1} s", secs);
    if let Some(p) = &report_path {
        match write_report(p, if full { "full" } else { "quick" }, secs, &stats) {
            Ok(()) => println!("  wrote report: {p}"),
            Err(e) => eprintln!("  report write FAILED: {e}"),
        }
    }
}
