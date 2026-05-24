fn psnr(a: &[u8], b: &[u8]) -> f64 {
    let n = a.len().min(b.len());
    let mse: f64 = (0..n).map(|i| (a[i] as f64 - b[i] as f64).powi(2)).sum::<f64>() / n as f64;
    if mse <= 1e-12 { 99.0 } else { 10.0 * (255.0_f64.powi(2) / mse).log10() }
}
fn main() {
    let p = "/Users/dcharlot/data-share/vibe-coding/web_standard_new/corpus/image/synthetic/deadleaves.png";
    let bytes = std::fs::read(p).unwrap();
    let (rgb, h, w) = wai::codecs::image::png_decode(&bytes).unwrap();
    println!("source {}×{} px", w, h);
    for q in [50u8, 80] {
        let wai = wai::codecs::image::jpeg_encode(&rgb, h, w, q).unwrap();
        std::fs::write(format!("/tmp/dl_q{q}.jpg"), &wai).unwrap();
        let (wai_dec, _, _) = wai::codecs::image::jpeg_decode(&wai).unwrap();
        let ff = std::process::Command::new("ffmpeg").args([
            "-v", "error", "-i", &format!("/tmp/dl_q{q}.jpg"),
            "-f", "rawvideo", "-pix_fmt", "rgb24", "-"
        ]).output().unwrap().stdout;
        let psnr_src_wai = psnr(&rgb, &wai_dec);
        let psnr_src_ff = psnr(&rgb, &ff);
        let psnr_wai_vs_ff = psnr(&wai_dec, &ff);
        println!("  q={q}: src-vs-WAI={:.2} src-vs-ffmpeg={:.2} WAI-vs-ffmpeg={:.2} dB (interop threshold 35)",
                 psnr_src_wai, psnr_src_ff, psnr_wai_vs_ff);
    }
}
