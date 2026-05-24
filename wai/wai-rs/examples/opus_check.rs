fn main() {
    use std::fs;
    let sr = 48_000u32;
    let secs = 1.0_f32;
    let sig: Vec<f32> = (0..(sr as f32 * secs) as usize)
        .map(|i| 0.4 * (2.0 * std::f32::consts::PI * 440.0 * i as f32 / sr as f32).sin())
        .collect();
    let bytes = wai::codecs::audio::opus_encode(&sig, sr, 64_000).unwrap();
    fs::write("/tmp/wai.opus", &bytes).unwrap();
    println!("wrote {} bytes", bytes.len());
    let (wai_dec, wsr) = wai::codecs::audio::opus_decode(&bytes).unwrap();
    println!("wai decode: {} samples @ {} Hz", wai_dec.len(), wsr);
}
