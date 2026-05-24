fn main() {
    let h: u32 = 16; let w: u32 = 16;
    let mut rgb = vec![0u8; (h*w*3) as usize];
    for i in 0..h { for j in 0..w {
        let p = ((i*w+j)*3) as usize;
        rgb[p] = ((i+j)*8) as u8; rgb[p+1] = (i*16) as u8; rgb[p+2] = (j*16) as u8;
    }}
    let bytes = wai::codecs::image::jpeg_encode(&rgb, h, w, 80).unwrap();
    std::fs::write("/tmp/test.jpg", &bytes).unwrap();
    println!("wrote {} bytes", bytes.len());
    let (rec, rh, rw) = wai::codecs::image::jpeg_decode(&bytes).unwrap();
    println!("decoded {}x{}", rh, rw);
    let mse: f64 = rgb.iter().zip(&rec).map(|(&a,&b)| (a as f64 - b as f64).powi(2)).sum::<f64>() / rgb.len() as f64;
    println!("PSNR: {:.2} dB", 10.0 * (255.0f64.powi(2) / mse).log10());
}
