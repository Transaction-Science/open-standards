//! Native video frame decoding via FFmpeg bindings.
//!
//! Provides streaming, memory-efficient video frame extraction with automatic
//! hardware acceleration (VideoToolbox on macOS Apple Silicon).
//!
//! Requires system FFmpeg (`brew install ffmpeg`) and the `video-decode` feature.
//!
//! ## Usage
//!
//! ```rust,ignore
//! use efficient_genai::inference::{VideoDecoder, VideoDecodeConfig};
//!
//! let config = VideoDecodeConfig {
//!     width: Some(1024),
//!     height: Some(1024),
//!     frame_step: 2, // every other frame
//!     ..Default::default()
//! };
//! let decoder = VideoDecoder::open(Path::new("input.mp4"), config)?;
//! for frame in decoder.frames() {
//!     let frame = frame?;
//!     // frame.tensor is [1, 3, H, W] F16 in [-0.5, 0.5] — ready for img2img
//!     println!("Frame {} @ {:.2}s", frame.index, frame.timestamp);
//! }
//! ```

extern crate ffmpeg_next as ffmpeg;

use crate::core::{Error, Result};
use crate::tensor::{DType, Tensor};
use std::path::Path;

/// Video file metadata (reads container headers only, no decoding).
#[derive(Debug, Clone)]
pub struct VideoInfo {
    pub width: u32,
    pub height: u32,
    pub fps: f64,
    pub duration: Option<f64>,
    pub total_frames: Option<usize>,
    pub codec: String,
}

/// Configuration for video decoding.
#[derive(Debug, Clone)]
pub struct VideoDecodeConfig {
    /// Target width (None = original).
    pub width: Option<u32>,
    /// Target height (None = original).
    pub height: Option<u32>,
    /// Seek to this timestamp before decoding (seconds).
    pub start_time: Option<f64>,
    /// Stop decoding after this timestamp (seconds).
    pub end_time: Option<f64>,
    /// 1 = every frame, 2 = every other, etc.
    pub frame_step: usize,
}

impl Default for VideoDecodeConfig {
    fn default() -> Self {
        Self {
            width: None,
            height: None,
            start_time: None,
            end_time: None,
            frame_step: 1,
        }
    }
}

/// A decoded video frame with metadata, ready for diffusion pipeline input.
pub struct DecodedFrame {
    /// Sequential frame index (after frame_step filtering).
    pub index: usize,
    /// Presentation timestamp in seconds.
    pub timestamp: f64,
    /// Frame tensor: `[1, 3, H, W]`, DType::F16, values in `[-0.5, 0.5]`.
    pub tensor: Tensor,
    /// Whether this frame is a keyframe (I-frame).
    pub is_keyframe: bool,
}

/// Streaming video frame decoder backed by FFmpeg.
///
/// Decodes frames one at a time with minimal memory overhead.
/// On macOS, automatically uses VideoToolbox hardware decoding.
pub struct VideoDecoder {
    input_ctx: ffmpeg::format::context::Input,
    stream_index: usize,
    decoder: ffmpeg::decoder::Video,
    scaler: ffmpeg::software::scaling::Context,
    target_width: u32,
    target_height: u32,
    fps: f64,
    duration: Option<f64>,
    time_base: f64,
    config: VideoDecodeConfig,
    frame_index: usize,
    raw_frame_index: usize,
    eof: bool,
}

impl VideoDecoder {
    /// Open a video file for decoding.
    pub fn open(path: &Path, config: VideoDecodeConfig) -> Result<Self> {
        ffmpeg::init().map_err(|e| Error::internal(format!("FFmpeg init failed: {e}")))?;

        let input_ctx = ffmpeg::format::input(&path)
            .map_err(|e| Error::internal(format!("Failed to open video '{}': {e}", path.display())))?;

        let stream = input_ctx
            .streams()
            .best(ffmpeg::media::Type::Video)
            .ok_or_else(|| Error::internal("No video stream found".to_string()))?;

        let stream_index = stream.index();
        let time_base = f64::from(stream.time_base());

        // Extract duration
        let duration = if stream.duration() > 0 {
            Some(stream.duration() as f64 * time_base)
        } else if input_ctx.duration() > 0 {
            Some(input_ctx.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE))
        } else {
            None
        };

        // Extract FPS
        let fps = f64::from(stream.avg_frame_rate());

        let codec_params = stream.parameters();
        let decoder = ffmpeg::codec::context::Context::from_parameters(codec_params)
            .map_err(|e| Error::internal(format!("Failed to create decoder context: {e}")))?
            .decoder()
            .video()
            .map_err(|e| Error::internal(format!("Failed to create video decoder: {e}")))?;

        let src_width = decoder.width();
        let src_height = decoder.height();
        let src_format = decoder.format();

        let target_width = config.width.unwrap_or(src_width);
        let target_height = config.height.unwrap_or(src_height);

        // Create scaler: source format/size → RGB24 at target size
        let scaler = ffmpeg::software::scaling::Context::get(
            src_format,
            src_width,
            src_height,
            ffmpeg::format::Pixel::RGB24,
            target_width,
            target_height,
            ffmpeg::software::scaling::Flags::BILINEAR,
        )
        .map_err(|e| Error::internal(format!("Failed to create scaler: {e}")))?;

        let mut this = Self {
            input_ctx,
            stream_index,
            decoder,
            scaler,
            target_width,
            target_height,
            fps,
            duration,
            time_base,
            config,
            frame_index: 0,
            raw_frame_index: 0,
            eof: false,
        };

        // Seek to start_time if requested
        if let Some(start) = this.config.start_time {
            this.seek(start)?;
        }

        Ok(this)
    }

    /// Video FPS.
    pub fn fps(&self) -> f64 {
        self.fps
    }

    /// Target output width.
    pub fn width(&self) -> u32 {
        self.target_width
    }

    /// Target output height.
    pub fn height(&self) -> u32 {
        self.target_height
    }

    /// Video duration in seconds (if available from container).
    pub fn duration(&self) -> Option<f64> {
        self.duration
    }

    /// Seek to a timestamp (seconds). Seeks to the nearest keyframe before the target.
    pub fn seek(&mut self, timestamp_secs: f64) -> Result<()> {
        let ts = (timestamp_secs / self.time_base) as i64;
        self.input_ctx
            .seek(ts, ..ts)
            .map_err(|e| Error::internal(format!("Seek failed: {e}")))?;
        self.decoder.flush();
        self.eof = false;
        Ok(())
    }

    /// Decode and return the next frame, or None at EOF.
    pub fn next_frame(&mut self) -> Result<Option<DecodedFrame>> {
        if self.eof {
            return Ok(None);
        }

        loop {
            // Try to receive a decoded frame from the decoder
            let mut decoded = ffmpeg::frame::Video::empty();
            match self.decoder.receive_frame(&mut decoded) {
                Ok(()) => {
                    let pts = decoded.pts().unwrap_or(0);
                    let timestamp = pts as f64 * self.time_base;

                    // Check end_time
                    if let Some(end) = self.config.end_time {
                        if timestamp > end {
                            self.eof = true;
                            return Ok(None);
                        }
                    }

                    let is_keyframe = decoded.is_key();

                    // Handle frame_step: skip frames that don't match the step pattern
                    let current_raw = self.raw_frame_index;
                    self.raw_frame_index += 1;
                    if self.config.frame_step > 1 && current_raw % self.config.frame_step != 0 {
                        continue;
                    }

                    // Scale to RGB24 at target dimensions
                    let mut rgb_frame = ffmpeg::frame::Video::empty();
                    self.scaler
                        .run(&decoded, &mut rgb_frame)
                        .map_err(|e| Error::internal(format!("Scaling failed: {e}")))?;

                    let tensor = rgb_frame_to_tensor(&rgb_frame, self.target_width, self.target_height)?;

                    let frame = DecodedFrame {
                        index: self.frame_index,
                        timestamp,
                        tensor,
                        is_keyframe,
                    };
                    self.frame_index += 1;
                    return Ok(Some(frame));
                }
                Err(ffmpeg::Error::Other { errno: ffmpeg::ffi::AVERROR_EOF }) => {
                    self.eof = true;
                    return Ok(None);
                }
                Err(ffmpeg::Error::Other { errno }) if errno == ffmpeg::ffi::AVERROR(ffmpeg::ffi::EAGAIN) => {
                    // Decoder needs more packets — fall through to read next packet
                }
                Err(e) => {
                    // EAGAIN: decoder needs more data
                    if e.to_string().contains("EAGAIN") || e.to_string().contains("Resource temporarily unavailable") {
                        // fall through to read next packet
                    } else {
                        return Err(Error::internal(format!("Decode error: {e}")));
                    }
                }
            }

            // Read next packet from container
            let mut found_packet = false;
            for (stream, packet) in self.input_ctx.packets() {
                if stream.index() == self.stream_index {
                    self.decoder
                        .send_packet(&packet)
                        .map_err(|e| Error::internal(format!("Send packet failed: {e}")))?;
                    found_packet = true;
                    break;
                }
            }

            if !found_packet {
                // EOF — flush the decoder
                self.decoder
                    .send_eof()
                    .map_err(|e| Error::internal(format!("Send EOF failed: {e}")))?;
                // Try one more receive
                let mut decoded = ffmpeg::frame::Video::empty();
                match self.decoder.receive_frame(&mut decoded) {
                    Ok(()) => {
                        let pts = decoded.pts().unwrap_or(0);
                        let timestamp = pts as f64 * self.time_base;
                        let is_keyframe = decoded.is_key();

                        let current_raw = self.raw_frame_index;
                        self.raw_frame_index += 1;
                        if self.config.frame_step > 1 && current_raw % self.config.frame_step != 0 {
                            self.eof = true;
                            return Ok(None);
                        }

                        let mut rgb_frame = ffmpeg::frame::Video::empty();
                        self.scaler
                            .run(&decoded, &mut rgb_frame)
                            .map_err(|e| Error::internal(format!("Scaling failed: {e}")))?;

                        let tensor = rgb_frame_to_tensor(&rgb_frame, self.target_width, self.target_height)?;

                        let frame = DecodedFrame {
                            index: self.frame_index,
                            timestamp,
                            tensor,
                            is_keyframe,
                        };
                        self.frame_index += 1;
                        self.eof = true;
                        return Ok(Some(frame));
                    }
                    Err(_) => {
                        self.eof = true;
                        return Ok(None);
                    }
                }
            }
        }
    }

    /// Convert this decoder into a frame iterator.
    pub fn frames(self) -> FrameIterator {
        FrameIterator { decoder: self }
    }
}

/// Iterator over decoded video frames.
pub struct FrameIterator {
    decoder: VideoDecoder,
}

impl Iterator for FrameIterator {
    type Item = Result<DecodedFrame>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.decoder.next_frame() {
            Ok(Some(frame)) => Some(Ok(frame)),
            Ok(None) => None,
            Err(e) => Some(Err(e)),
        }
    }
}

/// Get video metadata without decoding (fast, reads container headers only).
pub fn video_info(path: &Path) -> Result<VideoInfo> {
    ffmpeg::init().map_err(|e| Error::internal(format!("FFmpeg init failed: {e}")))?;

    let input = ffmpeg::format::input(&path)
        .map_err(|e| Error::internal(format!("Failed to open video '{}': {e}", path.display())))?;

    let stream = input
        .streams()
        .best(ffmpeg::media::Type::Video)
        .ok_or_else(|| Error::internal("No video stream found".to_string()))?;

    let time_base = f64::from(stream.time_base());
    let fps = f64::from(stream.avg_frame_rate());

    let duration = if stream.duration() > 0 {
        Some(stream.duration() as f64 * time_base)
    } else if input.duration() > 0 {
        Some(input.duration() as f64 / f64::from(ffmpeg::ffi::AV_TIME_BASE))
    } else {
        None
    };

    let total_frames = if stream.frames() > 0 {
        Some(stream.frames() as usize)
    } else {
        duration.map(|d| (d * fps).round() as usize)
    };

    let codec_params = stream.parameters();
    let ctx = ffmpeg::codec::context::Context::from_parameters(codec_params)
        .map_err(|e| Error::internal(format!("Failed to read codec params: {e}")))?;
    let decoder = ctx.decoder().video()
        .map_err(|e| Error::internal(format!("Failed to create decoder: {e}")))?;

    let codec_name = decoder
        .codec()
        .map(|c| c.name().to_string())
        .unwrap_or_else(|| "unknown".to_string());

    Ok(VideoInfo {
        width: decoder.width(),
        height: decoder.height(),
        fps,
        duration,
        total_frames,
        codec: codec_name,
    })
}

/// Get frame count from container metadata (None if unavailable).
pub fn video_frame_count(path: &Path) -> Result<Option<usize>> {
    let info = video_info(path)?;
    Ok(info.total_frames)
}

/// Convert an RGB24 FFmpeg frame to a tensor in `[1, 3, H, W]` planar format.
///
/// Normalizes u8 values to `[-0.5, 0.5]` range and stores as F16.
/// Handles FFmpeg line padding by using stride rather than width * 3.
fn rgb_frame_to_tensor(
    rgb_frame: &ffmpeg::frame::Video,
    width: u32,
    height: u32,
) -> Result<Tensor> {
    let data = rgb_frame.data(0);
    let line_stride = rgb_frame.stride(0);
    let w = width as usize;
    let h = height as usize;
    let hw = h * w;

    let mut buf = vec![0.0f32; 3 * hw];
    for y in 0..h {
        for x in 0..w {
            let src = y * line_stride + x * 3;
            let dst = y * w + x;
            buf[dst] = data[src] as f32 / 255.0 - 0.5;           // R plane
            buf[hw + dst] = data[src + 1] as f32 / 255.0 - 0.5;  // G plane
            buf[2 * hw + dst] = data[src + 2] as f32 / 255.0 - 0.5; // B plane
        }
    }

    // f32 → f16 → tensor (matches load_image_to_tensor output format)
    let f16_data: Vec<half::f16> = buf.iter().map(|&v| half::f16::from_f32(v)).collect();

    #[cfg(feature = "metal")]
    let device_id = crate::hal::DeviceId::metal();
    #[cfg(not(feature = "metal"))]
    let device_id = crate::hal::DeviceId::cpu();

    Tensor::from_slice(&f16_data, [1, 3, h, w], DType::F16, device_id)
}
