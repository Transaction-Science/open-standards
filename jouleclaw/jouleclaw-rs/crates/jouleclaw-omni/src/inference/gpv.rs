//! Generation Parameter Vector (GPV) types.
//!
//! The GPV is the universal interface between neural understanding
//! and procedural execution in the fusion architecture.

/// A signed distance field operation node.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub enum SDFOp {
    /// Sphere at origin with given radius.
    Sphere {
        /// Sphere radius.
        radius: f32,
    },
    /// Axis-aligned box with half-extents.
    Box {
        /// Half-extents along each axis [x, y, z].
        half_extents: [f32; 3],
    },
    /// Torus in the XZ plane.
    Torus {
        /// Distance from center of torus to center of tube.
        major_radius: f32,
        /// Radius of the tube.
        minor_radius: f32,
    },
    /// Infinite plane with normal and offset.
    Plane {
        /// Plane normal direction [x, y, z].
        normal: [f32; 3],
        /// Distance offset along normal.
        offset: f32,
    },
    /// Cylinder along Y axis.
    Cylinder {
        /// Cylinder radius.
        radius: f32,
        /// Cylinder height.
        height: f32,
    },
    /// Cone along Y axis.
    Cone {
        /// Base radius of the cone.
        radius: f32,
        /// Height of the cone.
        height: f32,
    },
    /// Translate a child operation.
    Translate {
        /// Translation offset [x, y, z].
        offset: [f32; 3],
        /// Index of the child SDF operation.
        child: usize,
    },
    /// Rotate a child (axis-angle, radians).
    Rotate {
        /// Rotation axis [x, y, z].
        axis: [f32; 3],
        /// Rotation angle in radians.
        angle: f32,
        /// Index of the child SDF operation.
        child: usize,
    },
    /// Scale a child uniformly.
    Scale {
        /// Uniform scale factor.
        factor: f32,
        /// Index of the child SDF operation.
        child: usize,
    },
    /// Smooth union of two children.
    SmoothUnion {
        /// Index of the first child.
        a: usize,
        /// Index of the second child.
        b: usize,
        /// Smoothing factor.
        k: f32,
    },
    /// Smooth subtraction (a minus b).
    SmoothSubtraction {
        /// Index of the first child.
        a: usize,
        /// Index of the second child.
        b: usize,
        /// Smoothing factor.
        k: f32,
    },
    /// Smooth intersection of two children.
    SmoothIntersection {
        /// Index of the first child.
        a: usize,
        /// Index of the second child.
        b: usize,
        /// Smoothing factor.
        k: f32,
    },
}

/// A layer of procedural noise to composite.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct NoiseLayer {
    /// Noise type.
    pub noise_type: NoiseType,
    /// Number of FBM octaves (1 = single pass).
    pub octaves: u32,
    /// Frequency multiplier per octave.
    pub lacunarity: f32,
    /// Amplitude multiplier per octave.
    pub persistence: f32,
    /// Base frequency scale.
    pub scale: f32,
    /// Amplitude of this layer.
    pub amplitude: f32,
    /// Blend mode with previous layers.
    pub blend: BlendMode,
    /// Seed for deterministic generation.
    pub seed: u32,
}

/// Supported noise types.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub enum NoiseType {
    /// Perlin gradient noise.
    Perlin,
    /// Simplex noise.
    Simplex,
    /// Worley (cellular/Voronoi) noise.
    Worley,
    /// Value noise.
    Value,
}

/// Blend mode for compositing noise layers.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub enum BlendMode {
    /// Replace: output = this_layer
    Replace,
    /// Additive: output = prev + this_layer
    Add,
    /// Multiplicative: output = prev * this_layer
    Multiply,
    /// Screen: output = 1 - (1-prev) * (1-this_layer)
    Screen,
    /// Overlay: conditional multiply/screen
    Overlay,
}

/// RGB color with f32 components [0,1].
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct Color {
    /// Red channel [0,1].
    pub r: f32,
    /// Green channel [0,1].
    pub g: f32,
    /// Blue channel [0,1].
    pub b: f32,
}

/// A color gradient stop.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct GradientStop {
    /// Position along the gradient [0,1].
    pub position: f32,
    /// Color at this stop.
    pub color: Color,
}

/// Color palette for procedural image generation.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct ColorPalette {
    /// Gradient stops defining the palette.
    pub stops: Vec<GradientStop>,
}

/// Image composition parameters.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct Composition {
    /// Camera eye position for SDF ray marching.
    pub camera_eye: [f32; 3],
    /// Camera look-at target.
    pub camera_target: [f32; 3],
    /// Field of view in radians.
    pub fov: f32,
    /// Maximum ray march distance.
    pub max_distance: f32,
    /// Maximum ray march steps.
    pub max_steps: u32,
    /// Background color.
    pub background: Color,
    /// Light direction (normalized).
    pub light_dir: [f32; 3],
}

/// Waveguide synthesis parameters (Karplus-Strong family).
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct WaveguideParams {
    /// Delay line lengths in samples (determines pitch per string).
    pub delay_samples: Vec<u32>,
    /// Feedback damping factor per string [0,1] (higher = longer sustain).
    pub damping: Vec<f32>,
    /// Brightness filter coefficient per string [0,1].
    pub brightness: Vec<f32>,
}

/// Excitation signal parameters for waveguide synthesis.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct ExcitationParams {
    /// Type of excitation.
    pub excitation_type: ExcitationType,
    /// Pluck/strike position along string [0,1].
    pub position: f32,
    /// Velocity/amplitude [0,1].
    pub velocity: f32,
}

/// Excitation signal types.
#[derive(Debug, Clone, Copy)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub enum ExcitationType {
    /// Short noise burst (plucked string).
    Pluck,
    /// Continuous noise (bowed string).
    Bow,
    /// Impulse (struck string/percussion).
    Strike,
}

/// ADSR envelope.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct ADSREnvelope {
    /// Attack time in seconds.
    pub attack: f32,
    /// Decay time in seconds.
    pub decay: f32,
    /// Sustain level [0,1].
    pub sustain: f32,
    /// Release time in seconds.
    pub release: f32,
}

/// A note event in a sequence.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct NoteEvent {
    /// MIDI note number (60 = middle C).
    pub midi_note: u8,
    /// Velocity [0,1].
    pub velocity: f32,
    /// Duration in beats.
    pub duration: f32,
    /// Start time in beats.
    pub start: f32,
}

/// A sequence of note events.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct NoteSequence {
    /// Tempo in BPM.
    pub bpm: f32,
    /// Sample rate.
    pub sample_rate: u32,
    /// Note events.
    pub notes: Vec<NoteEvent>,
}

/// Generation Parameter Vector — the universal interface between
/// neural understanding and procedural execution.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub struct GPV {
    /// Modality this GPV targets.
    pub modality: GPVModality,
    /// Confidence that procedural generation will be sufficient [0,1].
    pub procedural_confidence: f32,
    /// Structured parameters for the procedural engine.
    pub params: GPVParams,
}

/// Target modality for a GPV.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub enum GPVModality {
    /// 2D image generation.
    Image,
    /// Audio synthesis.
    Audio,
    /// 3D model generation.
    ThreeD,
    /// Video generation.
    Video,
}

/// Modality-specific procedural parameters.
#[derive(Debug, Clone)]
#[cfg_attr(feature = "server", derive(serde::Serialize, serde::Deserialize))]
pub enum GPVParams {
    /// Parameters for procedural image generation.
    Image {
        /// SDF operations defining the scene geometry.
        sdf_ops: Vec<SDFOp>,
        /// Noise layers for procedural texturing.
        noise_layers: Vec<NoiseLayer>,
        /// Color palette for the image.
        palette: ColorPalette,
        /// Camera and rendering composition settings.
        composition: Composition,
    },
    /// Parameters for procedural audio synthesis.
    Audio {
        /// Waveguide synthesis parameters.
        waveguide: WaveguideParams,
        /// Excitation signal parameters.
        excitation: ExcitationParams,
        /// ADSR amplitude envelope.
        envelope: ADSREnvelope,
        /// Note event sequence.
        sequence: NoteSequence,
    },
}

impl Default for Composition {
    fn default() -> Self {
        Self {
            camera_eye: [0.0, 0.0, 3.0],
            camera_target: [0.0, 0.0, 0.0],
            fov: 1.0,
            max_distance: 100.0,
            max_steps: 128,
            background: Color { r: 0.1, g: 0.1, b: 0.15 },
            light_dir: [0.577, 0.577, -0.577],
        }
    }
}

impl Default for ADSREnvelope {
    fn default() -> Self {
        Self {
            attack: 0.01,
            decay: 0.1,
            sustain: 0.5,
            release: 0.3,
        }
    }
}

impl Default for ColorPalette {
    fn default() -> Self {
        Self {
            stops: vec![
                GradientStop { position: 0.0, color: Color { r: 0.2, g: 0.3, b: 0.8 } },
                GradientStop { position: 0.5, color: Color { r: 0.8, g: 0.6, b: 0.2 } },
                GradientStop { position: 1.0, color: Color { r: 0.9, g: 0.9, b: 0.9 } },
            ],
        }
    }
}
