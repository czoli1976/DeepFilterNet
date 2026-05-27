//! Streaming GTCRN speech enhancement inference.
//!
//! GTCRN (Rong et al., 2024) is an ultra-light speech enhancement model. This
//! crate runs the streaming ONNX export (`gtcrn_simple.onnx`) frame-by-frame via
//! tract, with persistent convolution / attention / RNN caches, wrapped in a
//! sqrt-Hann STFT analysis-synthesis pipeline matching the reference.
//!
//! The hot path is built to lean on tract: the runnable plan is decluttered and
//! optimized once, a [`tract_onnx::prelude::SimpleState`] is spawned once and
//! reused for every frame (reusing scratch buffers), and the recurrent caches
//! are cycled output->input as reference-counted [`TValue`]s without any copy.

use anyhow::{Context, Result};
use realfft::num_complex::Complex32;
use realfft::{ComplexToReal, RealFftPlanner, RealToComplex};
use std::sync::Arc;
use tract_onnx::prelude::*;

pub mod rewrite;

pub const SAMPLE_RATE: u32 = 16_000;
pub const N_FFT: usize = 512;
pub const HOP: usize = 256;
pub const N_FREQ: usize = N_FFT / 2 + 1; // 257

// Cache tensor shapes of the streaming export, in model input order after `mix`.
const CONV_CACHE: [usize; 5] = [2, 1, 16, 16, 33];
const TRA_CACHE: [usize; 5] = [2, 3, 1, 1, 16];
const INTER_CACHE: [usize; 4] = [2, 1, 33, 16];

type State = TypedSimpleState;

/// sqrt-Hann window: `hann_periodic(N) ** 0.5`, matching the reference STFT.
fn sqrt_hann(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            let h = 0.5 - 0.5 * (2.0 * std::f32::consts::PI * i as f32 / n as f32).cos();
            h.sqrt()
        })
        .collect()
}

/// Fixed-size 512-point STFT helper (analysis + synthesis).
struct Stft {
    r2c: Arc<dyn RealToComplex<f32>>,
    c2r: Arc<dyn ComplexToReal<f32>>,
    window: Vec<f32>,
    fft_in: Vec<f32>,
    fft_out: Vec<Complex32>,
    ifft_in: Vec<Complex32>,
    ifft_out: Vec<f32>,
}

impl Stft {
    fn new() -> Self {
        let mut planner = RealFftPlanner::<f32>::new();
        let r2c = planner.plan_fft_forward(N_FFT);
        let c2r = planner.plan_fft_inverse(N_FFT);
        let fft_in = r2c.make_input_vec();
        let fft_out = r2c.make_output_vec();
        let ifft_in = c2r.make_input_vec();
        let ifft_out = c2r.make_output_vec();
        Self {
            r2c,
            c2r,
            window: sqrt_hann(N_FFT),
            fft_in,
            fft_out,
            ifft_in,
            ifft_out,
        }
    }

    /// Windowed forward transform of a 512-sample frame -> 257 complex bins.
    fn forward(&mut self, frame: &[f32]) -> &[Complex32] {
        debug_assert_eq!(frame.len(), N_FFT);
        for i in 0..N_FFT {
            self.fft_in[i] = frame[i] * self.window[i];
        }
        self.r2c.process(&mut self.fft_in, &mut self.fft_out).expect("rfft");
        &self.fft_out
    }

    /// Inverse transform of 257 complex bins -> windowed 512-sample frame.
    fn inverse(&mut self, spec: &[Complex32]) -> &[f32] {
        debug_assert_eq!(spec.len(), N_FREQ);
        self.ifft_in.copy_from_slice(spec);
        self.c2r.process(&mut self.ifft_in, &mut self.ifft_out).expect("irfft");
        let norm = N_FFT as f32;
        for i in 0..N_FFT {
            self.ifft_out[i] = self.ifft_out[i] / norm * self.window[i];
        }
        &self.ifft_out
    }
}

fn zero_cache(shape: &[usize]) -> TValue {
    Tensor::zero::<f32>(shape).expect("zero cache tensor").into()
}

/// Persistent GTCRN streaming state: tract session + DSP buffers + recurrent caches.
pub struct GtcrnStream {
    state: State,
    stft: Stft,
    // Sliding 512-sample analysis buffer (oldest..newest).
    ana_buf: [f32; N_FFT],
    // 256-sample overlap-add tail carried to the next frame.
    ola: [f32; HOP],
    // Flat (1, 257, 1, 2) staging buffer for the `mix` input tensor.
    mix_buf: Vec<f32>,
    // Recurrent caches, cycled output->input as Arc-backed TValues.
    conv_cache: TValue,
    tra_cache: TValue,
    inter_cache: TValue,
}

impl GtcrnStream {
    /// Load the bundled streaming GTCRN model.
    pub fn new() -> Result<Self> {
        Self::from_onnx_bytes(include_bytes!("../models/gtcrn_simple.onnx"))
    }

    pub fn from_onnx_bytes(bytes: &[u8]) -> Result<Self> {
        let mut cursor = std::io::Cursor::new(bytes);
        let mut typed = tract_onnx::onnx()
            .model_for_read(&mut cursor)
            .context("Failed to parse GTCRN ONNX model")?
            .into_typed()
            .context("Failed to type GTCRN model")?;
        typed.declutter().context("Failed to declutter GTCRN model")?;
        let rewritten = rewrite::replace_const_scatternd(&mut typed)
            .context("Failed to rewrite ScatterNd nodes")?;
        log::debug!("Rewrote {rewritten} constant-index ScatterNd nodes");
        // `into_runnable` already yields an `Arc<RunnableModel>`.
        let plan = typed
            .into_optimized()
            .context("Failed to optimize GTCRN model")?
            .into_runnable()
            .context("Failed to make GTCRN model runnable")?;
        // Spawn the session state once; reused (with its scratch) for every frame.
        let state = TypedSimpleState::new(&plan)
            .context("Failed to spawn GTCRN session state")?;
        Ok(Self {
            state,
            stft: Stft::new(),
            ana_buf: [0.0; N_FFT],
            ola: [0.0; HOP],
            mix_buf: vec![0.0; N_FREQ * 2],
            conv_cache: zero_cache(&CONV_CACHE),
            tra_cache: zero_cache(&TRA_CACHE),
            inter_cache: zero_cache(&INTER_CACHE),
        })
    }

    /// Reset DSP buffers and recurrent caches to silence/zero.
    pub fn reset(&mut self) {
        self.ana_buf = [0.0; N_FFT];
        self.ola = [0.0; HOP];
        self.conv_cache = zero_cache(&CONV_CACHE);
        self.tra_cache = zero_cache(&TRA_CACHE);
        self.inter_cache = zero_cache(&INTER_CACHE);
    }

    /// Process exactly `HOP` (256) input samples, returning `HOP` enhanced
    /// samples. Output is delayed by one hop relative to the input.
    pub fn process_frame(&mut self, input: &[f32]) -> Result<[f32; HOP]> {
        assert_eq!(input.len(), HOP, "process_frame expects {HOP} samples");

        // Slide analysis buffer: drop oldest hop, append new hop.
        self.ana_buf.copy_within(HOP.., 0);
        self.ana_buf[HOP..].copy_from_slice(input);

        // Analysis STFT -> flat (1, 257, 1, 2) [re, im] layout.
        let spec = self.stft.forward(&self.ana_buf);
        for (f, c) in spec.iter().enumerate() {
            self.mix_buf[f * 2] = c.re;
            self.mix_buf[f * 2 + 1] = c.im;
        }
        let mix: TValue = Tensor::from_shape(&[1, N_FREQ, 1, 2], &self.mix_buf)?.into();

        // Run one frame, reusing the spawned session state.
        let outputs = self.state.run(tvec!(
            mix,
            self.conv_cache.clone(),
            self.tra_cache.clone(),
            self.inter_cache.clone(),
        ))?;

        // Cycle caches: outputs[1..=3] become next frame's inputs (cheap Arc clone).
        self.conv_cache = outputs[1].clone();
        self.tra_cache = outputs[2].clone();
        self.inter_cache = outputs[3].clone();

        // Enhanced spectrum (flat re/im) -> synthesis frame.
        let enh = outputs[0].try_as_plain()?.as_slice::<f32>()?;
        let mut out_spec = [Complex32::default(); N_FREQ];
        for (f, c) in out_spec.iter_mut().enumerate() {
            c.re = enh[f * 2];
            c.im = enh[f * 2 + 1];
        }
        // A real signal's DC and Nyquist bins must be purely real.
        out_spec[0].im = 0.0;
        out_spec[N_FREQ - 1].im = 0.0;
        let frame = self.stft.inverse(&out_spec);

        // Overlap-add. sqrt-Hann on both analysis and synthesis squares to a
        // periodic Hann, which is COLA at 50% overlap, so no extra normalization.
        let mut out = [0.0f32; HOP];
        for i in 0..HOP {
            out[i] = frame[i] + self.ola[i];
            self.ola[i] = frame[i + HOP];
        }
        Ok(out)
    }
}

/// Offline convenience: enhance a full mono 16 kHz signal.
pub fn enhance(samples: &[f32]) -> Result<Vec<f32>> {
    let mut stream = GtcrnStream::new()?;
    let mut out = Vec::with_capacity(samples.len() + N_FFT);

    let mut pos = 0;
    while pos < samples.len() {
        let mut frame = [0.0f32; HOP];
        let end = (pos + HOP).min(samples.len());
        frame[..end - pos].copy_from_slice(&samples[pos..end]);
        out.extend_from_slice(&stream.process_frame(&frame)?);
        pos += HOP;
    }
    // Flush one hop of zeros to recover the tail held in the OLA buffer.
    out.extend_from_slice(&stream.process_frame(&[0.0; HOP])?);

    // Compensate the one-hop algorithmic latency and match input length.
    let start = HOP.min(out.len());
    let mut out = out.split_off(start);
    out.resize(samples.len(), 0.0);
    Ok(out)
}
