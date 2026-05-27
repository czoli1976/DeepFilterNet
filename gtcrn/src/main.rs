use anyhow::{bail, Context, Result};
use clap::Parser;
use gtcrn::{GtcrnStream, HOP, SAMPLE_RATE};
use std::path::PathBuf;
use std::time::Instant;

/// Enhance a noisy WAV file with the streaming GTCRN model.
#[derive(Parser)]
#[command(name = "gtcrn-enhance", about)]
struct Args {
    /// Input WAV file (mono, 16 kHz).
    input: PathBuf,
    /// Output WAV file.
    output: PathBuf,
}

fn main() -> Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();
    let args = Args::parse();

    let mut reader = hound::WavReader::open(&args.input)
        .with_context(|| format!("Could not open input WAV: {}", args.input.display()))?;
    let spec = reader.spec();
    if spec.sample_rate != SAMPLE_RATE {
        bail!(
            "GTCRN expects {} Hz audio, but input is {} Hz. Please resample first.",
            SAMPLE_RATE,
            spec.sample_rate
        );
    }

    // Decode to f32 in [-1, 1] and downmix to mono.
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<_, _>>()?,
        hound::SampleFormat::Int => {
            let max = (1i64 << (spec.bits_per_sample - 1)) as f32;
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| v as f32 / max))
                .collect::<Result<_, _>>()?
        }
    };
    let ch = spec.channels as usize;
    let mono: Vec<f32> = if ch <= 1 {
        interleaved
    } else {
        interleaved
            .chunks(ch)
            .map(|f| f.iter().sum::<f32>() / ch as f32)
            .collect()
    };

    let audio_secs = mono.len() as f32 / SAMPLE_RATE as f32;
    log::info!("Enhancing {} samples ({:.2}s)", mono.len(), audio_secs);

    let t_load = Instant::now();
    let mut stream = GtcrnStream::new()?;
    log::info!("Model load + optimize: {:.1} ms", t_load.elapsed().as_secs_f32() * 1e3);

    // Drive the streaming API one hop (256 samples) at a time.
    let mut enhanced = Vec::with_capacity(mono.len() + HOP);
    let t_proc = Instant::now();
    for chunk in mono.chunks(HOP) {
        let mut frame = [0.0f32; HOP];
        frame[..chunk.len()].copy_from_slice(chunk);
        enhanced.extend_from_slice(&stream.process_frame(&frame)?);
    }
    enhanced.extend_from_slice(&stream.process_frame(&[0.0; HOP])?);
    let proc_secs = t_proc.elapsed().as_secs_f32();

    // Compensate the one-hop algorithmic latency and match input length.
    let mut enhanced = enhanced.split_off(HOP.min(enhanced.len()));
    enhanced.resize(mono.len(), 0.0);

    let n_frames = mono.len().div_ceil(HOP) + 1;
    log::info!(
        "Processed {} frames in {:.1} ms -> RTF = {:.4} ({:.0} us/frame, {:.1}x real-time)",
        n_frames,
        proc_secs * 1e3,
        proc_secs / audio_secs,
        proc_secs / n_frames as f32 * 1e6,
        audio_secs / proc_secs,
    );

    let out_spec = hound::WavSpec {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut writer = hound::WavWriter::create(&args.output, out_spec)
        .with_context(|| format!("Could not create output WAV: {}", args.output.display()))?;
    for s in enhanced {
        let v = (s.clamp(-1.0, 1.0) * i16::MAX as f32).round() as i16;
        writer.write_sample(v)?;
    }
    writer.finalize()?;
    log::info!("Wrote enhanced audio to {}", args.output.display());
    Ok(())
}
