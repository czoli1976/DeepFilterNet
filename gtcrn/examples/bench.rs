//! Apples-to-apples inference benchmark mirroring the onnxruntime methodology:
//! pure per-frame `state.run` loop (no STFT), warmup + best-of-N.
//! Run: cargo run -p gtcrn --release --example bench

use std::time::Instant;
use tract_onnx::prelude::*;

const FRAMES: usize = 612;
const HOP: usize = 256;
const SR: f32 = 16_000.0;

fn main() -> TractResult<()> {
    let bytes = include_bytes!("../models/gtcrn_simple.onnx");
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let t0 = Instant::now();
    let mut typed = tract_onnx::onnx().model_for_read(&mut cursor)?.into_typed()?;
    typed.declutter()?;
    if std::env::var("GTCRN_NO_REWRITE").is_err() {
        let n = gtcrn::rewrite::replace_const_scatternd(&mut typed)?;
        eprintln!("rewrote {n} ScatterNd nodes");
    }
    let plan = typed.into_optimized()?.into_runnable()?;
    let mut state = TypedSimpleState::new(&plan)?;
    eprintln!("model load+optimize: {:.1} ms", t0.elapsed().as_secs_f32() * 1e3);

    let mix: TValue = Tensor::zero::<f32>(&[1, 257, 1, 2])?.into();
    let conv0 = Tensor::zero::<f32>(&[2, 1, 16, 16, 33])?;
    let tra0 = Tensor::zero::<f32>(&[2, 3, 1, 1, 16])?;
    let inter0 = Tensor::zero::<f32>(&[2, 1, 33, 16])?;

    let run = |state: &mut TypedSimpleState| -> TractResult<()> {
        let mut conv: TValue = conv0.clone().into();
        let mut tra: TValue = tra0.clone().into();
        let mut inter: TValue = inter0.clone().into();
        for _ in 0..FRAMES {
            let o = state.run(tvec!(mix.clone(), conv, tra, inter))?;
            conv = o[1].clone();
            tra = o[2].clone();
            inter = o[3].clone();
        }
        Ok(())
    };

    for _ in 0..20 {
        let mut conv: TValue = conv0.clone().into();
        let mut tra: TValue = tra0.clone().into();
        let mut inter: TValue = inter0.clone().into();
        let o = state.run(tvec!(mix.clone(), conv.clone(), tra.clone(), inter.clone()))?;
        conv = o[1].clone();
        tra = o[2].clone();
        inter = o[3].clone();
        let _ = (conv, tra, inter);
    }

    let mut best = f32::MAX;
    for _ in 0..5 {
        let t = Instant::now();
        run(&mut state)?;
        best = best.min(t.elapsed().as_secs_f32());
    }

    let audio_s = FRAMES as f32 * HOP as f32 / SR;
    println!(
        "tract: {} frames (best of 5): {:.1} ms -> {:.0} us/frame  RTF={:.4}  ({:.1}x RT)",
        FRAMES,
        best * 1e3,
        best / FRAMES as f32 * 1e6,
        best / audio_s,
        audio_s / best,
    );
    Ok(())
}
