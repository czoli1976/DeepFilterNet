//! Per-node profiler for the streaming GTCRN model to locate hot ops.
//! Run: cargo run -p gtcrn --release --example profile

use std::collections::HashMap;
use std::time::{Duration, Instant};
use tract_onnx::prelude::*;
use tract_onnx::tract_core::ops::OpState;
use tract_onnx::tract_core::plan::{eval as core_eval, TurnState};

fn main() -> TractResult<()> {
    let bytes = include_bytes!("../models/gtcrn_simple.onnx");
    let mut cursor = std::io::Cursor::new(&bytes[..]);
    let plan = tract_onnx::onnx()
        .model_for_read(&mut cursor)?
        .into_optimized()?
        .into_runnable()?;
    let mut state = TypedSimpleState::new(&plan)?;

    let mix = Tensor::zero::<f32>(&[1, 257, 1, 2])?;
    let mut conv: TValue = Tensor::zero::<f32>(&[2, 1, 16, 16, 33])?.into();
    let mut tra: TValue = Tensor::zero::<f32>(&[2, 3, 1, 1, 16])?.into();
    let mut inter: TValue = Tensor::zero::<f32>(&[2, 1, 33, 16])?.into();

    let iters = 300usize;
    let mut timings: HashMap<String, (Duration, usize)> = HashMap::new();
    let t0 = Instant::now();
    for _ in 0..iters {
        let inputs = tvec!(mix.clone().into(), conv.clone(), tra.clone(), inter.clone());
        let eval = |turn: &mut TurnState,
                    op_state: Option<&mut (dyn OpState + 'static)>,
                    node: &Node<TypedFact, Box<dyn TypedOp>>,
                    inputs: TVec<TValue>| {
            let t = Instant::now();
            let r = core_eval(turn, op_state, node, inputs);
            let e = timings.entry(node.op().name().to_string()).or_default();
            e.0 += t.elapsed();
            e.1 += 1;
            r
        };
        let outputs = state.run_plan_with_eval(inputs, eval)?;
        conv = outputs[1].clone();
        tra = outputs[2].clone();
        inter = outputs[3].clone();
    }
    let total = t0.elapsed();

    let mut rows: Vec<_> = timings.into_iter().collect();
    rows.sort_by_key(|(_, (d, _))| std::cmp::Reverse(*d));
    println!("Total: {:.2} ms/frame over {} frames\n", total.as_secs_f64() * 1e3 / iters as f64, iters);
    println!("{:<28} {:>10} {:>8} {:>10}", "op", "ms/frame", "calls", "% time");
    for (name, (d, n)) in rows.iter().take(20) {
        let ms = d.as_secs_f64() * 1e3 / iters as f64;
        let pct = d.as_secs_f64() / total.as_secs_f64() * 100.0;
        println!("{:<28} {:>10.4} {:>8} {:>9.1}%", name, ms, n / iters, pct);
    }
    Ok(())
}
