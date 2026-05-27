//! Graph-rewrite pass: replace constant-index `ScatterNd` nodes (the streaming
//! cache shift-register updates) with a fast custom op.
//!
//! tract's generic `ScatterNd` walks every scattered element through dynamic
//! `ndarray` views, which dominates the per-frame cost. The cache updates here
//! all use *constant* index tensors, so we can precompute flat (dst, src) block
//! offsets once and reduce evaluation to a handful of `copy_from_slice` calls.

use tract_onnx::prelude::*;
use tract_onnx::tract_core::internal::*;
use tract_onnx::tract_core::ops::array::{ScatterNd, ScatterReduction};

/// ScatterND with `reduction=none` and compile-time-constant indices, lowered
/// to contiguous block copies. Runtime inputs are `(data, updates)`.
#[derive(Debug, Clone, PartialEq, Eq)]
struct FastScatterConst {
    /// (dst_base, src_base) flat offsets, one per scattered slice.
    blocks: Vec<(usize, usize)>,
    /// Length of each contiguous copied slice.
    slice_size: usize,
}

impl Op for FastScatterConst {
    fn name(&self) -> StaticName {
        "FastScatterConst".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("{} blocks x {} elems", self.blocks.len(), self.slice_size)])
    }
    op_as_typed_op!();
}

impl EvalOp for FastScatterConst {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (data, updates) = args_2!(inputs);
        ensure!(
            data.datum_type() == f32::datum_type(),
            "FastScatterConst only supports f32, got {:?}",
            data.datum_type()
        );
        let updates = updates.try_as_plain()?;
        let upd = updates.as_slice::<f32>()?;
        let mut data = data.into_tensor();
        let d = unsafe { data.as_slice_mut_unchecked::<f32>() };
        let ss = self.slice_size;
        for &(db, sb) in &self.blocks {
            d[db..db + ss].copy_from_slice(&upd[sb..sb + ss]);
        }
        Ok(tvec!(data.into_tvalue()))
    }
}

impl TypedOp for FastScatterConst {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].datum_type.fact(inputs[0].shape.to_tvec())))
    }
}

/// Precompute the block-copy plan for a ScatterND with constant indices.
///
/// ONNX ScatterND: for each length-`k` index vector (there are `m` of them),
/// write a contiguous slice of `prod(data_shape[k..])` elements into `data`.
fn build_plan(data_shape: &[usize], indices: &Tensor) -> TractResult<FastScatterConst> {
    let indices = indices.cast_to::<i64>()?;
    let idx_shape = indices.shape();
    let k = *idx_shape.last().context("ScatterND indices must be rank >= 1")?;
    let r = data_shape.len();
    ensure!(k <= r, "ScatterND index width {k} exceeds data rank {r}");

    let m: usize = idx_shape[..idx_shape.len() - 1].iter().product();
    let idx = indices.try_as_plain()?;
    let idx = idx.as_slice::<i64>()?;

    let mut stride = vec![1usize; r];
    for i in (0..r.saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * data_shape[i + 1];
    }
    let slice_size: usize = data_shape[k..].iter().product();

    let mut blocks = Vec::with_capacity(m);
    for mm in 0..m {
        let coord = &idx[mm * k..mm * k + k];
        let mut db = 0usize;
        for (j, &c) in coord.iter().enumerate() {
            ensure!(c >= 0, "negative ScatterND index unsupported");
            db += c as usize * stride[j];
        }
        blocks.push((db, mm * slice_size));
    }
    Ok(FastScatterConst { blocks, slice_size })
}

/// Replace every eligible constant-index `ScatterNd` in `model`. Returns the
/// number of nodes rewritten.
pub fn replace_const_scatternd(model: &mut TypedModel) -> TractResult<usize> {
    let mut count = 0;
    let node_ids: Vec<usize> = (0..model.nodes().len()).collect();
    for id in node_ids {
        let node = model.node(id);
        let Some(scatter) = node.op_as::<ScatterNd>() else {
            continue;
        };
        if scatter.reduction != ScatterReduction::None || node.inputs.len() != 3 {
            continue;
        }
        let Some(data_shape) = model.outlet_fact(node.inputs[0])?.shape.as_concrete() else {
            continue;
        };
        let data_shape = data_shape.to_vec();
        let Some(indices) = model.outlet_fact(node.inputs[1])?.konst.clone() else {
            continue;
        };
        let op = build_plan(&data_shape, &indices)?;
        let node = model.node(id).clone();
        let wired = [node.inputs[0], node.inputs[2]];
        let patch = TypedModelPatch::replace_single_op(model, &node, &wired, op)?;
        patch.apply(model)?;
        count += 1;
    }
    if count > 0 {
        model.declutter()?;
    }
    Ok(count)
}
