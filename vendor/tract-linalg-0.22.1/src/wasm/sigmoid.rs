//! WASM SIMD128 implementation of the `sigmoid_f32` element-wise kernel.
//!
//! `WSigmoid4` is a vectorised drop-in for `crate::generic::sigmoid::SSigmoid4`.
//! It evaluates the same `P(x²) · x / Q(x²) + 0.5` rational polynomial with the
//! same 12 constants and the same Horner evaluation order, just on
//! `core::arch::wasm32::v128` lanes instead of scalar `f32`.
//!
//! Bit-identity to `SSigmoid4` is the load-bearing property. Every
//! multiply-accumulate is an explicit `f32x4_add(f32x4_mul(a, b), c)` —
//! `f32x4_relaxed_madd` is **not** used because relaxed FMA can change
//! intermediate precision in ways that defeat the bit-identity guarantee
//! against the scalar reference (whose accumulation has no fused round).
//!
//! Polynomial constants ported verbatim from
//! `linalg/src/generic/sigmoid.rs::ssigmoid` (commit 057e18e26).

#![cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
#![allow(clippy::excessive_precision)]

use std::arch::wasm32::*;

use crate::frame::element_wise::ElementWiseKer;

#[derive(Clone, Debug)]
pub struct WSigmoid4;

impl ElementWiseKer<f32> for WSigmoid4 {
    fn name() -> &'static str {
        "wasm_simd128"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    fn run(x: &mut [f32], _: ()) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);

        // SAFETY: caller guarantees a 16-byte aligned slice with len % 4 == 0
        // (see debug_asserts and the contract of `ElementWiseKer::run`). All
        // `v128_load` / `v128_store` reads and writes stay strictly within the
        // slice's bounds because we step `i` by 4 elements at a time.
        unsafe {
            let mut p = x.as_mut_ptr() as *mut v128;
            let end = p.add(x.len() / 4);
            while p < end {
                let v = v128_load(p);
                v128_store(p, sigmoid_f32x4(v));
                p = p.add(1);
            }
        }
    }
}

/// Apply the sigmoid approximation lane-wise to a single `f32x4` vector.
///
/// Polynomial constants and evaluation order match
/// `crate::generic::sigmoid::ssigmoid` exactly. Tier-1 bit-identity test
/// (`bit_identity_vs_ssigmoid_*`) verifies this at every release.
#[inline(always)]
unsafe fn sigmoid_f32x4(x: v128) -> v128 {
    // Clamp domain — outside ±18.6 the polynomial is bit-flat at the
    // asymptote anyway, and clamping protects the squared term from f32
    // overflow on extreme inputs.
    const LOW: f32 = -18.6;
    const HIGH: f32 = -LOW;

    // Numerator coefficients of P(x²).
    const ALPHA_13: f32 = -4.433153405e-18;
    const ALPHA_11: f32 = 1.169974371e-14;
    const ALPHA_9: f32 = -1.875289645e-11;
    const ALPHA_7: f32 = 4.257889523e-8;
    const ALPHA_5: f32 = 0.00004811817576;
    const ALPHA_3: f32 = 0.008163842030;
    const ALPHA_1: f32 = 0.2499999971;

    // Denominator coefficients of Q(x²).
    const BETA_6: f32 = 3.922935744e-6;
    const BETA_4: f32 = 0.001524872358;
    const BETA_2: f32 = 0.1159886749;
    const BETA_0: f32 = 1.0;

    let x = f32x4_max(f32x4_splat(LOW), f32x4_min(f32x4_splat(HIGH), x));
    let x2 = f32x4_mul(x, x);

    // P(x²) · x — Horner on x², then a single multiply by x.
    let p = f32x4_splat(ALPHA_13);
    let p = f32x4_add(f32x4_mul(p, x2), f32x4_splat(ALPHA_11));
    let p = f32x4_add(f32x4_mul(p, x2), f32x4_splat(ALPHA_9));
    let p = f32x4_add(f32x4_mul(p, x2), f32x4_splat(ALPHA_7));
    let p = f32x4_add(f32x4_mul(p, x2), f32x4_splat(ALPHA_5));
    let p = f32x4_add(f32x4_mul(p, x2), f32x4_splat(ALPHA_3));
    let p = f32x4_add(f32x4_mul(p, x2), f32x4_splat(ALPHA_1));
    let p = f32x4_mul(p, x);

    // Q(x²) — Horner on x².
    let q = f32x4_splat(BETA_6);
    let q = f32x4_add(f32x4_mul(q, x2), f32x4_splat(BETA_4));
    let q = f32x4_add(f32x4_mul(q, x2), f32x4_splat(BETA_2));
    let q = f32x4_add(f32x4_mul(q, x2), f32x4_splat(BETA_0));

    f32x4_add(f32x4_div(p, q), f32x4_splat(0.5))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::generic::sigmoid::{SSigmoid4, ssigmoid};

    /// Round `n` up to a multiple of `WSigmoid4::nr()` (= 4) and pad with
    /// zeros so we satisfy the `ElementWiseKer` precondition.
    fn pad_to_nr(mut v: Vec<f32>) -> Vec<f32> {
        while v.len() % WSigmoid4::nr() != 0 {
            v.push(0.0);
        }
        v
    }

    fn run_w(values: &[f32]) -> Vec<f32> {
        let mut out = pad_to_nr(values.to_vec());
        WSigmoid4::ew().run(&mut out).unwrap();
        out
    }

    fn run_s(values: &[f32]) -> Vec<f32> {
        let mut out = pad_to_nr(values.to_vec());
        SSigmoid4::ew().run(&mut out).unwrap();
        out
    }

    /// Tier-1 bit-identity: same polynomial constants, same Horner order,
    /// same `f32x4_mul` + `f32x4_add` sequence (no relaxed FMA) → byte-for-byte
    /// equal output to `SSigmoid4` for every input that matters.
    #[test]
    fn bit_identity_vs_ssigmoid_dense() {
        let mut xs = Vec::with_capacity(1024);
        for i in 0..1024 {
            // 1024 evenly-spaced points across [-20, 20].
            xs.push(-20.0 + (i as f32) * (40.0 / 1023.0));
        }
        let w = run_w(&xs);
        let s = run_s(&xs);
        let mut max_abs_err = 0f32;
        for (a, b) in w.iter().zip(s.iter()) {
            max_abs_err = max_abs_err.max((a - b).abs());
        }
        assert_eq!(max_abs_err, 0.0, "WSigmoid4 ≠ SSigmoid4 on dense [-20, 20] sweep");
    }

    /// Tier-1 bit-identity at the saturation asymptotes and ±0.
    #[test]
    fn bit_identity_edge_cases() {
        // ssigmoid clamps to ±18.6; outside that range every input maps to a
        // single asymptotic value. Inputs at ±18.6 and ±0 must match exactly.
        let inputs = [
            -18.6, 18.6, -1e6, 1e6, -88.0, 88.0, 0.0, -0.0, -100.0, 100.0,
        ];
        let w = run_w(&inputs);
        let s = run_s(&inputs);
        for (i, (a, b)) in w.iter().zip(s.iter()).enumerate() {
            assert_eq!(
                a.to_bits(),
                b.to_bits(),
                "WSigmoid4 vs SSigmoid4 differs at input[{i}] = {} (ssigmoid({}) = {b}, wsigmoid = {a})",
                inputs[i],
                inputs[i],
            );
        }
    }

    /// Domain coverage: 100k pseudo-random samples in [-88, 88] (the
    /// f32-safe input range; outside this range `(-x).exp()` overflows
    /// in the scalar reference but the polynomial path clamps first).
    #[test]
    fn bit_identity_domain_coverage_100k() {
        // Simple LCG so the test is deterministic and seed-fixed without
        // pulling in a randomness crate. Same scheme as the rest of
        // tract-linalg's deterministic test fixtures.
        let mut s: u64 = 0xC0FFEE_DEADBEEFu64;
        let mut xs = Vec::with_capacity(100_000);
        for _ in 0..100_000 {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            // 24-bit fraction → f32 in [0, 1)
            let bits = ((s >> 40) & 0x00FFFFFF) as u32;
            let unit = (bits as f32) / ((1u32 << 24) as f32);
            // Scale to [-88, 88]
            xs.push(unit * 176.0 - 88.0);
        }
        let w = run_w(&xs);
        let s_ = run_s(&xs);
        let mut max_abs_err = 0f32;
        let mut first_diff: Option<(usize, f32, f32, f32)> = None;
        for (i, (a, b)) in w.iter().zip(s_.iter()).enumerate() {
            let e = (a - b).abs();
            if e > max_abs_err {
                max_abs_err = e;
            }
            if first_diff.is_none() && a.to_bits() != b.to_bits() {
                first_diff = Some((i, xs[i], *a, *b));
            }
        }
        if let Some((i, x, a, b)) = first_diff {
            panic!(
                "WSigmoid4 ≠ SSigmoid4 on 100k random sweep — first diff at index {i}: \
                 x={x}, wsigmoid={a} (bits {:#010x}), ssigmoid={b} (bits {:#010x}); \
                 max_abs_err over the sweep = {max_abs_err:e}",
                a.to_bits(),
                b.to_bits(),
            );
        }
    }

    /// Sanity-check the polynomial against the textbook sigmoid for the
    /// well-conditioned region. Tighter than tract's own
    /// `Approximation::Close` to catch any drift in the SIMD path.
    #[test]
    fn close_to_textbook_sigmoid_central_region() {
        let xs: Vec<f32> = (-50..=50).map(|i| (i as f32) / 5.0).collect(); // step 0.2 over [-10, 10]
        let w = run_w(&xs);
        let mut max_abs_err = 0f32;
        for (x, w) in xs.iter().zip(w.iter()) {
            let exact = 1.0 / (1.0 + (-*x).exp());
            max_abs_err = max_abs_err.max((w - exact).abs());
        }
        // Same tolerance the reference polynomial achieves; the published
        // accuracy is ~5e-7 in the central region.
        assert!(
            max_abs_err < 1e-5,
            "WSigmoid4 vs textbook sigmoid on [-10, 10] step 0.2: max_abs_err = {max_abs_err:e}"
        );
    }

    /// The vectorised path must match the scalar reference on every chunk
    /// boundary (we hit `nr=4` chunks; misalignment would manifest there).
    #[test]
    fn bit_identity_at_chunk_boundaries() {
        for n in [4, 8, 12, 16, 20, 32, 64, 100, 256, 480] {
            let xs: Vec<f32> = (0..n).map(|i| (i as f32) * 0.1 - (n as f32) * 0.05).collect();
            let w = run_w(&xs);
            let s = run_s(&xs);
            for (i, (a, b)) in w.iter().zip(s.iter()).enumerate() {
                assert_eq!(
                    a.to_bits(),
                    b.to_bits(),
                    "n={n}: differs at element {i}, x={}, wsigmoid={a} ssigmoid={b}",
                    xs.get(i).copied().unwrap_or(0.0),
                );
            }
        }
    }

    /// Cross-check the inner SIMD function against `ssigmoid` directly,
    /// bypassing the `ElementWiseKer` wrapper. Fails fast if the polynomial
    /// itself drifts.
    #[test]
    fn sigmoid_f32x4_matches_scalar_per_lane() {
        let lanes = [-3.5_f32, -0.25, 0.25, 4.5];
        let v = f32x4(lanes[0], lanes[1], lanes[2], lanes[3]);
        let r = unsafe { sigmoid_f32x4(v) };
        let extracted = [
            f32x4_extract_lane::<0>(r),
            f32x4_extract_lane::<1>(r),
            f32x4_extract_lane::<2>(r),
            f32x4_extract_lane::<3>(r),
        ];
        for (i, (lane, got)) in lanes.iter().zip(extracted.iter()).enumerate() {
            let expected = ssigmoid(*lane);
            assert_eq!(
                got.to_bits(),
                expected.to_bits(),
                "lane {i}: ssigmoid({lane}) = {expected}, sigmoid_f32x4 = {got}",
            );
        }
    }
}
