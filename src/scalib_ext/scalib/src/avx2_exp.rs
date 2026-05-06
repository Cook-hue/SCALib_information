/// AVX2 + FMA vectorised `exp()` for four `f64` lanes.
///
/// # How to use
///
/// 1. Copy this file into your crate (e.g. `src/avx2_exp.rs`).
/// 2. Add `mod avx2_exp;` (or `pub mod avx2_exp;`) to `src/lib.rs` / `src/main.rs`.
/// 3. Build with AVX2 + FMA available, either:
///    - globally: `RUSTFLAGS="-C target-cpu=native" cargo build --release`
///    - per-crate in `.cargo/config.toml`:
///      ```toml
///      [build]
///      rustflags = ["-C", "target-feature=+avx2,+fma"]
///      ```
///    The safe wrapper (`avx2_exp`) performs a runtime CPUID check and will
///    panic on unsupported hardware regardless of compile-time flags.
///
/// # Example
///
/// ```rust
/// use crate::avx2_exp::avx2_exp;
///
/// let result = avx2_exp([0.0, 1.0, 2.0, -1.0]);
/// // ≈ [1.0, 2.718, 7.389, 0.368]
/// ```

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;

// ---------------------------------------------------------------------------
// Taylor-series coefficients and algorithm constants
// ---------------------------------------------------------------------------

/// `log2(e)` — used to convert the base-e argument to base-2.
const LOG2EF: f64 = 1.4426950408889634;

/// Clamping bounds (in base-2 space, i.e. `±709 × log2(e)`).
const THIGH: f64 = 709.0 * LOG2EF;
const TLOW: f64 = -709.0 * LOG2EF;

/// Floating-point magic constant for the fast `double → int64` exponent trick.
/// Adding this to a double whose integer part fits in 52 bits moves that integer
/// into the mantissa, where it can be extracted with an integer add + shift.
const MAGIC_LONG_DOUBLE_ADD: f64 = 6755399441055744.0;

/// IEEE 754 double exponent bias.
const EXP_BIAS: i64 = 1023;

/// Minimax Taylor coefficients for `exp2(x)` on `(-0.5, 0.5)`.
/// Even-indexed terms go into the `y` accumulator, odd-indexed into `yo`.
const T0: f64 = 1.0;
const T1: f64 = 0.6931471805599453087156032;
const T2: f64 = 0.240226506959101195979507231;
const T3: f64 = 0.05550410866482166557484;
const T4: f64 = 0.00961812910759946061829085;
const T5: f64 = 0.0013333558146398846396;
const T6: f64 = 0.0001540353044975008196326;
const T7: f64 = 0.000015252733847608224;
const T8: f64 = 0.000001321543919937730177;
const T9: f64 = 0.00000010178055034703;
const T10: f64 = 0.000000007073075504998510;
const T11: f64 = 0.00000000044560630323;

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

/// Masked blend: returns `if_true` where `mask` is all-ones, `if_false` elsewhere.
/// Corresponds to `Util.IfElse` in the original C# code.
///
/// # Safety
/// Requires AVX.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
#[inline]
unsafe fn blend(mask: __m256d, if_true: __m256d, if_false: __m256d) -> __m256d {
    // _mm256_blendv_pd(a, b, mask) returns b where mask high-bit is 1, a otherwise.
    _mm256_blendv_pd(if_false, if_true, mask)
}

// ---------------------------------------------------------------------------
// Core kernel: exp2 on a pre-scaled argument
// ---------------------------------------------------------------------------

/// Computes `exp2(x)` (i.e. `2^x`) for four `f64` lanes.
///
/// The argument **must already be in base-2 space** (`x × log2(e)` for a
/// base-e exponential).  Clamping, NaN propagation, and the `+inf` sentinel
/// are all handled internally.
///
/// Matches the C# method:
/// ```csharp
/// public static void Two(in Vector256<double> x, ref Vector256<double> y)
/// ```
///
/// # Safety
/// Requires AVX, AVX2, and FMA CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
unsafe fn exp2_avx2(x: __m256d) -> __m256d {
    // --- broadcast scalar constants ---
    let thigh = _mm256_set1_pd(THIGH);
    let tlow = _mm256_set1_pd(TLOW);
    let magic = _mm256_set1_pd(MAGIC_LONG_DOUBLE_ADD);
    let pos_inf = _mm256_set1_pd(f64::INFINITY);
    let nan_vec = _mm256_set1_pd(f64::NAN);
    let i1023 = _mm256_set1_epi64x(EXP_BIAS);

    let t0 = _mm256_set1_pd(T0);
    let t1 = _mm256_set1_pd(T1);
    let t2 = _mm256_set1_pd(T2);
    let t3 = _mm256_set1_pd(T3);
    let t4 = _mm256_set1_pd(T4);
    let t5 = _mm256_set1_pd(T5);
    let t6 = _mm256_set1_pd(T6);
    let t7 = _mm256_set1_pd(T7);
    let t8 = _mm256_set1_pd(T8);
    let t9 = _mm256_set1_pd(T9);
    let t10 = _mm256_set1_pd(T10);
    let t11 = _mm256_set1_pd(T11);

    // --- clamp to [TLOW, THIGH] ---
    let mut xx = _mm256_max_pd(_mm256_min_pd(x, thigh), tlow);

    // --- fx = round(xx) to nearest integer, kept as f64 ---
    let fx = _mm256_round_pd(xx, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);

    // --- reduce: xx -= fx  =>  xx in (-0.5, 0.5) ---
    xx = _mm256_sub_pd(xx, fx);
    let xsq = _mm256_mul_pd(xx, xx);

    // --- Estrin evaluation of exp2 Taylor series ---
    // Even-degree terms accumulate in `y`, odd-degree in `yo`,
    // then they are combined at the end with a single FMA.
    let mut y = _mm256_fmadd_pd(t11, xsq, t9);
    let mut yo = _mm256_fmadd_pd(t10, xsq, t8);
    y = _mm256_fmadd_pd(y, xsq, t7);
    yo = _mm256_fmadd_pd(yo, xsq, t6);
    y = _mm256_fmadd_pd(y, xsq, t5);
    yo = _mm256_fmadd_pd(yo, xsq, t4);
    y = _mm256_fmadd_pd(y, xsq, t3);
    yo = _mm256_fmadd_pd(yo, xsq, t2);
    y = _mm256_fmadd_pd(y, xsq, t1);
    yo = _mm256_fmadd_pd(yo, xsq, t0);
    // Combine: y*x + yo
    y = _mm256_fmadd_pd(y, xx, yo);

    // --- fast 2^n via IEEE 754 exponent-bias trick ---
    //
    // Adding MAGIC shifts the integer part of fx into the mantissa field so
    // that a plain i64 add of the exponent bias, followed by a 52-bit left
    // shift, produces a valid IEEE 754 double equal to 2^round(x).
    let fx_magic = _mm256_add_pd(fx, magic);
    let fx_i64 = _mm256_castpd_si256(fx_magic); // bitcast, no conversion
    let fx_biased = _mm256_add_epi64(fx_i64, i1023);
    let fx_shift = _mm256_slli_epi64(fx_biased, 52);
    let pow2n = _mm256_castsi256_pd(fx_shift); // bitcast back

    y = _mm256_mul_pd(pow2n, y);

    // --- special-case handling ---

    // x >= THIGH  →  +inf
    // (_CMP_GE_OQ: ordered, quiet — NaN inputs yield false, which is correct
    //  since NaN is handled in the next step.)
    let cmp_high = _mm256_cmp_pd(x, thigh, _CMP_GE_OQ);
    y = blend(cmp_high, pos_inf, y);

    // x is NaN  →  NaN
    // The only value where `v != v` is NaN (IEEE 754 §5.11).
    let cmp_nan = _mm256_cmp_pd(x, x, _CMP_NEQ_UQ);
    y = blend(cmp_nan, nan_vec, y);

    y
}

// ---------------------------------------------------------------------------
// Public unsafe API
// ---------------------------------------------------------------------------

/// Computes `exp(x)` for four `f64` lanes (`__m256d`).
///
/// Scales the argument by `log2(e)` then delegates to [`exp2_avx2`].
///
/// Matches the C# method:
/// ```csharp
/// public static void Exp(in Vector256<double> x, ref Vector256<double> y)
/// ```
///
/// # Safety
/// Requires AVX, AVX2, and FMA CPU features.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx,avx2,fma")]
pub unsafe fn exp_avx2(x: __m256d) -> __m256d {
    let log2ef = _mm256_set1_pd(LOG2EF);
    let xx = _mm256_mul_pd(x, log2ef);
    exp2_avx2(xx)
}

// ---------------------------------------------------------------------------
// Public safe API
// ---------------------------------------------------------------------------

/// Computes `exp(x)` for four `f64` values using AVX2 + FMA SIMD.
///
/// This is the safe, `#[inline]`-friendly entry point.  It performs a runtime
/// CPUID check and panics on CPUs that do not support AVX2 or FMA.
///
/// For hot loops, prefer calling [`exp_avx2`] directly inside a
/// `#[target_feature(enable = "avx,avx2,fma")]` context to avoid the repeated
/// feature check and potential function-call overhead.
///
/// # Panics
/// Panics if the CPU does not support AVX2 and FMA.
///
/// # Example
/// ```rust
/// let y = avx2_exp([0.0, 1.0, 2.0, -1.0]);
/// assert!((y[0] - 1.0_f64).abs() < 1e-12);
/// assert!((y[1] - std::f64::consts::E).abs() < 1e-12);
/// ```
#[cfg(target_arch = "x86_64")]
pub fn avx2_exp(x: [f64; 4]) -> [f64; 4] {
    assert!(
        is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"),
        "avx2_exp: this CPU does not support AVX2 + FMA"
    );
    // SAFETY: feature support verified above; unaligned load/store are safe for
    // any stack-allocated array regardless of alignment.
    unsafe {
        let input = _mm256_loadu_pd(x.as_ptr());
        let result = exp_avx2(input);
        let mut out = [0.0_f64; 4];
        _mm256_storeu_pd(out.as_mut_ptr(), result);
        out
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// Relative error tolerance — the Taylor series is accurate to ~1 ULP.
    const TOL: f64 = 1e-13;

    fn check(input: f64, expected: f64) {
        let [y, ..] = avx2_exp([input, 0.0, 0.0, 0.0]);
        let rel_err = (y - expected).abs() / expected.abs().max(1e-300);
        assert!(
            rel_err < TOL,
            "exp({input}) = {y}, expected {expected}, rel_err = {rel_err:.2e}"
        );
    }

    #[test]
    fn exp_zero() {
        check(0.0, 1.0);
    }

    #[test]
    fn exp_one() {
        check(1.0, std::f64::consts::E);
    }

    #[test]
    fn exp_minus_one() {
        check(-1.0, 1.0 / std::f64::consts::E);
    }

    #[test]
    fn exp_two() {
        check(2.0, std::f64::consts::E * std::f64::consts::E);
    }

    #[test]
    fn exp_large_clamps_to_infinity() {
        let [y, ..] = avx2_exp([1e300, 0.0, 0.0, 0.0]);
        assert!(y.is_infinite() && y > 0.0);
    }

    #[test]
    fn exp_very_negative_is_zero() {
        let [y, ..] = avx2_exp([-1e300, 0.0, 0.0, 0.0]);
        assert_eq!(y, 0.0);
    }

    #[test]
    fn exp_nan_propagates() {
        let [y, ..] = avx2_exp([f64::NAN, 0.0, 0.0, 0.0]);
        assert!(y.is_nan());
    }

    #[test]
    fn all_four_lanes_independent() {
        let inputs = [0.0, 1.0, 2.0, -1.0];
        let results = avx2_exp(inputs);
        let reference: [f64; 4] = inputs.map(f64::exp);
        for (r, e) in results.iter().zip(reference.iter()) {
            let rel_err = (r - e).abs() / e.abs().max(1e-300);
            assert!(rel_err < TOL, "rel_err = {rel_err:.2e}");
        }
    }
}
