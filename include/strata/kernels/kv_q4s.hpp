// include/strata/kernels/kv_q4s.hpp - Q4S KV storage (`--kv q4s`): sixteen values per block, searched e4m3 scale.
//
// Companion to q4_0 (kv_q4.hpp): same 256-point Hadamard rotation (fwht256_cuda there, always applied for q4s too),
// same total bytes. q4_0 quantizes 32 values with one max-derived scale and no search; q4s halves the group and
// searches a clipped scale, which is what NVFP4 (per-16, scale search) and SKVQ (clipped MSE search) do - expected
// to cut the measured +8-12% long-document perplexity cost of `--kv q4_0` to near zero at unchanged size.
//
// Block: `struct block_q4s`, 9 bytes - one e4m3 scale plus 16 4-bit codes. Byte j of qs[] holds value 2j in the
// low nibble and value 2j + 1 in the high one; the stored nibble is q + 8 in [0, 15], q in [-8, 7]. Sixteen
// blocks per head of 256 -> 144 B per head per cell, 576 B per cell for K and V of both heads: q4_0's numbers.
//
// The quantizer is DETERMINISTIC and bit-identical on host and device: three candidate scales
// m * {1, 0.9, 0.8} / 7 (m = the group maximum, ties in |x| to the larger value), each rounded to e4m3, and
// the error of a candidate is its squared residuals summed in double over a FIXED pairwise tree (the device's
// shuffle-butterfly order; q4s_quantize_group_host below and q4s_group in kv_q4s.cu carry the same comment).
// The first candidate with strictly smallest error wins, ties to the earlier one. That is what lets the parity
// test memcmp host and device blocks, near-ties included.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cmath>
#include <cstdint>

#if defined(__CUDACC__) || defined(__HIPCC__)
#ifndef STRATA_HD
#define STRATA_HD __host__ __device__
#endif
#else
#ifndef STRATA_HD
#define STRATA_HD
#endif
#endif

namespace strata::kernels {

inline constexpr int QK4S_GROUP = 16;

#pragma pack(push, 1)
struct block_q4s {
    uint8_t d;                    // scale, e4m3 bits (e4m3_decode)
    uint8_t qs[QK4S_GROUP / 2];   // 16 4-bit codes: value 2j in the low nibble of qs[j], 2j + 1 in the high one
};
#pragma pack(pop)

static_assert(sizeof(block_q4s) == 9, "block_q4s must be 9 bytes");

/// Bytes per cell and KV head: 16 blocks of 16 for head_dim 256 -> 144, the same stride as q4_0.
inline uint64_t kv_q4s_bytes_per_head(int head_dim) { return (uint64_t) (head_dim / QK4S_GROUP) * sizeof(block_q4s); }

/// Bytes per cell (one token, one layer): K and V of every KV head.
inline uint64_t kv_q4s_bytes_per_cell(const QsaShapes& s) {
    return (uint64_t) s.n_head_kv * kv_q4s_bytes_per_head((int) s.head_dim) * 2;
}

/// e4m3 with the sign bit always 0 (a scale is never negative): 4 exponent bits, 3 mantissa bits, max normal
/// 448, subnormals on the 2^-9 grid. 0x7f is the one NaN slot and is never produced; it decodes to 0. Both
/// functions are exact (frexp/ldexp move exponents, nothing rounds), so host and device agree on every bit.
STRATA_HD inline float e4m3_decode(uint8_t b) {
    const int e = (b >> 3) & 15, m = (int) (b & 7);
    if (e == 0) return ldexpf((float) m, -9);   // subnormals
    if (e == 15 && m == 7) return 0.0f;         // the NaN slot, never produced
    return ldexpf((float) (8 + m), e - 10);     // (8 + m) * 2^(e-10)
}

/// Round-to-nearest-even float -> int: `lrintf` on the host, `__float2int_rn` on the device - both round halves
/// to even, which is what makes the two sides produce the same codes.
STRATA_HD inline int q4s_rn(float x) {
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
    return __float2int_rn(x);
#else
    return (int) std::lrintf(x);
#endif
}

/// e4m3_encode, x >= 0 only (scales are non-negative; NaN and negatives go to 0).
STRATA_HD inline uint8_t e4m3_encode(float x) {
    if (!(x > 0.0f)) return 0;
    int e;
    (void) frexpf(x, &e);
    int E = e - 1;                                         // x in [2^E, 2^(E+1))
    if (E < -6) {                                          // subnormal, quantum 2^-9
        const int M = q4s_rn(ldexpf(x, 9));
        return (uint8_t) (M > 7 ? 8 : M);                  // 8 = the smallest normal
    }
    if (x > 448.0f) return 0x7E;                           // clamp to the max normal; the NaN slot stays out of reach
    int M = q4s_rn(ldexpf(x, 3 - E));                      // the mantissa 8+m in [8, 16)
    if (M == 16) {
        M = 8;
        if (++E > 8) return 0x7E;
    }
    return (uint8_t) (((E + 7) << 3) | (M - 8));
}

/// The spec-faithful HOST reference for one 16-value group; the device kernel `q4s_group` (kv_q4s.cu) implements
/// the same arithmetic and THE PARITY TEST MEMCPYS THE TWO, so every step here is a contract:
///
/// 1. m = the group maximum of |x|, ties in |x| resolved to the larger value (only |m| feeds the scale, but the
///    rule is what makes every device lane agree on m, as in q4_0's q4_group).
/// 2. Candidates IN THIS ORDER: c in {1.0, 0.9, 0.8}, d_c = e4m3_encode(m * c / 7.0f). The /7 full-range scale
///    represents BOTH extremes exactly (m at q = 7, -m at q = -7), unlike a m/8 signed scale whose positive
///    extreme always clips one full step; the measured reason it wins is in bench/tools/q4s_snr.cpp (-15..-27%
///    RMSE against q4_0 per-32 on gaussian/laplace/spiked draws, where the m/8 grid gains at most ~5%).
/// 3. Per candidate: q_i = clamp(q4s_rn(x_i / d_c), -8, 7) with d_c DECODED from its e4m3 bits, and
///    err_c = sum of (x_i - q_i * d_c)^2, the residual as ONE fused multiply-add (std::fma here, fmaf there),
///    squared in float, the 16 squares summed in double over the FIXED tree written out below. q4s_rn SATURATES
///    out-of-range quotients on both sides (__float2int_rn clamps, PTX cvt.s32.f32 does too), and d_c == 0.0f
///    makes the quotient +-inf for every nonzero x_i - it happens exactly when amax is so small that all three
///    candidate scales round to e4m3 0 (amax below ~0.008). The host spells the saturation out instead of
///    casting lrintf's out-of-range value: +inf -> 7, -inf -> -8, 0/0 -> 0 - the device bytes those conversions
///    produce. The candidate error is unaffected (q_i * d_c is 0 either way); only the stored codes are.
/// 4. The FIRST candidate with strictly smallest err_c wins; ties keep the earlier one. m == 0 encodes as
///    d = 0 and all codes 0.
///
/// THE ERROR TREE, HOST AND DEVICE: the device reduces the 16 squares with a half-warp butterfly over xor
/// offsets 8, 4, 2, 1 - lane i adds lane i^8 first, then i^4, then i^2, then i^1. The loops below add the same
/// operands in that same order (s[0] ends as ((x0+x8)+(x4+x12)) + ((x2+x10)+(x6+x14)) then + lane 1's four),
/// so both sides get the same double and pick the same candidate on near-ties.
inline void q4s_quantize_group_host(const float x[16], uint8_t out[9]) {
    float amax = 0.0f, mval = 0.0f;
    for (int i = 0; i < 16; ++i) {
        const float a = std::fabs(x[i]);
        if (a > amax || (a == amax && x[i] > mval)) { amax = a; mval = x[i]; }
    }
    (void) mval;
    uint8_t d = 0;
    int q[16] = {};
    if (amax > 0.0f) {
        double best = 0.0;
        int best_q[16] = {};
        bool got = false;
        for (int ci = 0; ci < 3; ++ci) {
            const float c = ci == 0 ? 1.0f : (ci == 1 ? 0.9f : 0.8f);
            const uint8_t cb = e4m3_encode(amax * c / 7.0f);
            const float dc = e4m3_decode(cb);
            int cq[16];
            double s[16];
            for (int i = 0; i < 16; ++i) {
                if (dc == 0.0f)   // the quotient is +-inf or 0/0: replicate the device's saturating conversion
                    cq[i] = x[i] > 0.0f ? 7 : (x[i] < 0.0f ? -8 : 0);
                else {
                    cq[i] = q4s_rn(x[i] / dc);
                    if (cq[i] < -8) cq[i] = -8;
                    if (cq[i] > 7) cq[i] = 7;
                }
                const float r = std::fma((float) -cq[i], dc, x[i]);
                s[i] = (double) (r * r);
            }
            // the contract tree, host side (see above; kv_q4s.cu q4s_group is the device half)
            for (int o = 8; o > 0; o >>= 1)
                for (int i = 0; i < o; ++i) s[i] += s[i + o];
            if (!got || s[0] < best) {
                got = true;
                best = s[0];
                d = cb;
                for (int i = 0; i < 16; ++i) best_q[i] = cq[i];
            }
        }
        for (int i = 0; i < 16; ++i) q[i] = best_q[i];
    }
    out[0] = d;
    for (int j = 0; j < QK4S_GROUP / 2; ++j)
        out[1 + j] = (uint8_t) ((q[2 * j] + 8) | ((q[2 * j + 1] + 8) << 4));
}

struct KvHostPools;   // kv_stream.hpp

/// Append the (already rotated) cell at step[kStepPos]. With a host copy (KV streaming) it is written there too,
/// and to VRAM only if its block is resident.
void kv_append_q4s_step(uint8_t* k_q4s, uint8_t* v_q4s, const int32_t* page_table, const int32_t* step,
                        const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                        const KvHostPools* host = nullptr);

void kv_append_q4s_steps(uint8_t* k_q4s, uint8_t* v_q4s, const int32_t* page_table, const int32_t* step,
                         int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                         void* stream, const KvHostPools* host = nullptr);

/// The prompt path: T consecutive (rotated) cells from pos0, K/V [T, n_head_kv, 256]; also into `stage` (identity
/// layout, the one-layer staging pool of a streamed session) when given.
void kv_append_q4s(uint8_t* k_q4s, uint8_t* v_q4s, const int32_t* page_table, int64_t pos0, int64_t T,
                   const float* K, const float* V, const QsaShapes& s, void* stream,
                   const KvHostPools* host = nullptr, const KvHostPools* stage = nullptr);

/// Gather step[kStepWidth] cells named by `ids` into FP16 scratch [id][kv_head][head_dim] (still rotated).
void kv_gather_q4s_step(const uint8_t* k_q4s, const uint8_t* v_q4s, const int32_t* page_table, const int32_t* ids,
                        const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                        uint16_t* v_scratch, void* stream);

}  // namespace strata::kernels
