// src/kernels/kv_q4s_parity.cpp - Parity test for the Q4S KV cache (`--kv q4s`, kv_q4s.hpp): q4_0's rotation and
// footprint with sixteen values per block and a searched e4m3 scale.
// Validates:
// 1. The e4m3 scale codec: decode of all 128 sign-0 byte values against the spec table (bitwise), encode/decode
//    roundtrip over a dense sweep (every representable value, every exact midpoint tie, their float neighbors, a
//    grid over thirty binades and randoms) against the nearest-representable-value rule with half-even ties, the
//    448 clamp, and proof the NaN slot 0x7f is never produced.
// 2. The search: a crafted peaked group (one -9.0 outlier among +-1.6875) whose winning candidate is c = 0.8 -
//    scale 0x00 (1.0), not the full-range c = 1.0 scale 0x02 (1.25) - asserted on the host reference and
//    memcmp'd against the device block. The bulk sits nearer the 1.0-step grid (residual 0.0977 each) than the
//    1.25-step one (0.1914 each), which pays for the outlier's one-step clamp.
// 3. Q4S quantization through a paged pool - kv_append_q4s_step, the batched kv_append_q4s_steps and the prompt
//    path kv_append_q4s - BITWISE vs q4s_quantize_group_host on random, tiny, all-zero and peaked groups.
// 4. kv_gather_q4s_step vs a host dequant that replicates the kernel's f16_from_f32(q * d), bitwise.

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q4s.hpp"
#include "strata/kernels/qsa.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// Exact CPU reference for the orthonormal Walsh-Hadamard transform of 256 floats (scale 1/16), as in kv_q4_parity:
// the appends below take already-rotated cells, so the random fixtures are rotated before upload.
void fwht256_host_reference(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2) {
        for (int i = 0; i < n; i += 2 * s) {
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j];
                const float v = x[i + j + s];
                x[i + j]     = u + v;
                x[i + j + s] = u - v;
            }
        }
    }
    for (int i = 0; i < n; ++i) x[i] *= 1.0f / 16.0f;
}

// The spec table, written independently of e4m3_decode: subnormals on the 2^-9 grid, the NaN slot decoding to 0,
// normals as (8 + m) * 2^(e-10) through a double power of two - exact, so the comparison can be bitwise.
void spec_table(float t[128]) {
    for (int b = 0; b < 128; ++b) {
        const int e = (b >> 3) & 15, m = b & 7;
        if (e == 0) t[b] = (float) std::ldexp((double) m, -9);
        else if (e == 15 && m == 7) t[b] = 0.0f;
        else t[b] = (float) ((8 + m) * std::pow(2.0, e - 10));
    }
}

// What e4m3_encode MEANS: the nearest value of the spec table, an exact tie going to the even mantissa nibble
// (the byte's low bit), 0 for nothing positive and the max normal 448 for everything above it.
uint8_t nearest_byte(const float t[128], float x) {
    if (!(x > 0.0f)) return 0;
    const float inf = std::numeric_limits<float>::infinity();
    if (!(x < inf) || x > 448.0f) return 0x7E;
    int best = -1;
    float best_d = 0.0f;
    for (int b = 0; b < 127; ++b) {
        const float d = std::fabs(t[b] - x);
        if (best < 0 || d < best_d || (d == best_d && (b & 1) == 0)) {
            best = b;
            best_d = d;
        }
    }
    return (uint8_t) best;
}

// The peaked fixture of test 2: amax 9, so the candidates take d = e4m3(9/7) = 0x02 (1.25), e4m3(0.9*9/7) = 0x01
// (1.125), e4m3(0.8*9/7) = 0x00 (1.0). The +-1.6875 bulk is nearer the 1.0-step grid (+-2, residual 0.0977 each)
// than to +-1 at 1.25 (0.1914 each), and the outlier's clamp residual at d = 1.0 is 1.0: the c = 0.8 candidate
// wins, 2.4648 against 2.9336 for the full-range c = 1.0.
void peaked_group(float x[16]) {
    for (int i = 0; i < 16; ++i) x[i] = i == 0 ? -9.0f : (i % 2 ? 1.6875f : -1.6875f);
}

void print_block(const char* side, const uint8_t* blk) {
    std::printf("  %s d = 0x%02x, qs:", side, blk[0]);
    for (int i = 0; i < 8; ++i) std::printf(" %02x", blk[1 + i]);
    std::printf("\n");
}

} // namespace

int main() {
    std::printf("=== Running kv_q4s_parity test ===\n");
    std::mt19937 rng(42);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    // -------------------------------------------------------------
    // Test 1: the e4m3 scale codec against the spec table
    // -------------------------------------------------------------
    std::printf("[1/4] Verifying the e4m3 codec: decode table, roundtrip, 448 clamp, NaN slot...\n");
    float spec[128];
    spec_table(spec);

    int bad_decode = 0;
    for (int b = 0; b < 128; ++b) {
        const float got = k::e4m3_decode((uint8_t) b);
        if (std::memcmp(&got, &spec[b], sizeof(float)) != 0) {
            if (bad_decode == 0)
                std::fprintf(stderr, "FAIL: e4m3_decode(0x%02x) = %.9g, spec says %.9g\n", b, got, spec[b]);
            ++bad_decode;
        }
    }
    if (bad_decode > 0) {
        std::fprintf(stderr, "FAIL: %d of 128 e4m3_decode values differ from the spec table\n", bad_decode);
        ++g_fail;
    } else {
        std::printf("  -> e4m3_decode bitwise-equal to the spec table over all 128 sign-0 bytes (OK)\n");
    }

    // the dense sweep: every representable value, every exact midpoint (a half-even tie) and the floats on both
    // sides of it, their neighbors, a dense grid over the binades from 2^-30 to 2^10 (subnormals included), randoms
    // and the fixed extremes around the 448 clamp.
    std::vector<float> sweep;
    for (int b = 0; b < 127; ++b) {
        sweep.push_back(spec[b]);
        if (b + 1 < 127) {
            const float mid = (spec[b] + spec[b + 1]) * 0.5f;   // exact: the neighbors differ by one quantum
            const float inf = std::numeric_limits<float>::infinity();
            sweep.push_back(mid);
            sweep.push_back(std::nextafter(mid, 0.0f));
            sweep.push_back(std::nextafter(mid, inf));
            sweep.push_back(std::nextafter(spec[b], inf));
            sweep.push_back(std::nextafter(spec[b + 1], 0.0f));
        }
    }
    for (int kk = -30; kk <= 10; ++kk)
        for (int j = 0; j < 64; ++j) sweep.push_back(std::ldexp(1.0f + j / 64.0f, kk));
    for (int i = 0; i < 20000; ++i) sweep.push_back(std::uniform_real_distribution<float>(0.0f, 600.0f)(rng));
    const float extremes[] = {1e-30f, 1e-9f, 2e-6f, 0.0156f, 0.0157f, 447.9f, 448.0f, 449.0f,
                              460.0f, 464.0f, 480.0f, 496.0f, 1e30f, 3e38f};
    sweep.insert(sweep.end(), extremes, extremes + sizeof(extremes) / sizeof(extremes[0]));

    int bad_rt = 0, bad_idem = 0, bad_nan = 0;
    for (const float x : sweep) {
        const uint8_t e = k::e4m3_encode(x);
        if (e == 0x7F) ++bad_nan;                                    // the NaN slot, never produced
        const uint8_t want = nearest_byte(spec, x);
        const float back = k::e4m3_decode(e);
        if (e != want || std::memcmp(&back, &spec[e], sizeof(float)) != 0) {
            if (bad_rt == 0)
                std::fprintf(stderr, "FAIL: e4m3_encode(%.9g) = 0x%02x (%.9g), nearest spec value is 0x%02x (%.9g)\n",
                             x, e, k::e4m3_decode(e), want, spec[want]);
            ++bad_rt;
        }
    }
    for (int b = 0; b < 127; ++b)
        if (k::e4m3_encode(spec[b]) != (uint8_t) b) ++bad_idem;      // a representable value encodes to itself
    if (k::e4m3_decode(0x7F) != 0.0f) ++bad_nan;
    if (bad_rt > 0 || bad_idem > 0) {
        std::fprintf(stderr, "FAIL: %d of %zu sweep values and %d table values misround\n", bad_rt, sweep.size(),
                     bad_idem);
        ++g_fail;
    } else {
        std::printf("  -> encode/decode roundtrip bitwise over %zu values, %d idempotence checks (OK)\n",
                    sweep.size(), 127);
    }
    if (bad_nan > 0) {
        std::fprintf(stderr, "FAIL: the NaN slot 0x7f was produced %d times / decodes nonzero\n", bad_nan);
        ++g_fail;
    } else {
        std::printf("  -> the NaN slot 0x7f is never produced and decodes to 0; 448 clamp holds (OK)\n");
    }

    // -------------------------------------------------------------
    // Test 2: the scale search picks the c = 0.8 candidate on a peaked group
    // -------------------------------------------------------------
    std::printf("[2/4] Verifying the searched scale on a peaked group vs the full-range rule...\n");
    float peaked[16];
    peaked_group(peaked);
    uint8_t ref[9];
    k::q4s_quantize_group_host(peaked, ref);
    if (k::e4m3_encode(9.0f / 7.0f) != 0x02 || k::e4m3_encode(9.0f * 0.8f / 7.0f) != 0x00) {
        std::fprintf(stderr, "FAIL: the fixture's scale bytes are wrong: full-range 0x%x, searched 0x%x\n",
                     k::e4m3_encode(9.0f / 7.0f), k::e4m3_encode(9.0f * 0.8f / 7.0f));
        ++g_fail;
    }
    if (ref[0] != 0x00) {
        std::fprintf(stderr, "FAIL: the host reference kept the full-range scale 0x%02x, the search must pick 0x00\n",
                     ref[0]);
        ++g_fail;
    } else {
        std::printf("  -> the search picks c = 0.8 (d = 0x00) over the full-range 0x02 (OK)\n");
    }

    // the same group through the device: one cell whose head-0 block 0 is the peaked group (block 3 and 12 cover
    // the other warp halves), everything else random - the device block must match the host reference bitwise
    {
        k::QsaShapes s = k::qsa_real_shapes();
        s.page_size = 64;
        const int H = (int) s.n_head_kv, D = (int) s.head_dim;
        const int pages = 2, cells = pages * (int) s.page_size;
        std::vector<int32_t> table(pages);
        for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 5) % pages;
        int32_t* d_table = dalloc<int32_t>(pages);
        ck(cudaMemcpy(d_table, table.data(), pages * 4, cudaMemcpyHostToDevice), "memcpy table");
        const size_t pool_bytes = (size_t) pages * H * (unsigned) s.page_size * k::kv_q4s_bytes_per_head(D);
        uint8_t* d_k_q4s = dalloc<uint8_t>(pool_bytes);
        uint8_t* d_v_q4s = dalloc<uint8_t>(pool_bytes);
        float* d_kcur = dalloc<float>(H * D);
        float* d_vcur = dalloc<float>(H * D);
        int32_t* d_step = dalloc<int32_t>(k::kStepCount);

        std::vector<float> hk(H * D), hv(H * D);
        for (auto& x : hk) x = nd(rng);
        for (auto& x : hv) x = nd(rng);
        for (int blk : {0, 3, 12}) {   // warp halves 0, 1 and 0 of the second warp pair
            float g[16];
            peaked_group(g);
            std::copy(g, g + 16, hk.begin() + blk * k::QK4S_GROUP);
            peaked_group(g);
            std::copy(g, g + 16, hv.begin() + blk * k::QK4S_GROUP);
        }
        const int pos = 37;
        const int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, hk.data(), hk.size() * 4, cudaMemcpyHostToDevice), "memcpy k");
        ck(cudaMemcpy(d_vcur, hv.data(), hv.size() * 4, cudaMemcpyHostToDevice), "memcpy v");
        k::kv_append_q4s_step(d_k_q4s, d_v_q4s, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "kv_append_q4s_step");

        std::vector<uint8_t> h_k(pool_bytes), h_v(pool_bytes);
        ck(cudaMemcpy(h_k.data(), d_k_q4s, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4s");
        ck(cudaMemcpy(h_v.data(), d_v_q4s, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4s");
        long bad_blocks = 0;
        const int blocks_per_head = D / k::QK4S_GROUP;
        const uint64_t bytes_per_head = k::kv_q4s_bytes_per_head(D);
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = table[pos / (int) s.page_size];
                const long long row = (page * H + h) * (int) s.page_size + (pos % (int) s.page_size);
                const uint8_t* pool = (is_v ? h_v : h_k).data() + (size_t) row * bytes_per_head;
                const float* raw = (is_v ? hv : hk).data() + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    uint8_t host_blk[9];
                    k::q4s_quantize_group_host(raw + b * k::QK4S_GROUP, host_blk);
                    if (std::memcmp(host_blk, pool + (size_t) b * 9, 9) != 0) {
                        if (bad_blocks == 0) {
                            std::printf("MISMATCH at is_v=%d, h=%d, b=%d\n", is_v, h, b);
                            print_block("host", host_blk);
                            print_block("gpu ", pool + (size_t) b * 9);
                        }
                        ++bad_blocks;
                    }
                }
            }
        }
        if (bad_blocks > 0) {
            std::fprintf(stderr, "FAIL: %ld device blocks differ on the peaked cell\n", bad_blocks);
            ++g_fail;
        } else {
            std::printf("  -> the device reproduces the searched-scale blocks bitwise, all warp halves (OK)\n");
        }
        cudaFree(d_table);
        cudaFree(d_k_q4s);
        cudaFree(d_v_q4s);
        cudaFree(d_kcur);
        cudaFree(d_vcur);
        cudaFree(d_step);
    }

    // -------------------------------------------------------------
    // Test 3: paged append - per-step, batched and prompt paths - bitwise vs the host quantizer
    // -------------------------------------------------------------
    std::printf("[3/4] Verifying Q4S append (step, steps, batch) against q4s_quantize_group_host...\n");
    k::QsaShapes s = k::qsa_real_shapes();
    s.page_size = 64;
    const int H = (int) s.n_head_kv;   // 2
    const int D = (int) s.head_dim;    // 256
    const int P = (int) s.page_size;   // 64
    const int pages = 8;
    const int cells = pages * P;

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = (i * 3 + 5) % pages;   // permutation
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy table");

    const size_t pool_bytes = (size_t) pages * H * P * k::kv_q4s_bytes_per_head(D);
    uint8_t* d_k_q4s = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v_q4s = dalloc<uint8_t>(pool_bytes);
    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    // 250 scattered cells through kv_append_q4s_step, 80 scattered cells through ONE kv_append_q4s_steps call,
    // the contiguous run [200, 280) through ONE kv_append_q4s call.
    std::vector<int> positions(cells);
    for (int i = 0; i < cells; ++i) positions[i] = i;
    std::shuffle(positions.begin(), positions.end(), rng);
    std::vector<int> stepped, batched;
    for (const int pos : positions) {
        if (pos >= 200 && pos < 280) continue;
        (stepped.size() < 250 ? stepped : batched).push_back(pos);
    }
    batched.resize(80);
    std::vector<std::vector<float>> host_k(cells), host_v(cells);

    auto make_cell = [&](int pos) {
        std::vector<float> kv(H * D), vv(H * D);
        const float scale = pos % 11 == 0 ? 1e-3f : (pos % 13 == 0 ? 40.0f : 1.0f);
        for (auto& x : kv) x = nd(rng) * scale;
        for (auto& x : vv) x = nd(rng) * scale;
        if (pos % 17 == 0) std::fill(kv.begin(), kv.end(), 0.0f);          // an all-zero group
        if (pos % 19 == 0) std::fill(vv.begin(), vv.end(), 0.0f);
        const bool peaked = pos % 23 == 0;   // raw outliers for the search; the rotation would spread them away
        if (peaked) {
            for (int blk : {2, 7, 12}) {   // both warp halves
                float g[16];
                peaked_group(g);
                std::copy(g, g + 16, kv.begin() + blk * k::QK4S_GROUP);
                peaked_group(g);
                std::copy(g, g + 16, vv.begin() + blk * k::QK4S_GROUP);
            }
        } else if (pos % 29 != 0) {        // usually rotated, as the real path's appends are
            for (int h = 0; h < H; ++h) {
                fwht256_host_reference(kv.data() + h * D);
                fwht256_host_reference(vv.data() + h * D);
            }
        }
        host_k[pos] = kv;
        host_v[pos] = vv;
    };

    for (const int pos : stepped) {
        make_cell(pos);
        const int32_t hstep[k::kStepCount] = {pos, pos + 1, 0, 0};
        ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");
        ck(cudaMemcpy(d_kcur, host_k[pos].data(), (size_t) H * D * 4, cudaMemcpyHostToDevice), "memcpy k");
        ck(cudaMemcpy(d_vcur, host_v[pos].data(), (size_t) H * D * 4, cudaMemcpyHostToDevice), "memcpy v");
        k::kv_append_q4s_step(d_k_q4s, d_v_q4s, d_table, d_step, d_kcur, d_vcur, s, nullptr);
    }
    {
        const int n = (int) batched.size();
        std::vector<float> K((size_t) n * H * D), V((size_t) n * H * D);
        std::vector<int32_t> hstep((size_t) n * k::kStepCount, 0);
        for (int t = 0; t < n; ++t) {
            make_cell(batched[t]);
            std::copy(host_k[batched[t]].begin(), host_k[batched[t]].end(), K.begin() + (size_t) t * H * D);
            std::copy(host_v[batched[t]].begin(), host_v[batched[t]].end(), V.begin() + (size_t) t * H * D);
            hstep[(size_t) t * k::kStepCount + k::kStepPos] = batched[t];
            hstep[(size_t) t * k::kStepCount + 1] = batched[t] + 1;
        }
        float* d_K = dalloc<float>(K.size());
        float* d_V = dalloc<float>(V.size());
        int32_t* d_steps = dalloc<int32_t>(hstep.size());
        ck(cudaMemcpy(d_K, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "memcpy K");
        ck(cudaMemcpy(d_V, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "memcpy V");
        ck(cudaMemcpy(d_steps, hstep.data(), hstep.size() * 4, cudaMemcpyHostToDevice), "memcpy steps");
        k::kv_append_q4s_steps(d_k_q4s, d_v_q4s, d_table, d_steps, k::kStepCount, n, d_K, d_V, s, nullptr);
        ck(cudaDeviceSynchronize(), "kv_append_q4s_steps");
        cudaFree(d_K);
        cudaFree(d_V);
        cudaFree(d_steps);
    }
    {
        const int pos0 = 200, T = 80;
        std::vector<float> K((size_t) T * H * D), V((size_t) T * H * D);
        for (int t = 0; t < T; ++t) {
            make_cell(pos0 + t);   // the batch kernel quantizes what it is given; make_cell picks its own rotation
            std::copy(host_k[pos0 + t].begin(), host_k[pos0 + t].end(), K.begin() + (size_t) t * H * D);
            std::copy(host_v[pos0 + t].begin(), host_v[pos0 + t].end(), V.begin() + (size_t) t * H * D);
        }
        float* d_K = dalloc<float>(K.size());
        float* d_V = dalloc<float>(V.size());
        ck(cudaMemcpy(d_K, K.data(), K.size() * 4, cudaMemcpyHostToDevice), "memcpy K");
        ck(cudaMemcpy(d_V, V.data(), V.size() * 4, cudaMemcpyHostToDevice), "memcpy V");
        k::kv_append_q4s(d_k_q4s, d_v_q4s, d_table, pos0, T, d_K, d_V, s, nullptr);
        ck(cudaDeviceSynchronize(), "kv_append_q4s");
        cudaFree(d_K);
        cudaFree(d_V);
    }

    std::vector<uint8_t> h_k_q4s(pool_bytes), h_v_q4s(pool_bytes);
    ck(cudaMemcpy(h_k_q4s.data(), d_k_q4s, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h k_q4s");
    ck(cudaMemcpy(h_v_q4s.data(), d_v_q4s, pool_bytes, cudaMemcpyDeviceToHost), "memcpy d2h v_q4s");

    long bad_blocks = 0;
    const int blocks_per_head = D / k::QK4S_GROUP;   // 16
    const uint64_t bytes_per_head = k::kv_q4s_bytes_per_head(D);
    std::vector<int> covered(stepped);
    covered.insert(covered.end(), batched.begin(), batched.end());
    for (int t = 0; t < 80; ++t) covered.push_back(200 + t);
    for (const int pos : covered) {
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const long long page = table[pos / P];
                const long long row = (page * H + h) * P + (pos % P);
                const uint8_t* pool = (is_v ? h_v_q4s : h_k_q4s).data() + (size_t) row * bytes_per_head;
                const float* raw = (is_v ? host_v : host_k)[pos].data() + h * D;
                for (int b = 0; b < blocks_per_head; ++b) {
                    uint8_t host_blk[9];
                    k::q4s_quantize_group_host(raw + b * k::QK4S_GROUP, host_blk);
                    if (std::memcmp(host_blk, pool + (size_t) b * 9, 9) != 0) {
                        if (bad_blocks == 0) {
                            std::printf("MISMATCH at pos=%d, is_v=%d, h=%d, b=%d\n", pos, is_v, h, b);
                            print_block("host", host_blk);
                            print_block("gpu ", pool + (size_t) b * 9);
                        }
                        ++bad_blocks;
                    }
                }
            }
        }
    }
    if (bad_blocks > 0) {
        std::fprintf(stderr, "FAIL: %ld Q4S blocks differ from the host quantizer over %zu cells\n", bad_blocks,
                     covered.size());
        ++g_fail;
    } else {
        std::printf("  -> %zu cells through step/steps/batch appends bitwise-equal, random/tiny/zero/peaked (OK)\n",
                    covered.size());
    }

    // -------------------------------------------------------------
    // Test 4: gather into FP16 scratch vs the host dequant, bitwise
    // -------------------------------------------------------------
    std::printf("[4/4] Verifying Q4S gather unpacking and dequantization...\n");
    const int max_ids = 128;
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_k_scratch = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_v_scratch = dalloc<uint16_t>((size_t) max_ids * H * D);

    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) ids[i] = covered[rng() % covered.size()];
    ck(cudaMemcpy(d_ids, ids.data(), max_ids * sizeof(int32_t), cudaMemcpyHostToDevice), "memcpy ids");
    const int32_t hstep[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hstep, sizeof(hstep), cudaMemcpyHostToDevice), "memcpy step");

    k::kv_gather_q4s_step(d_k_q4s, d_v_q4s, d_table, d_ids, d_step, max_ids, s, d_k_scratch, d_v_scratch, nullptr);
    ck(cudaDeviceSynchronize(), "kv_gather_q4s_step");

    std::vector<uint16_t> h_k_scratch((size_t) max_ids * H * D), h_v_scratch((size_t) max_ids * H * D);
    ck(cudaMemcpy(h_k_scratch.data(), d_k_scratch, h_k_scratch.size() * 2, cudaMemcpyDeviceToHost), "memcpy d2h k_scratch");
    ck(cudaMemcpy(h_v_scratch.data(), d_v_scratch, h_v_scratch.size() * 2, cudaMemcpyDeviceToHost), "memcpy d2h v_scratch");

    long bad_words = 0;
    double worst_quant_err = 0.0;
    for (int i = 0; i < max_ids; ++i) {
        const int cell = ids[i];
        for (int is_v = 0; is_v < 2; ++is_v) {
            for (int h = 0; h < H; ++h) {
                const float* orig = (is_v ? host_v : host_k)[cell].data() + h * D;
                const uint16_t* gathered = (is_v ? h_v_scratch : h_k_scratch).data() + ((size_t) i * H + h) * D;
                const long long page = table[cell / P];
                const long long row = (page * H + h) * P + (cell % P);
                const uint8_t* pool = (is_v ? h_v_q4s : h_k_q4s).data() + (size_t) row * bytes_per_head;
                for (int b = 0; b < blocks_per_head; ++b) {
                    const uint8_t* blk = pool + (size_t) b * 9;
                    const float d = k::e4m3_decode(blk[0]);
                    for (int j = 0; j < 8; ++j) {
                        for (int nib = 0; nib < 2; ++nib) {   // value 2j in the low nibble, 2j + 1 in the high one
                            const int q = (int) (((blk[1 + j] >> (4 * nib)) & 0x0F)) - 8;
                            const int el = b * k::QK4S_GROUP + 2 * j + nib;
                            const uint16_t want = k::f16_from_f32((float) q * d);   // the kernel's exact arithmetic
                            const uint16_t got = gathered[el];
                            if (got != want) {
                                if (bad_words == 0)
                                    std::fprintf(stderr, "MISMATCH at id=%d cell=%d is_v=%d h=%d el=%d: gpu %04x, "
                                                         "host %04x\n", i, cell, is_v, h, el, got, want);
                                ++bad_words;
                            }
                            worst_quant_err = std::max(worst_quant_err,
                                                       (double) std::fabs(k::f32_from_f16(got) - orig[el]));
                        }
                    }
                }
            }
        }
    }
    if (bad_words > 0) {
        std::fprintf(stderr, "FAIL: %ld of %zu gathered fp16 words differ from the host dequant\n", bad_words,
                     h_k_scratch.size() + h_v_scratch.size());
        ++g_fail;
    } else {
        std::printf("  -> gathered fp16 scratch bitwise-equal to f16_from_f32(q * d), max error %.4f (OK)\n",
                    worst_quant_err);
    }

    cudaFree(d_table);
    cudaFree(d_k_q4s);
    cudaFree(d_v_q4s);
    cudaFree(d_kcur);
    cudaFree(d_vcur);
    cudaFree(d_step);
    cudaFree(d_ids);
    cudaFree(d_k_scratch);
    cudaFree(d_v_scratch);

    if (g_fail == 0) {
        std::printf("=== kv_q4s_parity: ALL TESTS PASSED SUCCESSFULLY! ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== kv_q4s_parity: FAILED (%d failures) ===\n", g_fail);
    return 1;
}
