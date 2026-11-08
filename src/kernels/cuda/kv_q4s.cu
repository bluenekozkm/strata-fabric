// src/kernels/cuda/kv_q4s.cu - see include/strata/kernels/kv_q4s.hpp. q4s: q4_0's rotation and footprint,
// sixteen values per block and a searched e4m3 scale instead of one max-derived fp16 scale.
#include "strata/kernels/kv_q4s.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4s: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// One 16-value group in one half-warp (lane = element; lanes 0-15 cover block 2b, 16-31 block 2b + 1).
// THE ERROR TREE IS PART OF THE HOST/DEVICE CONTRACT (q4s_quantize_group_host in kv_q4s.hpp): the residual
// x - q*d is one fused multiply-add, squared in float, and the 16 squares are summed in double in the
// shuffle-butterfly order - xor offsets 8, 4, 2, 1, i.e. lane i adds lane i^8 first, then i^4, then i^2,
// then i^1 - so every lane of the half ends with the same total and the host reference runs the same
// additions in the same order. Both sides therefore pick the same candidate on near-ties and the parity
// test can memcmp the blocks. The amax butterfly keeps q4_0's tie rule (ties in |x| to the larger value)
// so all lanes agree on m, though only |m| feeds the scale.
__device__ __forceinline__ void q4s_group(float x, int t, uint8_t& dbits, uint8_t& byte) {
    const int l = t & 15;
    float amax = fabsf(x), mval = x;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1) {
        const float a = __shfl_xor_sync(0xffffffffu, amax, o, 16);
        const float v = __shfl_xor_sync(0xffffffffu, mval, o, 16);
        if (a > amax || (a == amax && v > mval)) { amax = a; mval = v; }
    }
    // amax is identical across each half-warp, but the branch below must be uniform for the WHOLE warp: HIP's
    // __shfl*_sync runs a mask waterfall per call and never settles when one wave carries two masks (and CUDA
    // demands that every thread named in the mask executes the call). A zero half enters with the rest - its
    // candidate scale is e4m3_encode(0) = 0 - and after the loop, outside every shuffle, it is forced back to the
    // zero block. The error tree keeps the half-warp butterfly: xor o in {8, 4, 2, 1} never crosses the half, so
    // width 16 with the full mask exchanges exactly the half's lanes.
    const float other = __shfl_xor_sync(0xffffffffu, amax, 16);
    uint8_t bits = 0;
    int q = 0;
    if (amax > 0.0f || other > 0.0f) {
        double best = 0.0;
        float dwin = 0.0f;
        bool got = false;
#pragma unroll
        for (int ci = 0; ci < 3; ++ci) {
            const float c = ci == 0 ? 1.0f : (ci == 1 ? 0.9f : 0.8f);
            const uint8_t cb = e4m3_encode(amax * c / 7.0f);
            const float d = e4m3_decode(cb);
            const int qq = q4s_rn(x / d);
            q = qq < -8 ? -8 : (qq > 7 ? 7 : qq);
            const float r = fmaf(-(float) q, d, x);
            double s = (double) (r * r);
            // the contract tree, device side (see q4s_quantize_group_host for the host half)
#pragma unroll
            for (int o = 8; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o, 16);
            if (!got || s < best) { got = true; best = s; bits = cb; dwin = d; }
        }
        // the codes are the WINNING candidate's, not the last one's
        const int qw = q4s_rn(x / dwin);
        q = qw < -8 ? -8 : (qw > 7 ? 7 : qw);
    }
    if (!(amax > 0.0f)) { bits = 0; q = 0; }   // the zero half keeps the zero block (its candidate d was 0)
    dbits = bits;
    const uint8_t packed = (uint8_t) (q + 8);
    // byte j holds element 2j in the low nibble, 2j + 1 in the high one; lane j of the half packs it
    const uint8_t even = __shfl_sync(0xffffffffu, packed, 2 * (l & 7), 16);
    const uint8_t odd = __shfl_sync(0xffffffffu, packed, 2 * (l & 7) + 1, 16);
    byte = l < 8 ? (uint8_t) (even | (odd << 4)) : 0;
}

__device__ __forceinline__ void q4s_store(uint8_t* pool, long long row, int b, int t, uint8_t dbits, uint8_t byte) {
    block_q4s* blk = reinterpret_cast<block_q4s*>(pool + row * (long long) sizeof(block_q4s) * 16) + b;
    if ((t & 15) == 0) blk->d = dbits;
    if ((t & 15) < 8) blk->qs[t & 15] = byte;
}

// One warp = two 16-value groups of one KV head of K (plane 0) or V (plane 1) for token step_idx; 32 threads.
__global__ void kv_append_q4s_kernel(uint8_t* __restrict__ k_q4s, uint8_t* __restrict__ v_q4s,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ step,
                                     int step_stride, int planes,
                                     const float* __restrict__ kcur, const float* __restrict__ vcur,
                                     int kv_heads, int head_dim, int page_size, KvHostPools host) {
    const int step_idx = (int) (blockIdx.z / (unsigned) planes);
    const bool is_v = (blockIdx.z % (unsigned) planes) == 1u;
    const long long pos = (long long) __ldg(step + (long long) step_idx * step_stride + kStepPos);
    const int h = blockIdx.x, bw = blockIdx.y, t = threadIdx.x;
    const int b = bw * 2 + (t >= 16 ? 1 : 0);
    const long long step_off = (long long) step_idx * kv_heads * head_dim;
    const float x = (is_v ? vcur : kcur)[step_off + h * head_dim + b * QK4S_GROUP + (t & 15)];
    uint8_t byte, dbits;
    q4s_group(x, t, dbits, byte);
    const long long page = (long long) table[pos / page_size];
    if (page >= 0)
        q4s_store(is_v ? v_q4s : k_q4s, (page * kv_heads + h) * page_size + (pos % page_size), b, t, dbits, byte);
    if (host.k_q4s != nullptr)
        q4s_store(is_v ? host.v_q4s : host.k_q4s,
                  ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t, dbits, byte);
}

// The prompt path: grid (T, kv_heads, warps), K then V; also into the staging pool (identity layout) when given.
__global__ void kv_append_q4s_batch_kernel(uint8_t* __restrict__ k_q4s, uint8_t* __restrict__ v_q4s,
                                           const int32_t* __restrict__ table, int64_t pos0,
                                           const float* __restrict__ K, const float* __restrict__ V,
                                           int kv_heads, int head_dim, int page_size, int is_v_grid,
                                           KvHostPools host, KvHostPools stage) {
    const long long tok = blockIdx.x;
    const long long pos = pos0 + tok;
    const int h = blockIdx.y, bw = blockIdx.z, t = threadIdx.x;
    const bool is_v = is_v_grid != 0;
    const int b = bw * 2 + (t >= 16 ? 1 : 0);
    const float x = (is_v ? V : K)[tok * (kv_heads * head_dim) + h * head_dim + b * QK4S_GROUP + (t & 15)];
    uint8_t byte, dbits;
    q4s_group(x, t, dbits, byte);
    const long long page = (long long) table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0)
        q4s_store(is_v ? v_q4s : k_q4s, (page * kv_heads + h) * page_size + (pos % page_size), b, t, dbits, byte);
    if (host.k_q4s != nullptr) q4s_store(is_v ? host.v_q4s : host.k_q4s, row_id, b, t, dbits, byte);
    if (stage.k_q4s != nullptr) q4s_store(is_v ? stage.v_q4s : stage.k_q4s, row_id, b, t, dbits, byte);
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths); 16 of the 32 threads per row
// unpack one block's 8 bytes, low nibble then high.
__global__ void kv_gather_q4s_kernel(const uint8_t* __restrict__ k_q4s, const uint8_t* __restrict__ v_q4s,
                                     const int32_t* __restrict__ table, const int32_t* __restrict__ ids,
                                     const int32_t* __restrict__ step, int kv_heads, int head_dim, int page_size,
                                     uint16_t* __restrict__ k_scratch, uint16_t* __restrict__ v_scratch) {
    const long long n_ids = (long long) __ldg(step + kStepWidth);
    const int blocks_per_head = head_dim / QK4S_GROUP;                  // 16
    const int bytes_per_head = blocks_per_head * (int) sizeof(block_q4s);   // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long) blockIdx.x * blockDim.y + threadIdx.y;
    if (blk_idx >= total_blocks) return;
    const int t = threadIdx.x;
    if (t >= QK4S_GROUP) return;

    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int) (blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long) table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const block_q4s* k_blk = reinterpret_cast<const block_q4s*>(k_q4s + row * bytes_per_head) + b;
    const block_q4s* v_blk = reinterpret_cast<const block_q4s*>(v_q4s + row * bytes_per_head) + b;
    const float kd = e4m3_decode(k_blk->d);
    const float vd = e4m3_decode(v_blk->d);
    const uint8_t k_byte = k_blk->qs[t >> 1];
    const uint8_t v_byte = v_blk->qs[t >> 1];
    const int kq = (t & 1) != 0 ? ((k_byte >> 4) - 8) : ((k_byte & 0x0F) - 8);
    const int vq = (t & 1) != 0 ? ((v_byte >> 4) - 8) : ((v_byte & 0x0F) - 8);
    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4S_GROUP + t);
    k_scratch[dst_offset] = f16_from_f32((float) kq * kd);
    v_scratch[dst_offset] = f16_from_f32((float) vq * vd);
}

void need_256(const QsaShapes& s, const char* what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

}  // namespace

void kv_append_q4s_steps(uint8_t* k_q4s, uint8_t* v_q4s, const int32_t* page_table, const int32_t* step,
                         int step_stride, int n_steps, const float* kcur, const float* vcur, const QsaShapes& s,
                         void* stream, const KvHostPools* host) {
    if (n_steps <= 0) return;
    need_256(s, "kv_append_q4s");
    const int planes = (v_q4s == nullptr || (v_q4s == k_q4s && vcur == kcur)) ? 1 : 2;
    const dim3 grid((unsigned) s.n_head_kv, (unsigned) (s.head_dim / (QK4S_GROUP * 2)),
                    (unsigned) (n_steps * planes));
    kv_append_q4s_kernel<<<grid, 32, 0, (cudaStream_t) stream>>>(
        k_q4s, v_q4s, page_table, step, step_stride, planes, kcur, vcur, (int) s.n_head_kv, (int) s.head_dim,
        (int) s.page_size, host ? *host : KvHostPools{});
    check("kv_append_q4s launch");
}

void kv_append_q4s_step(uint8_t* k_q4s, uint8_t* v_q4s, const int32_t* page_table, const int32_t* step,
                        const float* kcur, const float* vcur, const QsaShapes& s, void* stream,
                        const KvHostPools* host) {
    kv_append_q4s_steps(k_q4s, v_q4s, page_table, step, 0, 1, kcur, vcur, s, stream, host);
}

void kv_append_q4s(uint8_t* k_q4s, uint8_t* v_q4s, const int32_t* page_table, int64_t pos0, int64_t T,
                   const float* K, const float* V, const QsaShapes& s, void* stream, const KvHostPools* host,
                   const KvHostPools* stage) {
    if (T <= 0) return;
    need_256(s, "kv_append_q4s");
    const dim3 grid((unsigned) T, (unsigned) s.n_head_kv, (unsigned) (s.head_dim / (QK4S_GROUP * 2)));
    cudaStream_t cs = (cudaStream_t) stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_q4s_batch_kernel<<<grid, 32, 0, cs>>>(k_q4s, v_q4s, page_table, pos0, K, V, (int) s.n_head_kv,
                                                        (int) s.head_dim, (int) s.page_size, is_v, h, st);
    check("kv_append_q4s batch launch");
}

void kv_gather_q4s_step(const uint8_t* k_q4s, const uint8_t* v_q4s, const int32_t* page_table, const int32_t* ids,
                        const int32_t* step, int64_t max_ids, const QsaShapes& s, uint16_t* k_scratch,
                        uint16_t* v_scratch, void* stream) {
    if (max_ids <= 0) return;
    const int blocks_per_head = (int) (s.head_dim / QK4S_GROUP);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned) ((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_q4s_kernel<<<dim3(num_blocks), dim3(32, rows_per_block), 0, (cudaStream_t) stream>>>(
        k_q4s, v_q4s, page_table, ids, step, (int) s.n_head_kv, (int) s.head_dim, (int) s.page_size, k_scratch,
        v_scratch);
    check("kv_gather_q4s launch");
}

}  // namespace strata::kernels
