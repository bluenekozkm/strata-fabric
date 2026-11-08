// src/core/kv_block.cpp - Block-based CAS and serialization for tiered KV cache.
#include "strata/core/kv_block.hpp"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <cstring>
#include <new>

#ifdef _WIN32
#include <malloc.h>
#else
#include <stdlib.h>
#endif

namespace strata::core {

const char* kv_block_error_string(KvBlockError err) {
    switch (err) {
        case KvBlockError::kSuccess: return "success";
        case KvBlockError::kInvalidMagic: return "invalid KV block magic";
        case KvBlockError::kUnsupportedVersion: return "unsupported KV block version";
        case KvBlockError::kCorruptedChecksum: return "corrupted KV block checksum";
        case KvBlockError::kTruncatedPayload: return "truncated or overlapping KV payload";
        case KvBlockError::kMisalignedBuffer: return "buffer is not 4096-byte aligned";
        case KvBlockError::kInvalidFormat: return "invalid KV format";
        case KvBlockError::kBufferTooSmall: return "buffer too small to contain header or payload";
        default: return "unknown KV block error";
    }
}

// ---- Memory allocation helpers ----

uint8_t* kv_alloc_aligned(size_t size, size_t alignment) {
    if (size == 0) return nullptr;
    const size_t aligned_size = kv_align_up(size, alignment);
#ifdef _WIN32
    void* p = _aligned_malloc(aligned_size, alignment);
#else
    void* p = nullptr;
    if (posix_memalign(&p, alignment, aligned_size) != 0) {
        p = nullptr;
    }
#endif
    if (!p) throw std::bad_alloc();
    return static_cast<uint8_t*>(p);
}

void kv_free_aligned(void* ptr) noexcept {
    if (!ptr) return;
#ifdef _WIN32
    _aligned_free(ptr);
#else
    std::free(ptr);
#endif
}

// ---- AlignedBuffer ----

AlignedBuffer::AlignedBuffer(size_t size, size_t alignment)
    : size_(size), alignment_(alignment) {
    if (size > 0) {
        capacity_ = kv_align_up(size, alignment_);
        data_ = kv_alloc_aligned(capacity_, alignment_);
        std::memset(data_, 0, capacity_);
    }
}

AlignedBuffer::AlignedBuffer(AlignedBuffer&& other) noexcept
    : data_(std::exchange(other.data_, nullptr)),
      size_(std::exchange(other.size_, 0)),
      capacity_(std::exchange(other.capacity_, 0)),
      alignment_(other.alignment_) {}

AlignedBuffer& AlignedBuffer::operator=(AlignedBuffer&& other) noexcept {
    if (this == &other) return *this;
    reset();
    data_ = std::exchange(other.data_, nullptr);
    size_ = std::exchange(other.size_, 0);
    capacity_ = std::exchange(other.capacity_, 0);
    alignment_ = other.alignment_;
    return *this;
}

void AlignedBuffer::resize(size_t new_size, size_t alignment) {
    if (alignment != alignment_) {
        reset();
        alignment_ = alignment;
    }
    const size_t needed_cap = kv_align_up(new_size, alignment_);
    if (needed_cap > capacity_) {
        uint8_t* new_data = kv_alloc_aligned(needed_cap, alignment_);
        std::memset(new_data, 0, needed_cap);
        if (data_ && size_ > 0) {
            std::memcpy(new_data, data_, std::min(size_, new_size));
        }
        kv_free_aligned(data_);
        data_ = new_data;
        capacity_ = needed_cap;
    } else if (new_size > size_ && data_) {
        std::memset(data_ + size_, 0, new_size - size_);
    }
    size_ = new_size;
}

void AlignedBuffer::reset() noexcept {
    if (data_) {
        kv_free_aligned(data_);
        data_ = nullptr;
    }
    size_ = 0;
    capacity_ = 0;
}

// ---- Fast 64-bit Checksum (xxHash64) ----

namespace {

constexpr uint64_t XXH_PRIME64_1 = 0x9E3779B185EBCA87ULL;
constexpr uint64_t XXH_PRIME64_2 = 0xC2B2AE3D27D4EB4FULL;
constexpr uint64_t XXH_PRIME64_3 = 0x165667B19E3779F9ULL;
constexpr uint64_t XXH_PRIME64_4 = 0x85EBCA77C2B2AE63ULL;
constexpr uint64_t XXH_PRIME64_5 = 0x27D4EB2F165667C5ULL;

inline uint64_t xxh_rotl(uint64_t x, int r) {
    return (x << r) | (x >> (64 - r));
}

inline uint64_t xxh_round(uint64_t acc, uint64_t input) {
    acc += input * XXH_PRIME64_2;
    acc = xxh_rotl(acc, 31);
    acc *= XXH_PRIME64_1;
    return acc;
}

inline uint64_t xxh_merge_round(uint64_t acc, uint64_t val) {
    val = xxh_round(0, val);
    acc ^= val;
    acc = acc * XXH_PRIME64_1 + XXH_PRIME64_4;
    return acc;
}

inline uint64_t xxh_read64(const uint8_t* p) {
    uint64_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

inline uint32_t xxh_read32(const uint8_t* p) {
    uint32_t v;
    std::memcpy(&v, p, sizeof(v));
    return v;
}

} // namespace

uint64_t kv_block_checksum(const void* data, size_t size, uint64_t seed) {
    const uint8_t* p = static_cast<const uint8_t*>(data);
    const uint8_t* const b_end = p + size;
    uint64_t h;

    if (size >= 32) {
        const uint8_t* const limit = b_end - 32;
        uint64_t v1 = seed + XXH_PRIME64_1 + XXH_PRIME64_2;
        uint64_t v2 = seed + XXH_PRIME64_2;
        uint64_t v3 = seed + 0;
        uint64_t v4 = seed - XXH_PRIME64_1;

        do {
            v1 = xxh_round(v1, xxh_read64(p)); p += 8;
            v2 = xxh_round(v2, xxh_read64(p)); p += 8;
            v3 = xxh_round(v3, xxh_read64(p)); p += 8;
            v4 = xxh_round(v4, xxh_read64(p)); p += 8;
        } while (p <= limit);

        h = xxh_rotl(v1, 1) + xxh_rotl(v2, 7) + xxh_rotl(v3, 12) + xxh_rotl(v4, 18);
        h = xxh_merge_round(h, v1);
        h = xxh_merge_round(h, v2);
        h = xxh_merge_round(h, v3);
        h = xxh_merge_round(h, v4);
    } else {
        h = seed + XXH_PRIME64_5;
    }

    h += static_cast<uint64_t>(size);

    while (p + 8 <= b_end) {
        uint64_t k1 = xxh_round(0, xxh_read64(p));
        h ^= k1;
        h = xxh_rotl(h, 27) * XXH_PRIME64_1 + XXH_PRIME64_4;
        p += 8;
    }

    if (p + 4 <= b_end) {
        h ^= static_cast<uint64_t>(xxh_read32(p)) * XXH_PRIME64_1;
        h = xxh_rotl(h, 23) * XXH_PRIME64_2 + XXH_PRIME64_3;
        p += 4;
    }

    while (p < b_end) {
        h ^= static_cast<uint64_t>(*p) * XXH_PRIME64_5;
        h = xxh_rotl(h, 11) * XXH_PRIME64_1;
        p++;
    }

    h ^= h >> 33;
    h *= XXH_PRIME64_2;
    h ^= h >> 29;
    h *= XXH_PRIME64_3;
    h ^= h >> 32;

    return h;
}

uint64_t kv_compute_block_checksum(const uint8_t* buffer, size_t total_bytes) {
    constexpr size_t csum_offset = offsetof(KvBlockHeader, checksum);
    constexpr size_t csum_size = sizeof(uint64_t);

    if (total_bytes < sizeof(KvBlockHeader)) {
        return 0;
    }

    // Hash prefix before checksum field, then hash remainder using prefix hash as seed.
    uint64_t h_prefix = kv_block_checksum(buffer, csum_offset, 0);
    return kv_block_checksum(buffer + csum_offset + csum_size,
                             total_bytes - (csum_offset + csum_size),
                             h_prefix);
}

// ---- Content-Addressable Storage (CAS) Hash ----

uint64_t kv_block_hash(uint64_t parent_hash,
                      const int32_t* token_ids,
                      size_t num_tokens,
                      int format,
                      uint32_t num_layers,
                      uint32_t num_heads,
                      uint32_t head_dim) {
    if (token_ids == nullptr) {
        num_tokens = 0;
    }
    constexpr size_t kStackLimit = 8 + 4 + kKvBlockTokens * sizeof(int32_t) + 16;
    const size_t token_bytes = num_tokens * sizeof(int32_t);
    const size_t total_len = 8 + 4 + token_bytes + 16;

    alignas(8) std::array<uint8_t, kStackLimit> stack_buf;
    std::vector<uint8_t> heap_buf;
    uint8_t* dst = nullptr;

    if (total_len <= kStackLimit) {
        dst = stack_buf.data();
    } else {
        heap_buf.resize(total_len);
        dst = heap_buf.data();
    }

    uint8_t* p = dst;
    std::memcpy(p, &parent_hash, 8); p += 8;

    uint32_t nt = static_cast<uint32_t>(num_tokens);
    std::memcpy(p, &nt, 4); p += 4;

    if (token_bytes > 0 && token_ids != nullptr) {
        std::memcpy(p, token_ids, token_bytes); p += token_bytes;
    }

    int32_t fmt = format;
    std::memcpy(p, &fmt, 4); p += 4;
    std::memcpy(p, &num_layers, 4); p += 4;
    std::memcpy(p, &num_heads, 4); p += 4;
    std::memcpy(p, &head_dim, 4); p += 4;

    return kv_block_checksum(dst, total_len, 0x5354524154415F43ULL); // 'STRATA_C' seed
}

uint64_t kv_block_hash(const KvBlockHeader& header, const int32_t* token_ids, size_t num_tokens) {
    return kv_block_hash(header.parent_hash,
                         token_ids,
                         num_tokens,
                         header.format,
                         header.num_layers,
                         header.num_heads,
                         header.head_dim);
}

uint64_t kv_block_hash(const KvBlock& block) {
    return kv_block_hash(block.header,
                         block.payload.token_ids.data(),
                         block.payload.token_ids.size());
}

// ---- Serialize / Deserialize ----

KvBlockError kv_block_serialize(const KvBlock& block, AlignedBuffer& out_buffer) {
    const size_t payload_bytes = block.payload.total_payload_bytes();
    const size_t total_file_bytes = kv_align_up(sizeof(KvBlockHeader) + payload_bytes, kKvBlockAlignment);

    out_buffer.resize(total_file_bytes, kKvBlockAlignment);
    size_t written = 0;
    return kv_block_serialize(block, out_buffer.data(), total_file_bytes, &written);
}

KvBlockError kv_block_serialize(const KvBlock& block,
                               uint8_t* out_ptr,
                               size_t out_capacity,
                               size_t* out_written_bytes) {
    if (!out_ptr) return KvBlockError::kBufferTooSmall;

    if ((reinterpret_cast<uintptr_t>(out_ptr) & (kKvBlockAlignment - 1)) != 0) {
        return KvBlockError::kMisalignedBuffer;
    }

    if (block.header.format < kKvF16 || block.header.format > kKvQ4S) {
        return KvBlockError::kInvalidFormat;
    }

    if (block.payload.token_ids.size() > block.header.block_tokens) {
        return KvBlockError::kTruncatedPayload;
    }

    const size_t token_ids_bytes = block.payload.token_ids.size() * sizeof(int32_t);
    const size_t k_data_bytes = block.payload.k_data.size();
    const size_t v_data_bytes = block.payload.v_data.size();
    const size_t k_scale_bytes = block.payload.k_scale.size();
    const size_t v_scale_bytes = block.payload.v_scale.size();
    const size_t recurrence_bytes = block.payload.recurrence.size();

    const size_t payload_bytes = token_ids_bytes + k_data_bytes + v_data_bytes +
                                 k_scale_bytes + v_scale_bytes + recurrence_bytes;
    const size_t total_file_bytes = kv_align_up(sizeof(KvBlockHeader) + payload_bytes, kKvBlockAlignment);

    if (out_capacity < total_file_bytes) {
        return KvBlockError::kBufferTooSmall;
    }

    // Zero entire buffer so trailing padding is zeroed
    std::memset(out_ptr, 0, total_file_bytes);

    // Compute layout offsets
    uint32_t offset = sizeof(KvBlockHeader);

    const uint32_t token_ids_offset = offset;
    offset += static_cast<uint32_t>(token_ids_bytes);

    const uint32_t k_data_offset = offset;
    offset += static_cast<uint32_t>(k_data_bytes);

    const uint32_t v_data_offset = offset;
    offset += static_cast<uint32_t>(v_data_bytes);

    const uint32_t k_scale_offset = offset;
    offset += static_cast<uint32_t>(k_scale_bytes);

    const uint32_t v_scale_offset = offset;
    offset += static_cast<uint32_t>(v_scale_bytes);

    const uint32_t recurrence_offset = offset;
    offset += static_cast<uint32_t>(recurrence_bytes);

    // Prepare header
    KvBlockHeader hdr = block.header;
    std::memcpy(hdr.magic, kKvBlockMagic, sizeof(hdr.magic));
    hdr.version = kKvBlockVersion;
    hdr.token_count = static_cast<uint32_t>(block.payload.token_ids.size());
    hdr.payload_bytes = payload_bytes;
    hdr.total_file_bytes = total_file_bytes;

    hdr.token_ids_offset = token_ids_offset;
    hdr.token_ids_bytes = static_cast<uint32_t>(token_ids_bytes);
    hdr.k_data_offset = k_data_offset;
    hdr.k_data_bytes = static_cast<uint32_t>(k_data_bytes);
    hdr.v_data_offset = v_data_offset;
    hdr.v_data_bytes = static_cast<uint32_t>(v_data_bytes);
    hdr.k_scale_offset = k_scale_offset;
    hdr.k_scale_bytes = static_cast<uint32_t>(k_scale_bytes);
    hdr.v_scale_offset = v_scale_offset;
    hdr.v_scale_bytes = static_cast<uint32_t>(v_scale_bytes);
    hdr.recurrence_offset = recurrence_offset;
    hdr.recurrence_bytes = static_cast<uint32_t>(recurrence_bytes);

    // CAS hash: recalculate or preserve
    hdr.block_hash = kv_block_hash(hdr, block.payload.token_ids.data(), block.payload.token_ids.size());
    hdr.checksum = 0;

    // Write header
    std::memcpy(out_ptr, &hdr, sizeof(KvBlockHeader));

    // Write payload sections verbatim
    if (token_ids_bytes > 0) {
        std::memcpy(out_ptr + token_ids_offset, block.payload.token_ids.data(), token_ids_bytes);
    }
    if (k_data_bytes > 0) {
        std::memcpy(out_ptr + k_data_offset, block.payload.k_data.data(), k_data_bytes);
    }
    if (v_data_bytes > 0) {
        std::memcpy(out_ptr + v_data_offset, block.payload.v_data.data(), v_data_bytes);
    }
    if (k_scale_bytes > 0) {
        std::memcpy(out_ptr + k_scale_offset, block.payload.k_scale.data(), k_scale_bytes);
    }
    if (v_scale_bytes > 0) {
        std::memcpy(out_ptr + v_scale_offset, block.payload.v_scale.data(), v_scale_bytes);
    }
    if (recurrence_bytes > 0) {
        std::memcpy(out_ptr + recurrence_offset, block.payload.recurrence.data(), recurrence_bytes);
    }

    // Compute checksum over entire file (with header.checksum = 0)
    const uint64_t csum = kv_compute_block_checksum(out_ptr, total_file_bytes);
    std::memcpy(out_ptr + offsetof(KvBlockHeader, checksum), &csum, sizeof(csum));

    if (out_written_bytes) {
        *out_written_bytes = total_file_bytes;
    }
    return KvBlockError::kSuccess;
}

KvBlockError kv_block_deserialize(const uint8_t* buffer,
                                 size_t buffer_size,
                                 KvBlock& out_block,
                                 bool require_alignment) {
    if (!buffer) return KvBlockError::kBufferTooSmall;

    if (require_alignment && ((reinterpret_cast<uintptr_t>(buffer) & (kKvBlockAlignment - 1)) != 0)) {
        return KvBlockError::kMisalignedBuffer;
    }

    if (buffer_size < sizeof(KvBlockHeader)) {
        return KvBlockError::kBufferTooSmall;
    }

    KvBlockHeader hdr;
    std::memcpy(&hdr, buffer, sizeof(KvBlockHeader));

    if (std::memcmp(hdr.magic, kKvBlockMagic, sizeof(kKvBlockMagic)) != 0) {
        return KvBlockError::kInvalidMagic;
    }

    if (hdr.version != kKvBlockVersion) {
        return KvBlockError::kUnsupportedVersion;
    }

    if (hdr.format < kKvF16 || hdr.format > kKvQ4S) {
        return KvBlockError::kInvalidFormat;
    }

    if (hdr.total_file_bytes < sizeof(KvBlockHeader) || hdr.total_file_bytes > buffer_size) {
        return KvBlockError::kTruncatedPayload;
    }

    if (require_alignment && (hdr.total_file_bytes % kKvBlockAlignment) != 0) {
        return KvBlockError::kMisalignedBuffer;
    }

    // Verify checksum
    const uint64_t expected_csum = kv_compute_block_checksum(buffer, hdr.total_file_bytes);
    if (hdr.checksum != expected_csum) {
        return KvBlockError::kCorruptedChecksum;
    }

    // Verify sections do not overflow buffer or payload boundaries
    const uint64_t total_payload = hdr.payload_bytes;
    if (sizeof(KvBlockHeader) + total_payload > hdr.total_file_bytes) {
        return KvBlockError::kTruncatedPayload;
    }

    auto check_section = [&](uint32_t offset, uint32_t len) -> bool {
        if (len == 0) return true;
        if (offset < sizeof(KvBlockHeader)) return false;
        if (static_cast<uint64_t>(offset) + len > hdr.total_file_bytes) return false;
        return true;
    };

    if (!check_section(hdr.token_ids_offset, hdr.token_ids_bytes) ||
        !check_section(hdr.k_data_offset, hdr.k_data_bytes) ||
        !check_section(hdr.v_data_offset, hdr.v_data_bytes) ||
        !check_section(hdr.k_scale_offset, hdr.k_scale_bytes) ||
        !check_section(hdr.v_scale_offset, hdr.v_scale_bytes) ||
        !check_section(hdr.recurrence_offset, hdr.recurrence_bytes)) {
        return KvBlockError::kTruncatedPayload;
    }

    if (hdr.token_ids_bytes != hdr.token_count * sizeof(int32_t)) {
        return KvBlockError::kTruncatedPayload;
    }

    if (hdr.token_count > hdr.block_tokens) {
        return KvBlockError::kTruncatedPayload;
    }

    const uint64_t sum_section_bytes = static_cast<uint64_t>(hdr.token_ids_bytes) +
                                       hdr.k_data_bytes + hdr.v_data_bytes +
                                       hdr.k_scale_bytes + hdr.v_scale_bytes +
                                       hdr.recurrence_bytes;
    if (sum_section_bytes != hdr.payload_bytes) {
        return KvBlockError::kTruncatedPayload;
    }

    // Unpack data
    out_block.header = hdr;

    out_block.payload.token_ids.resize(hdr.token_count);
    if (hdr.token_ids_bytes > 0) {
        std::memcpy(out_block.payload.token_ids.data(), buffer + hdr.token_ids_offset, hdr.token_ids_bytes);
    }

    out_block.payload.k_data.resize(hdr.k_data_bytes);
    if (hdr.k_data_bytes > 0) {
        std::memcpy(out_block.payload.k_data.data(), buffer + hdr.k_data_offset, hdr.k_data_bytes);
    }

    out_block.payload.v_data.resize(hdr.v_data_bytes);
    if (hdr.v_data_bytes > 0) {
        std::memcpy(out_block.payload.v_data.data(), buffer + hdr.v_data_offset, hdr.v_data_bytes);
    }

    out_block.payload.k_scale.resize(hdr.k_scale_bytes);
    if (hdr.k_scale_bytes > 0) {
        std::memcpy(out_block.payload.k_scale.data(), buffer + hdr.k_scale_offset, hdr.k_scale_bytes);
    }

    out_block.payload.v_scale.resize(hdr.v_scale_bytes);
    if (hdr.v_scale_bytes > 0) {
        std::memcpy(out_block.payload.v_scale.data(), buffer + hdr.v_scale_offset, hdr.v_scale_bytes);
    }

    out_block.payload.recurrence.resize(hdr.recurrence_bytes);
    if (hdr.recurrence_bytes > 0) {
        std::memcpy(out_block.payload.recurrence.data(), buffer + hdr.recurrence_offset, hdr.recurrence_bytes);
    }

    return KvBlockError::kSuccess;
}

} // namespace strata::core
