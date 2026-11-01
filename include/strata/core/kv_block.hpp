// include/strata/core/kv_block.hpp - Block-based CAS and serialization for tiered KV cache.
// Each block holds a fixed chunk of tokens (kKvBlockTokens = 128) and their layer KV states.
// Memory alignment: 4096 bytes for Direct I/O (O_DIRECT).
#pragma once

#include "strata/kernels/kv_stream.hpp"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

namespace strata::core {

using strata::kernels::KvFormat;
using strata::kernels::kKvF16;
using strata::kernels::kKvInt8;
using strata::kernels::kKvQ4;
using strata::kernels::kKvHybrid;
using strata::kernels::kKvQ4S;

inline constexpr size_t kKvBlockAlignment = 4096;
inline constexpr uint32_t kKvBlockTokens = 128;
inline constexpr char kKvBlockMagic[8] = {'S', 'T', 'R', 'K', 'V', 'B', 'L', 'K'};
inline constexpr uint32_t kKvBlockVersion = 1;

/// Error codes for KV block serialization and deserialization.
enum class KvBlockError {
    kSuccess = 0,
    kInvalidMagic,
    kUnsupportedVersion,
    kCorruptedChecksum,
    kTruncatedPayload,
    kMisalignedBuffer,
    kInvalidFormat,
    kBufferTooSmall,
};

const char* kv_block_error_string(KvBlockError err);

/// Header stored at the beginning of each serialized KV block file/chunk (128 bytes fixed size).
#pragma pack(push, 1)
struct KvBlockHeader {
    char magic[8] = {'S', 'T', 'R', 'K', 'V', 'B', 'L', 'K'};
    uint32_t version = kKvBlockVersion;
    int32_t format = kKvInt8;
    uint32_t block_tokens = kKvBlockTokens;
    uint32_t num_layers = 1;
    uint32_t num_heads = 2;
    uint32_t head_dim = 256;
    uint64_t parent_hash = 0;       // CAS hash of parent block in the prefix chain (0 for root)
    uint64_t block_hash = 0;        // Content-Addressable Storage (CAS) hash for this block
    uint64_t checksum = 0;          // Data integrity checksum
    uint64_t payload_bytes = 0;     // Raw payload size
    uint64_t total_file_bytes = 0;  // Padded to multiple of 4096 bytes
    uint32_t flags = 0;
    uint32_t token_count = 0;       // Actual token count in payload (<= block_tokens)

    // Byte offsets (from file start) and byte counts of payload sections:
    uint32_t token_ids_offset = 0;
    uint32_t token_ids_bytes = 0;
    uint32_t k_data_offset = 0;
    uint32_t k_data_bytes = 0;
    uint32_t v_data_offset = 0;
    uint32_t v_data_bytes = 0;
    uint32_t k_scale_offset = 0;
    uint32_t k_scale_bytes = 0;
    uint32_t v_scale_offset = 0;
    uint32_t v_scale_bytes = 0;
    uint32_t recurrence_offset = 0;
    uint32_t recurrence_bytes = 0;

    bool operator==(const KvBlockHeader& o) const = default;
};
#pragma pack(pop)
static_assert(sizeof(KvBlockHeader) == 128, "KvBlockHeader must be exactly 128 bytes");

/// Unserialized KV payload containing token IDs, KV tensor codes, scales, and recurrence states.
struct KvBlockPayload {
    std::vector<int32_t> token_ids;
    std::vector<uint8_t> k_data;
    std::vector<uint8_t> v_data;
    std::vector<uint8_t> k_scale;
    std::vector<uint8_t> v_scale;
    std::vector<uint8_t> recurrence;

    size_t total_payload_bytes() const {
        return token_ids.size() * sizeof(int32_t) +
               k_data.size() + v_data.size() +
               k_scale.size() + v_scale.size() +
               recurrence.size();
    }

    bool operator==(const KvBlockPayload& o) const = default;
};

/// In-memory representation of a complete KV block (Header + Payload).
struct KvBlock {
    KvBlockHeader header;
    KvBlockPayload payload;

    bool operator==(const KvBlock& o) const = default;
};

/// Round size up to nearest multiple of alignment.
inline constexpr size_t kv_align_up(size_t size, size_t alignment = kKvBlockAlignment) {
    return (size + alignment - 1) & ~(alignment - 1);
}

/// Allocate memory aligned to 4096 bytes (or specified alignment).
uint8_t* kv_alloc_aligned(size_t size, size_t alignment = kKvBlockAlignment);

/// Free memory allocated with kv_alloc_aligned.
void kv_free_aligned(void* ptr) noexcept;

/// RAII container for 4096-byte aligned memory suitable for Direct I/O (O_DIRECT).
class AlignedBuffer {
public:
    AlignedBuffer() = default;
    explicit AlignedBuffer(size_t size, size_t alignment = kKvBlockAlignment);
    ~AlignedBuffer() { reset(); }

    AlignedBuffer(const AlignedBuffer&) = delete;
    AlignedBuffer& operator=(const AlignedBuffer&) = delete;

    AlignedBuffer(AlignedBuffer&& other) noexcept;
    AlignedBuffer& operator=(AlignedBuffer&& other) noexcept;

    void resize(size_t new_size, size_t alignment = kKvBlockAlignment);
    void reset() noexcept;

    uint8_t* data() noexcept { return data_; }
    const uint8_t* data() const noexcept { return data_; }

    size_t size() const noexcept { return size_; }
    size_t capacity() const noexcept { return capacity_; }
    bool empty() const noexcept { return size_ == 0; }

    bool is_aligned(size_t alignment = kKvBlockAlignment) const noexcept {
        return (reinterpret_cast<uintptr_t>(data_) & (alignment - 1)) == 0;
    }

private:
    uint8_t* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
    size_t alignment_ = kKvBlockAlignment;
};

/// Fast 64-bit checksum (xxHash64 algorithm) for data integrity.
uint64_t kv_block_checksum(const void* data, size_t size, uint64_t seed = 0);

/// Computes the integrity checksum for a complete serialized block buffer.
uint64_t kv_compute_block_checksum(const uint8_t* buffer, size_t total_bytes);

/// Content-Addressable Storage (CAS) hash:
/// Hashes parent_hash + token_ids + metadata. Uniquely identifies this prefix chunk.
uint64_t kv_block_hash(uint64_t parent_hash,
                      const int32_t* token_ids,
                      size_t num_tokens,
                      int format,
                      uint32_t num_layers = 1,
                      uint32_t num_heads = 2,
                      uint32_t head_dim = 256);

uint64_t kv_block_hash(const KvBlockHeader& header, const int32_t* token_ids, size_t num_tokens);

uint64_t kv_block_hash(const KvBlock& block);

/// Serialize a KvBlock into a 4096-aligned buffer ready for Direct I/O.
KvBlockError kv_block_serialize(const KvBlock& block, AlignedBuffer& out_buffer);

/// Serialize into a pre-allocated aligned memory buffer.
KvBlockError kv_block_serialize(const KvBlock& block,
                               uint8_t* out_ptr,
                               size_t out_capacity,
                               size_t* out_written_bytes);

/// Deserialize a KvBlock from memory with integrity and alignment checks.
KvBlockError kv_block_deserialize(const uint8_t* buffer,
                                 size_t buffer_size,
                                 KvBlock& out_block,
                                 bool require_alignment = true);

inline KvBlockError kv_block_deserialize(const AlignedBuffer& buffer,
                                        KvBlock& out_block,
                                        bool require_alignment = true) {
    return kv_block_deserialize(buffer.data(), buffer.size(), out_block, require_alignment);
}

} // namespace strata::core
