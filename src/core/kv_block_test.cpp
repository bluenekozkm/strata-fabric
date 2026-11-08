// src/core/kv_block_test.cpp - Standalone CPU unit tests for KvBlock CAS & serialization.
#include "strata/core/kv_block.hpp"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

static int fails = 0;
#define CHECK(c) do { \
    if (!(c)) { \
        std::printf("FAIL line %d: %s\n", __LINE__, #c); \
        ++fails; \
    } \
} while (0)

namespace {

using namespace strata::core;

// Helper to fill vector with deterministic pseudo-random bytes
std::vector<uint8_t> make_random_bytes(size_t n, uint32_t seed) {
    std::vector<uint8_t> v(n);
    std::mt19937 rng(seed);
    for (size_t i = 0; i < n; ++i) {
        v[i] = static_cast<uint8_t>(rng() & 0xFF);
    }
    return v;
}

// Helper to create token ID sequence
std::vector<int32_t> make_token_ids(size_t n, int32_t start_id = 100) {
    std::vector<int32_t> tokens(n);
    std::iota(tokens.begin(), tokens.end(), start_id);
    return tokens;
}

// -----------------------------------------------------------------------------
// Test 1: Hash determinism and parent-chaining CAS property
// -----------------------------------------------------------------------------
void test_cas_hash_determinism_and_chaining() {
    std::printf("[Test 1] CAS hash determinism and parent-chaining...\n");

    const auto tokens_a = make_token_ids(kKvBlockTokens, 1000);
    const auto tokens_b = make_token_ids(kKvBlockTokens, 2000);

    // 1. Determinism: same inputs must produce exact same hash
    uint64_t h1 = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvInt8, 1, 2, 256);
    uint64_t h2 = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvInt8, 1, 2, 256);
    CHECK(h1 != 0);
    CHECK(h1 == h2);

    // 2. Token sensitivity: different tokens must produce different hash
    uint64_t h_diff_tokens = kv_block_hash(0, tokens_b.data(), tokens_b.size(), kKvInt8, 1, 2, 256);
    CHECK(h_diff_tokens != h1);

    // Single token change
    auto tokens_mutated = tokens_a;
    tokens_mutated[tokens_mutated.size() / 2] ^= 0x55;
    uint64_t h_mutated = kv_block_hash(0, tokens_mutated.data(), tokens_mutated.size(), kKvInt8, 1, 2, 256);
    CHECK(h_mutated != h1);

    // 3. Parent-chaining: same tokens with different parent_hash must produce different hash
    uint64_t h_child1 = kv_block_hash(h1, tokens_b.data(), tokens_b.size(), kKvInt8, 1, 2, 256);
    uint64_t h_child2 = kv_block_hash(h_diff_tokens, tokens_b.data(), tokens_b.size(), kKvInt8, 1, 2, 256);
    CHECK(h_child1 != h_child2);
    CHECK(h_child1 != h1);

    // 4. Merkle prefix chaining: A -> B -> C
    // If A changes, B's parent changes, so B changes, and C's parent changes, so C changes.
    const auto tokens_c = make_token_ids(kKvBlockTokens, 3000);
    uint64_t chain1_a = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvInt8);
    uint64_t chain1_b = kv_block_hash(chain1_a, tokens_b.data(), tokens_b.size(), kKvInt8);
    uint64_t chain1_c = kv_block_hash(chain1_b, tokens_c.data(), tokens_c.size(), kKvInt8);

    uint64_t chain2_a = kv_block_hash(0, tokens_mutated.data(), tokens_mutated.size(), kKvInt8);
    uint64_t chain2_b = kv_block_hash(chain2_a, tokens_b.data(), tokens_b.size(), kKvInt8);
    uint64_t chain2_c = kv_block_hash(chain2_b, tokens_c.data(), tokens_c.size(), kKvInt8);

    CHECK(chain1_a != chain2_a);
    CHECK(chain1_b != chain2_b);
    CHECK(chain1_c != chain2_c);

    // 5. Metadata sensitivity: changing format or geometry changes hash
    uint64_t h_hybrid = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvHybrid, 1, 2, 256);
    uint64_t h_q4 = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvQ4, 1, 2, 256);
    uint64_t h_q4s = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvQ4S, 1, 2, 256);
    uint64_t h_layers2 = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvInt8, 2, 2, 256);
    uint64_t h_heads4 = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvInt8, 1, 4, 256);
    uint64_t h_dim128 = kv_block_hash(0, tokens_a.data(), tokens_a.size(), kKvInt8, 1, 2, 128);

    CHECK(h_hybrid != h1);
    CHECK(h_q4 != h1);
    CHECK(h_q4s != h1);
    CHECK(h_layers2 != h1);
    CHECK(h_heads4 != h1);
    CHECK(h_dim128 != h1);
}

// -----------------------------------------------------------------------------
// Test 2: Serialize and deserialize round-trip bit preservation across formats
// -----------------------------------------------------------------------------
void test_round_trip_format(int format, const char* format_name) {
    std::printf("[Test 2] Round-trip bit preservation for format: %s...\n", format_name);

    KvBlock block;
    block.header.format = format;
    block.header.block_tokens = kKvBlockTokens;
    block.header.num_layers = 2;
    block.header.num_heads = 2;
    block.header.head_dim = 256;
    block.header.parent_hash = 0x123456789ABCDEF0ULL;

    block.payload.token_ids = make_token_ids(kKvBlockTokens, 500);

    // Configure payload sections according to format characteristics
    constexpr size_t num_cells = kKvBlockTokens;
    constexpr size_t heads = 2;
    constexpr size_t head_dim = 256;
    constexpr size_t layers = 2;

    if (format == kKvInt8) {
        // Int8: K/V codes (1 byte per val) + K/V scales (fp16 per 64 vals = 2 bytes / 64 = 8 bytes per cell/head)
        const size_t code_bytes = layers * num_cells * heads * head_dim;
        const size_t scale_bytes = layers * num_cells * heads * (head_dim / 64) * sizeof(uint16_t);
        block.payload.k_data = make_random_bytes(code_bytes, 101);
        block.payload.v_data = make_random_bytes(code_bytes, 102);
        block.payload.k_scale = make_random_bytes(scale_bytes, 103);
        block.payload.v_scale = make_random_bytes(scale_bytes, 104);
    } else if (format == kKvHybrid) {
        // Hybrid (K8V4): K as int8 + scales, V as q4_0 (144 bytes per cell/head for head_dim=256)
        const size_t k_bytes = layers * num_cells * heads * head_dim;
        const size_t k_scale_bytes = layers * num_cells * heads * (head_dim / 64) * sizeof(uint16_t);
        const size_t v_q4_bytes = layers * num_cells * heads * (head_dim / 32 * 18);
        block.payload.k_data = make_random_bytes(k_bytes, 201);
        block.payload.k_scale = make_random_bytes(k_scale_bytes, 202);
        block.payload.v_data = make_random_bytes(v_q4_bytes, 203);
    } else if (format == kKvQ4) {
        // Q4_0: Both K and V are q4_0
        const size_t q4_bytes = layers * num_cells * heads * (head_dim / 32 * 18);
        block.payload.k_data = make_random_bytes(q4_bytes, 301);
        block.payload.v_data = make_random_bytes(q4_bytes, 302);
    } else if (format == kKvQ4S) {
        // Q4S: Both K and V are q4s (kv_q4s.hpp: one e4m3 scale + 8 code bytes per 16 values, q4_0's 144 B/cell-head)
        const size_t q4s_bytes = layers * num_cells * heads * (head_dim / 16 * 9);
        block.payload.k_data = make_random_bytes(q4s_bytes, 311);
        block.payload.v_data = make_random_bytes(q4s_bytes, 312);
    } else if (format == kKvF16) {
        // FP16: Both K and V are fp16 (2 bytes per val)
        const size_t f16_bytes = layers * num_cells * heads * head_dim * sizeof(uint16_t);
        block.payload.k_data = make_random_bytes(f16_bytes, 401);
        block.payload.v_data = make_random_bytes(f16_bytes, 402);
    }

    // Optional recurrence state
    block.payload.recurrence = make_random_bytes(512, 999);

    // Compute expected CAS block hash
    block.header.block_hash = kv_block_hash(block);

    // Serialize
    AlignedBuffer buf;
    KvBlockError err = kv_block_serialize(block, buf);
    CHECK(err == KvBlockError::kSuccess);
    CHECK(!buf.empty());
    CHECK(buf.size() >= sizeof(KvBlockHeader) + block.payload.total_payload_bytes());
    CHECK(buf.size() % kKvBlockAlignment == 0);

    // Deserialize
    KvBlock restored;
    err = kv_block_deserialize(buf, restored, true);
    CHECK(err == KvBlockError::kSuccess);

    // Check header fields
    CHECK(restored.header.magic[0] == 'S');
    CHECK(restored.header.version == kKvBlockVersion);
    CHECK(restored.header.format == format);
    CHECK(restored.header.block_tokens == kKvBlockTokens);
    CHECK(restored.header.num_layers == layers);
    CHECK(restored.header.num_heads == heads);
    CHECK(restored.header.head_dim == head_dim);
    CHECK(restored.header.parent_hash == block.header.parent_hash);
    CHECK(restored.header.block_hash == block.header.block_hash);
    CHECK(restored.header.token_count == kKvBlockTokens);
    CHECK(restored.header.checksum != 0);

    // Exact bit-preservation checks on all payload vectors
    CHECK(restored.payload.token_ids == block.payload.token_ids);
    CHECK(restored.payload.k_data == block.payload.k_data);
    CHECK(restored.payload.v_data == block.payload.v_data);
    CHECK(restored.payload.k_scale == block.payload.k_scale);
    CHECK(restored.payload.v_scale == block.payload.v_scale);
    CHECK(restored.payload.recurrence == block.payload.recurrence);
    CHECK(restored.payload == block.payload);

    // Verify round-trip re-serialization produces exact identical bytes
    AlignedBuffer buf2;
    err = kv_block_serialize(restored, buf2);
    CHECK(err == KvBlockError::kSuccess);
    CHECK(buf2.size() == buf.size());
    CHECK(std::memcmp(buf2.data(), buf.data(), buf.size()) == 0);
}

void test_serialize_deserialize_round_trip() {
    test_round_trip_format(kKvInt8, "INT8");
    test_round_trip_format(kKvHybrid, "K8V4 (Hybrid)");
    test_round_trip_format(kKvQ4, "Q4_0");
    test_round_trip_format(kKvQ4S, "Q4S (searched scale)");
    test_round_trip_format(kKvF16, "FP16");
}

// -----------------------------------------------------------------------------
// Test 3: 4096-byte alignment verification of serialized buffers
// -----------------------------------------------------------------------------
void test_4096_alignment_verification() {
    std::printf("[Test 3] 4096-byte memory alignment verification...\n");

    // 1. Raw allocator alignment
    for (size_t size : {1, 64, 128, 4095, 4096, 4097, 8192, 100000}) {
        uint8_t* ptr = kv_alloc_aligned(size, kKvBlockAlignment);
        CHECK(ptr != nullptr);
        CHECK((reinterpret_cast<uintptr_t>(ptr) % kKvBlockAlignment) == 0);
        kv_free_aligned(ptr);
    }

    // 2. AlignedBuffer RAII container
    AlignedBuffer buf(5000);
    CHECK(buf.is_aligned(kKvBlockAlignment));
    CHECK((reinterpret_cast<uintptr_t>(buf.data()) % kKvBlockAlignment) == 0);
    CHECK(buf.capacity() % kKvBlockAlignment == 0);

    // Resize container
    buf.resize(9000);
    CHECK(buf.is_aligned(kKvBlockAlignment));
    CHECK((reinterpret_cast<uintptr_t>(buf.data()) % kKvBlockAlignment) == 0);
    CHECK(buf.capacity() % kKvBlockAlignment == 0);

    // 3. Serialized output buffer alignment across various sizes
    for (size_t num_tokens : {1, 16, 64, 128}) {
        KvBlock block;
        block.header.format = kKvInt8;
        block.header.block_tokens = kKvBlockTokens;
        block.payload.token_ids = make_token_ids(num_tokens, 42);
        block.payload.k_data = make_random_bytes(num_tokens * 64, 1);
        block.payload.v_data = make_random_bytes(num_tokens * 64, 2);

        AlignedBuffer out_buf;
        KvBlockError err = kv_block_serialize(block, out_buf);
        CHECK(err == KvBlockError::kSuccess);

        CHECK(out_buf.is_aligned(kKvBlockAlignment));
        CHECK((reinterpret_cast<uintptr_t>(out_buf.data()) % kKvBlockAlignment) == 0);
        CHECK(out_buf.size() % kKvBlockAlignment == 0);

        const auto* hdr = reinterpret_cast<const KvBlockHeader*>(out_buf.data());
        CHECK(hdr->total_file_bytes == out_buf.size());
        CHECK(hdr->total_file_bytes % kKvBlockAlignment == 0);
    }
}

// -----------------------------------------------------------------------------
// Test 4: Corruption / tampering detection via checksum
// -----------------------------------------------------------------------------
void test_corruption_detection() {
    std::printf("[Test 4] Corruption and tampering detection...\n");

    KvBlock block;
    block.header.format = kKvInt8;
    block.header.block_tokens = kKvBlockTokens;
    block.header.parent_hash = 0xFEEDFACECAFEBEEFULL;
    block.payload.token_ids = make_token_ids(kKvBlockTokens, 1);
    block.payload.k_data = make_random_bytes(1024, 77);
    block.payload.v_data = make_random_bytes(1024, 88);

    AlignedBuffer buf;
    KvBlockError err = kv_block_serialize(block, buf);
    CHECK(err == KvBlockError::kSuccess);

    // 1. Untampered block must deserialize cleanly
    KvBlock valid_block;
    err = kv_block_deserialize(buf, valid_block, true);
    CHECK(err == KvBlockError::kSuccess);

    // 2. Tamper payload byte -> checksum mismatch
    {
        AlignedBuffer tampered(buf.size());
        std::memcpy(tampered.data(), buf.data(), buf.size());

        const auto* hdr = reinterpret_cast<const KvBlockHeader*>(buf.data());
        size_t tamper_pos = hdr->k_data_offset + 10;
        tampered.data()[tamper_pos] ^= 0xFF;

        KvBlock bad_block;
        err = kv_block_deserialize(tampered, bad_block, true);
        CHECK(err == KvBlockError::kCorruptedChecksum);
    }

    // 3. Tamper header field (e.g. parent_hash) -> checksum mismatch
    {
        AlignedBuffer tampered(buf.size());
        std::memcpy(tampered.data(), buf.data(), buf.size());

        auto* hdr = reinterpret_cast<KvBlockHeader*>(tampered.data());
        hdr->parent_hash ^= 0x1;

        KvBlock bad_block;
        err = kv_block_deserialize(tampered, bad_block, true);
        CHECK(err == KvBlockError::kCorruptedChecksum);
    }

    // 4. Tamper padding byte -> checksum mismatch
    {
        AlignedBuffer tampered(buf.size());
        std::memcpy(tampered.data(), buf.data(), buf.size());

        // Modify last byte in 4096-aligned padding
        tampered.data()[buf.size() - 1] ^= 0x42;

        KvBlock bad_block;
        err = kv_block_deserialize(tampered, bad_block, true);
        CHECK(err == KvBlockError::kCorruptedChecksum);
    }

    // 5. Tamper magic -> invalid magic error
    {
        AlignedBuffer tampered(buf.size());
        std::memcpy(tampered.data(), buf.data(), buf.size());

        tampered.data()[0] = 'X';

        KvBlock bad_block;
        err = kv_block_deserialize(tampered, bad_block, true);
        CHECK(err == KvBlockError::kInvalidMagic);
    }

    // 6. Truncated buffer -> buffer too small or truncated payload
    {
        KvBlock bad_block;
        err = kv_block_deserialize(buf.data(), sizeof(KvBlockHeader) - 1, bad_block, false);
        CHECK(err == KvBlockError::kBufferTooSmall);

        err = kv_block_deserialize(buf.data(), buf.size() - 1, bad_block, false);
        CHECK(err == KvBlockError::kTruncatedPayload);
    }

    // 7. Misaligned buffer pointer
    {
        // Allocate a vector with extra padding to construct an explicitly misaligned copy
        std::vector<uint8_t> unaligned_storage(buf.size() + 64);
        uint8_t* misaligned_ptr = unaligned_storage.data();
        if ((reinterpret_cast<uintptr_t>(misaligned_ptr) % kKvBlockAlignment) == 0) {
            misaligned_ptr += 8;
        }
        std::memcpy(misaligned_ptr, buf.data(), buf.size());

        KvBlock bad_block;
        // With require_alignment = true, misalignment must be rejected
        err = kv_block_deserialize(misaligned_ptr, buf.size(), bad_block, true);
        CHECK(err == KvBlockError::kMisalignedBuffer);

        // With require_alignment = false, valid unaligned data must be accepted
        KvBlock ok_block;
        err = kv_block_deserialize(misaligned_ptr, buf.size(), ok_block, false);
        CHECK(err == KvBlockError::kSuccess);
        CHECK(ok_block.payload == block.payload);
    }
}

// -----------------------------------------------------------------------------
// Test 5: Edge cases (empty payload, partial chunk, null pointers, overcapacity)
// -----------------------------------------------------------------------------
void test_edge_cases() {
    std::printf("[Test 5] Edge cases, capacity limits, and null handling...\n");

    // 1. Null token_ids handling in kv_block_hash
    uint64_t h_null = kv_block_hash(0, nullptr, 100, kKvInt8, 1, 2, 256);
    uint64_t h_zero = kv_block_hash(0, nullptr, 0, kKvInt8, 1, 2, 256);
    CHECK(h_null == h_zero);

    // 2. Empty block round-trip
    {
        KvBlock empty_block;
        empty_block.header.format = kKvInt8;
        empty_block.header.block_tokens = kKvBlockTokens;

        AlignedBuffer buf;
        KvBlockError err = kv_block_serialize(empty_block, buf);
        CHECK(err == KvBlockError::kSuccess);
        CHECK(buf.size() == kKvBlockAlignment);

        KvBlock restored;
        err = kv_block_deserialize(buf, restored, true);
        CHECK(err == KvBlockError::kSuccess);
        CHECK(restored.payload.token_ids.empty());
        CHECK(restored.payload.k_data.empty());
        CHECK(restored.header.token_count == 0);
    }

    // 3. Partial block (e.g. 17 tokens < 128 block_tokens)
    {
        KvBlock partial_block;
        partial_block.header.format = kKvInt8;
        partial_block.header.block_tokens = kKvBlockTokens;
        partial_block.payload.token_ids = make_token_ids(17, 33);
        partial_block.payload.k_data = make_random_bytes(17 * 64, 10);
        partial_block.payload.v_data = make_random_bytes(17 * 64, 20);

        AlignedBuffer buf;
        KvBlockError err = kv_block_serialize(partial_block, buf);
        CHECK(err == KvBlockError::kSuccess);

        KvBlock restored;
        err = kv_block_deserialize(buf, restored, true);
        CHECK(err == KvBlockError::kSuccess);
        CHECK(restored.header.token_count == 17);
        CHECK(restored.payload.token_ids == partial_block.payload.token_ids);
    }

    // 4. Overcapacity rejection during serialize (> block_tokens)
    {
        KvBlock overflow_block;
        overflow_block.header.format = kKvInt8;
        overflow_block.header.block_tokens = kKvBlockTokens;
        overflow_block.payload.token_ids = make_token_ids(kKvBlockTokens + 1, 1);

        AlignedBuffer buf;
        KvBlockError err = kv_block_serialize(overflow_block, buf);
        CHECK(err == KvBlockError::kTruncatedPayload);
    }

    // 5. Inconsistent section bytes sum vs payload_bytes in deserialize
    {
        KvBlock block;
        block.header.format = kKvInt8;
        block.header.block_tokens = kKvBlockTokens;
        block.payload.token_ids = make_token_ids(10, 1);
        block.payload.k_data = make_random_bytes(64, 2);

        AlignedBuffer buf;
        KvBlockError err = kv_block_serialize(block, buf);
        CHECK(err == KvBlockError::kSuccess);

        // Tamper payload_bytes in header and recompute checksum so checksum passes but section sum fails
        AlignedBuffer tampered(buf.size());
        std::memcpy(tampered.data(), buf.data(), buf.size());
        auto* hdr = reinterpret_cast<KvBlockHeader*>(tampered.data());
        hdr->payload_bytes += 16;
        hdr->checksum = 0;
        hdr->checksum = kv_compute_block_checksum(tampered.data(), hdr->total_file_bytes);

        KvBlock restored;
        err = kv_block_deserialize(tampered, restored, true);
        CHECK(err == KvBlockError::kTruncatedPayload);
    }

    // 6. token_count > block_tokens in deserialize
    {
        KvBlock block;
        block.header.format = kKvInt8;
        block.header.block_tokens = 10;
        block.payload.token_ids = make_token_ids(10, 1);

        AlignedBuffer buf;
        KvBlockError err = kv_block_serialize(block, buf);
        CHECK(err == KvBlockError::kSuccess);

        AlignedBuffer tampered(buf.size());
        std::memcpy(tampered.data(), buf.data(), buf.size());
        auto* hdr = reinterpret_cast<KvBlockHeader*>(tampered.data());
        hdr->block_tokens = 5; // now token_count (10) > block_tokens (5)
        hdr->checksum = 0;
        hdr->checksum = kv_compute_block_checksum(tampered.data(), hdr->total_file_bytes);

        KvBlock restored;
        err = kv_block_deserialize(tampered, restored, true);
        CHECK(err == KvBlockError::kTruncatedPayload);
    }
}

} // namespace

int main() {
    std::printf("Running kv_block_test...\n");
    test_cas_hash_determinism_and_chaining();
    test_serialize_deserialize_round_trip();
    test_4096_alignment_verification();
    test_corruption_detection();
    test_edge_cases();

    if (fails == 0) {
        std::printf("ALL KV BLOCK TESTS PASSED cleanly.\n");
        return 0;
    } else {
        std::printf("KV BLOCK TESTS FAILED: %d failures\n", fails);
        return 1;
    }
}
