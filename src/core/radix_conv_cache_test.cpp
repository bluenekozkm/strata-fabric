// src/core/radix_conv_cache_test.cpp - Unit tests for RadixConvCache.
#include "strata/core/radix_conv_cache.hpp"

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <thread>
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

// -----------------------------------------------------------------------------
// Test 1: Single branch insertion and exact prefix match
// -----------------------------------------------------------------------------
void test_single_branch_and_exact_match() {
    std::printf("[Test 1] Single branch insertion and exact prefix match...\n");

    RadixConvCacheConfig cfg;
    cfg.host_ram_budget_bytes = 100 * 1024 * 1024;
    RadixConvCache cache(cfg);

    const std::vector<int32_t> tokens = {10, 20, 30, 40, 50};
    auto node = cache.insert_sequence(tokens);
    CHECK(node != nullptr);
    CHECK(node->tokens == tokens);
    CHECK(node->node_id > 0);
    CHECK(node->block_hash != 0);

    // Exact match
    auto res_exact = cache.match_longest_prefix(tokens);
    CHECK(res_exact.matched_tokens == 5);
    CHECK(res_exact.is_exact == true);
    CHECK(res_exact.deepest_node == node);
    CHECK(res_exact.last_full_node == node);

    // Partial prefix query (shorter than stored node)
    const std::vector<int32_t> query_shorter = {10, 20, 30};
    auto res_shorter = cache.match_longest_prefix(query_shorter);
    CHECK(res_shorter.matched_tokens == 3);
    CHECK(res_shorter.is_exact == true);
    CHECK(res_shorter.deepest_node == node);
    CHECK(res_shorter.last_full_node == cache.root());

    // Query longer than stored node
    const std::vector<int32_t> query_longer = {10, 20, 30, 40, 50, 60, 70};
    auto res_longer = cache.match_longest_prefix(query_longer);
    CHECK(res_longer.matched_tokens == 5);
    CHECK(res_longer.is_exact == false);
    CHECK(res_longer.deepest_node == node);
    CHECK(res_longer.last_full_node == node);

    // Query with completely disjoint prefix
    const std::vector<int32_t> query_disjoint = {99, 100};
    auto res_disjoint = cache.match_longest_prefix(query_disjoint);
    CHECK(res_disjoint.matched_tokens == 0);
    CHECK(res_disjoint.is_exact == false);
    CHECK(res_disjoint.deepest_node == cache.root());
    CHECK(res_disjoint.last_full_node == cache.root());

    // Stats verification
    auto stats = cache.get_stats();
    CHECK(stats.total_nodes == 2); // root + node
    CHECK(stats.prefix_lookups == 4);
    CHECK(stats.prefix_hits == 3);
    CHECK(stats.prefix_misses == 1);
}

// -----------------------------------------------------------------------------
// Test 2: Multi-branch System Prompt coexistence
// -----------------------------------------------------------------------------
void test_multi_branch_system_prompts() {
    std::printf("[Test 2] Multi-branch System Prompt coexistence...\n");

    RadixConvCacheConfig cfg;
    RadixConvCache cache(cfg);

    const std::vector<int32_t> sys_prompt_a = {101, 102, 103, 104};
    const std::vector<int32_t> sys_prompt_b = {201, 202, 203, 204};

    auto node_a = cache.insert_sequence(sys_prompt_a, /*is_system_prompt=*/true, /*is_pinned=*/true);
    auto node_b = cache.insert_sequence(sys_prompt_b, /*is_system_prompt=*/true, /*is_pinned=*/true);

    CHECK(node_a != nullptr);
    CHECK(node_b != nullptr);
    CHECK(node_a->node_id != node_b->node_id);

    // Root must have both child branches coexisting cleanly
    CHECK(cache.root()->children.size() == 2);

    // Verify system prompt flags and pinning
    CHECK(cache.is_system_prompt(node_a->node_id));
    CHECK(cache.is_system_prompt(node_b->node_id));
    CHECK(cache.is_pinned(node_a->node_id));
    CHECK(cache.is_pinned(node_b->node_id));

    // Match prompt A
    auto match_a = cache.match_longest_prefix(sys_prompt_a);
    CHECK(match_a.matched_tokens == 4);
    CHECK(match_a.is_exact == true);
    CHECK(match_a.deepest_node == node_a);

    // Match prompt B
    auto match_b = cache.match_longest_prefix(sys_prompt_b);
    CHECK(match_b.matched_tokens == 4);
    CHECK(match_b.is_exact == true);
    CHECK(match_b.deepest_node == node_b);

    // Neither overwrites the other
    CHECK(node_a->tokens == sys_prompt_a);
    CHECK(node_b->tokens == sys_prompt_b);

    auto stats = cache.get_stats();
    CHECK(stats.system_prompt_nodes == 2);
    CHECK(stats.pinned_nodes == 2);
    CHECK(stats.total_nodes == 3); // root + A + B
}

// -----------------------------------------------------------------------------
// Test 3: Common prefix sharing and splitting
// -----------------------------------------------------------------------------
void test_common_prefix_sharing_and_splitting() {
    std::printf("[Test 3] Common prefix sharing and splitting...\n");

    RadixConvCacheConfig cfg;
    RadixConvCache cache(cfg);

    const std::vector<int32_t> prompt_a = {1, 2, 3, 4, 5, 6};
    const std::vector<int32_t> prompt_b = {1, 2, 3, 4, 7, 8};

    // Insert Prompt A
    auto node_a = cache.insert_sequence(prompt_a);
    CHECK(node_a != nullptr);
    CHECK(cache.root()->children.size() == 1);
    CHECK(cache.root()->children[0]->tokens == prompt_a);

    // Insert Prompt B with common prefix [1, 2, 3, 4]
    auto node_b = cache.insert_sequence(prompt_b);
    CHECK(node_b != nullptr);

    // Root must now have exactly 1 child: the split prefix node [1, 2, 3, 4]
    CHECK(cache.root()->children.size() == 1);
    auto split_node = cache.root()->children[0];
    CHECK(split_node != nullptr);

    const std::vector<int32_t> expected_prefix = {1, 2, 3, 4};
    CHECK(split_node->tokens == expected_prefix);

    // Split node must have two children: [5, 6] and [7, 8]
    CHECK(split_node->children.size() == 2);

    const std::vector<int32_t> expected_a_tail = {5, 6};
    const std::vector<int32_t> expected_b_tail = {7, 8};

    bool found_a = false;
    bool found_b = false;
    for (const auto& child : split_node->children) {
        if (child->tokens == expected_a_tail) found_a = true;
        if (child->tokens == expected_b_tail) found_b = true;
    }
    CHECK(found_a);
    CHECK(found_b);

    // Verify prefix matching on Prompt A and Prompt B
    auto match_a = cache.match_longest_prefix(prompt_a);
    CHECK(match_a.matched_tokens == 6);
    CHECK(match_a.is_exact == true);
    CHECK(match_a.deepest_node->tokens == expected_a_tail);

    auto match_b = cache.match_longest_prefix(prompt_b);
    CHECK(match_b.matched_tokens == 6);
    CHECK(match_b.is_exact == true);
    CHECK(match_b.deepest_node->tokens == expected_b_tail);

    // Prefix matching on the common prefix itself
    auto match_prefix = cache.match_longest_prefix(expected_prefix);
    CHECK(match_prefix.matched_tokens == 4);
    CHECK(match_prefix.is_exact == true);
    CHECK(match_prefix.deepest_node == split_node);
    CHECK(match_prefix.last_full_node == split_node);

    // Complete token sequence reconstruction from root
    CHECK(cache.get_sequence_tokens(match_a.deepest_node) == prompt_a);
    CHECK(cache.get_sequence_tokens(match_b.deepest_node) == prompt_b);
    CHECK(cache.get_sequence_tokens(split_node) == expected_prefix);
}

// -----------------------------------------------------------------------------
// Test 4: Host RAM budget and LRU eviction
// -----------------------------------------------------------------------------
void test_host_ram_budget_and_lru_eviction() {
    std::printf("[Test 4] Host RAM budget and LRU eviction...\n");

    RadixConvCacheConfig cfg;
    cfg.host_ram_budget_bytes = 0; // unlimited during initial setup
    cfg.auto_evict_on_insert = false;
    RadixConvCache cache(cfg);

    // Insert System Prompt A, System Prompt B, and a Pinned conversation
    const std::vector<int32_t> sys_a = {10, 11, 12};
    const std::vector<int32_t> sys_b = {20, 21, 22};
    const std::vector<int32_t> pinned_c = {30, 31, 32};

    auto node_sys_a = cache.insert_sequence(sys_a, /*is_system_prompt=*/true, /*is_pinned=*/true);
    auto node_sys_b = cache.insert_sequence(sys_b, /*is_system_prompt=*/true, /*is_pinned=*/false);
    auto node_pinned = cache.insert_sequence(pinned_c, /*is_system_prompt=*/false, /*is_pinned=*/true);

    uint64_t id_sys_a = node_sys_a->node_id;
    uint64_t id_sys_b = node_sys_b->node_id;
    uint64_t id_pinned = node_pinned->node_id;

    // Insert 20 chat turn branches under sys_a
    std::vector<uint64_t> chat_ids;
    for (int i = 0; i < 20; ++i) {
        std::vector<int32_t> chat = sys_a;
        chat.push_back(1000 + i);
        chat.push_back(2000 + i);
        auto n = cache.insert_sequence(chat);
        chat_ids.push_back(n->node_id);
        std::this_thread::sleep_for(std::chrono::microseconds(10));
    }

    CHECK(cache.total_nodes() > 20);

    // Target a budget that fits only the protected nodes and a few recent nodes
    size_t protected_bytes = cache.find_node(id_sys_a)->ram_bytes +
                             cache.find_node(id_sys_b)->ram_bytes +
                             cache.find_node(id_pinned)->ram_bytes +
                             cache.root()->ram_bytes;

    // Set budget to protected bytes + ~4 chat nodes
    size_t target_budget = protected_bytes + 4 * 200;

    cache.set_host_ram_budget_bytes(target_budget);
    size_t evicted = cache.evict_to_budget(0);
    CHECK(evicted > 0);

    // Verify protected nodes are strictly preserved
    CHECK(cache.find_node(id_sys_a) != nullptr);
    CHECK(cache.find_node(id_sys_b) != nullptr);
    CHECK(cache.find_node(id_pinned) != nullptr);

    CHECK(cache.is_system_prompt(id_sys_a));
    CHECK(cache.is_system_prompt(id_sys_b));
    CHECK(cache.is_pinned(id_pinned));

    // Verify oldest chat turns (chat_ids[0], chat_ids[1]...) were evicted
    CHECK(cache.find_node(chat_ids[0]) == nullptr);
    CHECK(cache.find_node(chat_ids[1]) == nullptr);

    // Verify prefix matches on protected nodes still work perfectly
    auto match_sys_a = cache.match_longest_prefix(sys_a);
    CHECK(match_sys_a.is_exact == true);
    CHECK(match_sys_a.matched_tokens == sys_a.size());

    auto match_sys_b = cache.match_longest_prefix(sys_b);
    CHECK(match_sys_b.is_exact == true);
    CHECK(match_sys_b.matched_tokens == sys_b.size());

    auto match_pinned = cache.match_longest_prefix(pinned_c);
    CHECK(match_pinned.is_exact == true);
    CHECK(match_pinned.matched_tokens == pinned_c.size());

    auto stats = cache.get_stats();
    CHECK(stats.evictions > 0);
    CHECK(stats.bytes_evicted > 0);
}

// -----------------------------------------------------------------------------
// Test 5: Concurrent read/write stress test
// -----------------------------------------------------------------------------
void test_concurrent_read_write_stress() {
    std::printf("[Test 5] Concurrent read/write stress test...\n");

    RadixConvCacheConfig cfg;
    cfg.host_ram_budget_bytes = 100 * 1024 * 1024;
    RadixConvCache cache(cfg);

    // Seed initial branches
    cache.insert_sequence({1, 2, 3, 4});
    cache.insert_sequence({1, 2, 5, 6});
    cache.insert_sequence({10, 20, 30, 40});

    std::atomic<bool> stop_flag{false};
    std::atomic<uint64_t> total_reads{0};
    std::atomic<uint64_t> total_writes{0};

    // Writer thread
    std::thread writer([&]() {
        int seq_num = 0;
        while (!stop_flag.load(std::memory_order_relaxed)) {
            std::vector<int32_t> seq = {1, 2, seq_num % 10, seq_num};
            cache.insert_sequence(seq);
            total_writes.fetch_add(1, std::memory_order_relaxed);
            ++seq_num;
            if (seq_num > 1000) seq_num = 0;
            std::this_thread::yield();
        }
    });

    // 4 Reader threads
    std::vector<std::thread> readers;
    for (int r = 0; r < 4; ++r) {
        readers.emplace_back([&, r]() {
            int query_id = r;
            while (!stop_flag.load(std::memory_order_relaxed)) {
                std::vector<int32_t> query = {1, 2, query_id % 10, query_id};
                auto res = cache.match_longest_prefix(query);
                CHECK(res.matched_tokens >= 2); // At least {1, 2} must match
                CHECK(res.deepest_node != nullptr);
                total_reads.fetch_add(1, std::memory_order_relaxed);
                query_id = (query_id + 1) % 50;
            }
        });
    }

    // Run concurrency test for 300 ms
    std::this_thread::sleep_for(std::chrono::milliseconds(300));
    stop_flag.store(true);

    writer.join();
    for (auto& t : readers) {
        t.join();
    }

    CHECK(total_writes.load() > 100);
    CHECK(total_reads.load() > 1000);

    auto stats = cache.get_stats();
    CHECK(stats.prefix_lookups > 1000);
    CHECK(stats.prefix_hits > 1000);
}

// -----------------------------------------------------------------------------
// Test 6: Branch retrieval & CAS Block Hashing
// -----------------------------------------------------------------------------
void test_branch_retrieval_and_cas_hashes() {
    std::printf("[Test 6] Branch retrieval & CAS Block Hashing...\n");

    RadixConvCache cache;

    const std::vector<int32_t> seq1 = {100, 101, 102};
    const std::vector<int32_t> seq2 = {100, 101, 102, 103, 104};

    auto node1 = cache.insert_sequence(seq1);
    auto node2 = cache.insert_sequence(seq2);

    auto tokens2 = cache.get_sequence_tokens(node2);
    CHECK(tokens2 == seq2);

    auto branch_nodes = cache.get_branch_nodes(node2);
    CHECK(branch_nodes.size() == 3); // root, node1, node2
    CHECK(branch_nodes[0] == cache.root());
    CHECK(branch_nodes[1] == node1);
    CHECK(branch_nodes[2] == node2);

    auto hashes = cache.get_branch_block_hashes(node2);
    CHECK(hashes.size() == 2);
    CHECK(hashes[0] == node1->block_hash);
    CHECK(hashes[1] == node2->block_hash);
    CHECK(hashes[0] != hashes[1]);
}

// -----------------------------------------------------------------------------
// Test 7: KV Block Attachment
// -----------------------------------------------------------------------------
void test_kv_block_attachment() {
    std::printf("[Test 7] KV Block Attachment...\n");

    RadixConvCache cache;
    auto node = cache.insert_sequence({500, 501, 502});
    CHECK(node != nullptr);

    size_t ram_before = cache.total_ram_bytes();

    auto block = std::make_shared<KvBlock>();
    block->header.token_count = 3;
    block->payload.token_ids = {500, 501, 502};
    block->payload.k_data.resize(1024, 42);
    block->payload.v_data.resize(1024, 84);

    CHECK(cache.set_kv_block(node->node_id, block));
    size_t ram_after = cache.total_ram_bytes();
    CHECK(ram_after > ram_before);

    auto retrieved_block = cache.get_kv_block(node->node_id);
    CHECK(retrieved_block != nullptr);
    CHECK(retrieved_block->payload.k_data.size() == 1024);
    CHECK(retrieved_block->payload.k_data[0] == 42);

    auto blocks = cache.get_branch_blocks(node);
    CHECK(blocks.size() == 1);
    CHECK(blocks[0] == retrieved_block);
}

// -----------------------------------------------------------------------------
// Test 8: Node Pinning, Unpinning and Removal
// -----------------------------------------------------------------------------
void test_pinning_and_removal() {
    std::printf("[Test 8] Node Pinning, Unpinning and Removal...\n");

    RadixConvCache cache;
    auto node = cache.insert_sequence({700, 701});
    uint64_t id = node->node_id;

    // Pin
    CHECK(cache.pin_node(id, true));
    CHECK(cache.is_pinned(id));

    // Removal should fail when pinned and force=false
    CHECK(!cache.remove_node(id, /*force=*/false));
    CHECK(cache.find_node(id) != nullptr);

    // Unpin
    CHECK(cache.unpin_node(id));
    CHECK(!cache.is_pinned(id));

    // Removal should succeed now
    CHECK(cache.remove_node(id, /*force=*/false));
    CHECK(cache.find_node(id) == nullptr);
}

// -----------------------------------------------------------------------------
// Test 8b: Explicit pin scope (the persist_tiered_kv pattern) and shared-prefix pin survival
// -----------------------------------------------------------------------------
void test_explicit_pin_scope_and_convergence() {
    std::printf("[Test 8b] Explicit pin scope and shared-prefix pin survival...\n");

    RadixConvCacheConfig cfg;
    cfg.host_ram_budget_bytes = 0; // unlimited during initial setup
    cfg.auto_evict_on_insert = false;
    RadixConvCache cache(cfg);

    const std::vector<int32_t> pinned_seq = {10, 20, 30};
    const std::vector<int32_t> tail_seq = {40, 50, 60};

    // The persist_tiered_kv pattern: insert with no flags, then set the pin explicitly.
    // Production pins set-only (never pin_node(false)): a shared prefix node may carry another
    // parked conversation's pin, and a clear could not stick anyway - insert_sequence only
    // latches true.
    auto pinned = cache.insert_sequence(pinned_seq, /*is_system_prompt=*/false, /*is_pinned=*/false);
    auto tail = cache.insert_sequence(tail_seq, /*is_system_prompt=*/false, /*is_pinned=*/false);
    CHECK(cache.pin_node(pinned->node_id, true));
    CHECK(cache.mark_system_prompt(pinned->node_id, true));
    CHECK(cache.is_pinned(pinned->node_id));
    CHECK(cache.is_system_prompt(pinned->node_id));

    // Budget fits only root and the pinned node: the unpinned tail must go, the pin stays
    const size_t protected_bytes = cache.root()->ram_bytes + pinned->ram_bytes;
    cache.set_host_ram_budget_bytes(protected_bytes);
    CHECK(cache.evict_to_budget(0) >= 1);
    CHECK(cache.find_node(pinned->node_id) != nullptr);
    CHECK(cache.find_node(tail->node_id) == nullptr);

    // A second conversation parks on the same shared prefix WITHOUT a pin: its own tail is
    // evictable, and it must NOT clear the first conversation's pin on the shared node
    auto shared_tail = cache.insert_sequence({10, 20, 30, 99},
                                             /*is_system_prompt=*/false, /*is_pinned=*/false);
    CHECK(shared_tail != nullptr);
    CHECK(!cache.is_pinned(shared_tail->node_id));
    CHECK(cache.is_pinned(pinned->node_id)); // the shared node keeps its pin

    cache.set_host_ram_budget_bytes(protected_bytes);
    CHECK(cache.evict_to_budget(0) >= 1);
    CHECK(cache.find_node(shared_tail->node_id) == nullptr); // the second tail goes
    CHECK(cache.find_node(pinned->node_id) != nullptr);      // the shared pin survives
}

// -----------------------------------------------------------------------------
// Test 9: Node splitting invalidates stale KV block on trimmed child
// -----------------------------------------------------------------------------
void test_node_split_with_kv_block_invalidation() {
    std::printf("[Test 9] Node splitting with KV block invalidation...\n");

    RadixConvCache cache;
    auto node = cache.insert_sequence({10, 20, 30, 40});
    CHECK(node != nullptr);

    auto block = std::make_shared<KvBlock>();
    block->header.token_count = 4;
    block->payload.token_ids = {10, 20, 30, 40};
    block->payload.k_data.resize(512, 1);
    CHECK(cache.set_kv_block(node->node_id, block));
    CHECK(cache.get_kv_block(node->node_id) != nullptr);

    size_t ram_with_block = cache.total_ram_bytes();

    // Now insert a diverging sequence {10, 20, 50, 60}
    // This splits node into prefix {10, 20} and tail {30, 40}.
    // The original block for {10, 20, 30, 40} must NOT remain on the tail node {30, 40}.
    auto new_node = cache.insert_sequence({10, 20, 50, 60});
    CHECK(new_node != nullptr);

    // Node is now the tail child holding only {30, 40}
    CHECK(node->tokens == std::vector<int32_t>({30, 40}));
    // Block must have been reset to prevent mismatched KV state
    CHECK(cache.get_kv_block(node->node_id) == nullptr);
    CHECK(cache.total_ram_bytes() < ram_with_block);
}

// -----------------------------------------------------------------------------
// Test 10: System prompts with common prefix sharing
// -----------------------------------------------------------------------------
void test_system_prompts_with_common_prefix() {
    std::printf("[Test 10] System prompts with common prefix sharing...\n");

    RadixConvCache cache;

    // Two system prompts that share prefix {1, 2, 3}
    const std::vector<int32_t> sys1 = {1, 2, 3, 10, 20};
    const std::vector<int32_t> sys2 = {1, 2, 3, 30, 40};

    auto n1 = cache.insert_sequence(sys1, /*is_system_prompt=*/true, /*is_pinned=*/true);
    auto n2 = cache.insert_sequence(sys2, /*is_system_prompt=*/true, /*is_pinned=*/true);

    CHECK(n1 != nullptr);
    CHECK(n2 != nullptr);
    CHECK(n1->node_id != n2->node_id);

    // Root should have 1 child (common prefix {1, 2, 3})
    CHECK(cache.root()->children.size() == 1);
    auto prefix_node = cache.root()->children[0];
    CHECK(prefix_node->tokens == std::vector<int32_t>({1, 2, 3}));
    CHECK(cache.is_system_prompt(prefix_node->node_id));
    CHECK(cache.is_pinned(prefix_node->node_id));

    // Both system prompts should be fully accessible and pinned
    CHECK(cache.is_system_prompt(n1->node_id));
    CHECK(cache.is_pinned(n1->node_id));
    CHECK(cache.is_system_prompt(n2->node_id));
    CHECK(cache.is_pinned(n2->node_id));

    auto m1 = cache.match_longest_prefix(sys1);
    CHECK(m1.is_exact == true);
    CHECK(m1.matched_tokens == 5);

    auto m2 = cache.match_longest_prefix(sys2);
    CHECK(m2.is_exact == true);
    CHECK(m2.matched_tokens == 5);
}

// -----------------------------------------------------------------------------
// Test 11: Concurrent branch retrieval and no shared_mutex deadlock
// -----------------------------------------------------------------------------
void test_branch_retrieval_concurrency_no_deadlock() {
    std::printf("[Test 11] Concurrent branch retrieval and no shared_mutex deadlock...\n");

    RadixConvCache cache;
    auto node = cache.insert_sequence({100, 200, 300, 400});
    uint64_t target_id = node->node_id;

    std::atomic<bool> stop{false};
    std::atomic<uint64_t> read_ops{0};

    // Reader thread continuously calls get_branch_nodes(target_id) and get_sequence_tokens(target_id)
    std::thread reader([&]() {
        while (!stop.load(std::memory_order_relaxed)) {
            auto branch = cache.get_branch_nodes(target_id);
            auto seq = cache.get_sequence_tokens(target_id);
            auto hashes = cache.get_branch_block_hashes(target_id);
            auto blocks = cache.get_branch_blocks(target_id);
            read_ops.fetch_add(1, std::memory_order_relaxed);
        }
    });

    // Writer thread mutates other sequences
    std::thread writer([&]() {
        for (int i = 0; i < 200; ++i) {
            std::vector<int32_t> seq = {100, 200, i + 500, i + 600};
            cache.insert_sequence(seq);
            std::this_thread::yield();
        }
    });

    writer.join();
    stop.store(true);
    reader.join();

    CHECK(read_ops.load() > 50);
}

} // namespace

int main() {
    std::printf("Running radix_conv_cache_test...\n");
    test_single_branch_and_exact_match();
    test_multi_branch_system_prompts();
    test_common_prefix_sharing_and_splitting();
    test_host_ram_budget_and_lru_eviction();
    test_concurrent_read_write_stress();
    test_branch_retrieval_and_cas_hashes();
    test_kv_block_attachment();
    test_pinning_and_removal();
    test_explicit_pin_scope_and_convergence();
    test_node_split_with_kv_block_invalidation();
    test_system_prompts_with_common_prefix();
    test_branch_retrieval_concurrency_no_deadlock();

    if (fails == 0) {
        std::printf("ALL RADIX CONV CACHE TESTS PASSED cleanly.\n");
        return 0;
    } else {
        std::printf("TESTS FAILED with %d failures.\n", fails);
        return 1;
    }
}
