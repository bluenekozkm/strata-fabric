// src/core/radix_conv_cache.cpp - Host-side multi-branch Radix Prefix Tree implementation.
#include "strata/core/radix_conv_cache.hpp"

#include <algorithm>
#include <chrono>
#include <climits>
#include <utility>

namespace strata::core {

namespace {

inline uint64_t current_time_us() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::microseconds>(
            std::chrono::steady_clock::now().time_since_epoch()).count());
}

} // namespace

size_t RadixKvNode::compute_ram_bytes() const noexcept {
    size_t bytes = sizeof(RadixKvNode);
    bytes += tokens.capacity() * sizeof(int32_t);
    bytes += children.capacity() * sizeof(std::shared_ptr<RadixKvNode>);
    if (kv_block) {
        bytes += sizeof(KvBlock) + kv_block->payload.total_payload_bytes();
    }
    return bytes;
}

RadixConvCache::RadixConvCache() : RadixConvCache(RadixConvCacheConfig{}) {}

RadixConvCache::RadixConvCache(RadixConvCacheConfig config) : config_(std::move(config)) {
    init_root();
}

RadixConvCache::~RadixConvCache() = default;

void RadixConvCache::init_root() {
    root_ = std::make_shared<RadixKvNode>();
    root_->node_id = 0;
    root_->tokens.clear();
    root_->block_hash = 0;
    root_->last_accessed_us.store(current_time_us(), std::memory_order_relaxed);
    root_->ram_bytes = root_->compute_ram_bytes();
    node_map_[0] = root_;
    total_ram_bytes_ = root_->ram_bytes;
}

PrefixMatchResult RadixConvCache::match_longest_prefix(const int32_t* tokens, size_t num_tokens) const {
    prefix_lookups_.fetch_add(1, std::memory_order_relaxed);

    if (tokens == nullptr || num_tokens == 0) {
        prefix_hits_.fetch_add(1, std::memory_order_relaxed);
        std::shared_lock<std::shared_mutex> lock(rw_mu_);
        return {root_, root_, 0, true};
    }

    std::shared_lock<std::shared_mutex> lock(rw_mu_);

    std::shared_ptr<RadixKvNode> current = root_;
    std::shared_ptr<RadixKvNode> deepest_node = root_;
    std::shared_ptr<RadixKvNode> last_full_node = root_;
    size_t matched_tokens = 0;
    const uint64_t now = current_time_us();

    while (matched_tokens < num_tokens) {
        int32_t next_token = tokens[matched_tokens];
        std::shared_ptr<RadixKvNode> matched_child = nullptr;

        for (const auto& child : current->children) {
            if (!child->tokens.empty() && child->tokens[0] == next_token) {
                matched_child = child;
                break;
            }
        }

        if (!matched_child) {
            break;
        }

        size_t remaining = num_tokens - matched_tokens;
        size_t max_compare = std::min(matched_child->tokens.size(), remaining);
        size_t common_len = 0;
        while (common_len < max_compare &&
               matched_child->tokens[common_len] == tokens[matched_tokens + common_len]) {
            ++common_len;
        }

        matched_tokens += common_len;
        deepest_node = matched_child;
        matched_child->last_accessed_us.store(now, std::memory_order_relaxed);

        if (common_len == matched_child->tokens.size()) {
            last_full_node = matched_child;
            current = matched_child;
        } else {
            // Diverged partially within this child
            break;
        }
    }

    if (matched_tokens > 0) {
        prefix_hits_.fetch_add(1, std::memory_order_relaxed);
    } else {
        prefix_misses_.fetch_add(1, std::memory_order_relaxed);
    }

    bool is_exact = (matched_tokens == num_tokens);
    return {deepest_node, last_full_node, matched_tokens, is_exact};
}

PrefixMatchResult RadixConvCache::match_longest_prefix(const std::vector<int32_t>& tokens) const {
    return match_longest_prefix(tokens.data(), tokens.size());
}

std::shared_ptr<RadixKvNode> RadixConvCache::insert_sequence(
    const int32_t* tokens,
    size_t num_tokens,
    bool is_system_prompt,
    bool is_pinned,
    std::shared_ptr<KvBlock> block) {

    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    const uint64_t now = current_time_us();

    if (tokens == nullptr || num_tokens == 0) {
        if (is_system_prompt) root_->is_system_prompt.store(true, std::memory_order_relaxed);
        if (is_pinned) root_->is_pinned.store(true, std::memory_order_relaxed);
        root_->last_accessed_us.store(now, std::memory_order_relaxed);
        return root_;
    }

    std::shared_ptr<RadixKvNode> current = root_;
    size_t token_idx = 0;

    while (token_idx < num_tokens) {
        int32_t next_token = tokens[token_idx];
        std::shared_ptr<RadixKvNode> matched_child = nullptr;

        for (const auto& child : current->children) {
            if (!child->tokens.empty() && child->tokens[0] == next_token) {
                matched_child = child;
                break;
            }
        }

        if (!matched_child) {
            // No matching child branch: create new node for all remaining tokens
            auto new_node = std::make_shared<RadixKvNode>();
            new_node->node_id = next_node_id_++;
            new_node->parent = current;
            new_node->tokens.assign(tokens + token_idx, tokens + num_tokens);
            new_node->block_hash = kv_block_hash(
                current->block_hash,
                new_node->tokens.data(),
                new_node->tokens.size(),
                config_.format,
                config_.num_layers,
                config_.num_heads,
                config_.head_dim);
            new_node->is_system_prompt.store(is_system_prompt, std::memory_order_relaxed);
            new_node->is_pinned.store(is_pinned, std::memory_order_relaxed);
            new_node->last_accessed_us.store(now, std::memory_order_relaxed);
            if (config_.host_ram_budget_bytes > 0) {
                if (config_.host_ram_budget_bytes > 0) {
            new_node->kv_block = std::move(block);
        }
            }
            new_node->ram_bytes = new_node->compute_ram_bytes();

            total_ram_bytes_ -= current->ram_bytes;
            current->children.push_back(new_node);
            current->ram_bytes = current->compute_ram_bytes();
            total_ram_bytes_ += current->ram_bytes;

            node_map_[new_node->node_id] = new_node;
            total_ram_bytes_ += new_node->ram_bytes;

            if (config_.auto_evict_on_insert) {
                evict_to_budget_locked(0);
            }
            return new_node;
        }

        // Matched child found, find longest common prefix length
        size_t remaining = num_tokens - token_idx;
        size_t max_compare = std::min(matched_child->tokens.size(), remaining);
        size_t common_len = 0;
        while (common_len < max_compare &&
               matched_child->tokens[common_len] == tokens[token_idx + common_len]) {
            ++common_len;
        }

        if (common_len == matched_child->tokens.size()) {
            // Full match of child's tokens
            matched_child->last_accessed_us.store(now, std::memory_order_relaxed);
            token_idx += common_len;
            if (token_idx == num_tokens) {
                // Exact match ends at this node
                if (is_system_prompt) matched_child->is_system_prompt.store(true, std::memory_order_relaxed);
                if (is_pinned) matched_child->is_pinned.store(true, std::memory_order_relaxed);
                if (block && config_.host_ram_budget_bytes > 0) {
                    total_ram_bytes_ -= matched_child->ram_bytes;
                    matched_child->kv_block = std::move(block);
                    matched_child->ram_bytes = matched_child->compute_ram_bytes();
                    total_ram_bytes_ += matched_child->ram_bytes;
                }
                if (config_.auto_evict_on_insert) {
                    evict_to_budget_locked(0);
                }
                return matched_child;
            }
            // More tokens remaining: traverse down
            current = matched_child;
            continue;
        }

        // Divergence within matched_child: split node at common_len!
        auto split_node = std::make_shared<RadixKvNode>();
        split_node->node_id = next_node_id_++;
        split_node->parent = current;
        split_node->tokens.assign(matched_child->tokens.begin(),
                                  matched_child->tokens.begin() + common_len);
        split_node->block_hash = kv_block_hash(
            current->block_hash,
            split_node->tokens.data(),
            split_node->tokens.size(),
            config_.format,
            config_.num_layers,
            config_.num_heads,
            config_.head_dim);
        split_node->is_system_prompt.store(
            matched_child->is_system_prompt.load(std::memory_order_relaxed) || is_system_prompt,
            std::memory_order_relaxed);
        split_node->is_pinned.store(
            matched_child->is_pinned.load(std::memory_order_relaxed) || is_pinned,
            std::memory_order_relaxed);
        split_node->last_accessed_us.store(now, std::memory_order_relaxed);

        // Adjust matched_child to hold remainder
        std::vector<int32_t> remainder_tokens(matched_child->tokens.begin() + common_len,
                                              matched_child->tokens.end());
        matched_child->tokens = std::move(remainder_tokens);
        matched_child->parent = split_node;
        matched_child->block_hash = kv_block_hash(
            split_node->block_hash,
            matched_child->tokens.data(),
            matched_child->tokens.size(),
            config_.format,
            config_.num_layers,
            config_.num_heads,
            config_.head_dim);

        // Invalidate pre-split kv_block since it covered the unsplit sequence
        total_ram_bytes_ -= matched_child->ram_bytes;
        if (matched_child->kv_block) {
            matched_child->kv_block.reset();
        }
        matched_child->ram_bytes = matched_child->compute_ram_bytes();
        total_ram_bytes_ += matched_child->ram_bytes;

        update_subtree_hashes_locked(matched_child);

        // Attach matched_child to split_node
        split_node->children.push_back(matched_child);

        // Replace matched_child with split_node in current->children
        for (auto& child_ptr : current->children) {
            if (child_ptr == matched_child) {
                child_ptr = split_node;
                break;
            }
        }

        node_map_[split_node->node_id] = split_node;

        token_idx += common_len;
        if (token_idx == num_tokens) {
            // Target sequence ends exactly at split_node
            if (is_system_prompt) split_node->is_system_prompt.store(true, std::memory_order_relaxed);
            if (is_pinned) split_node->is_pinned.store(true, std::memory_order_relaxed);
            if (block && config_.host_ram_budget_bytes > 0) {
                split_node->kv_block = std::move(block);
            }
            split_node->ram_bytes = split_node->compute_ram_bytes();
            total_ram_bytes_ += split_node->ram_bytes;

            if (config_.auto_evict_on_insert) {
                evict_to_budget_locked(0);
            }
            return split_node;
        }

        // Target sequence has remaining tokens: create new sibling under split_node
        auto new_node = std::make_shared<RadixKvNode>();
        new_node->node_id = next_node_id_++;
        new_node->parent = split_node;
        new_node->tokens.assign(tokens + token_idx, tokens + num_tokens);
        new_node->block_hash = kv_block_hash(
            split_node->block_hash,
            new_node->tokens.data(),
            new_node->tokens.size(),
            config_.format,
            config_.num_layers,
            config_.num_heads,
            config_.head_dim);
        new_node->is_system_prompt.store(is_system_prompt, std::memory_order_relaxed);
        new_node->is_pinned.store(is_pinned, std::memory_order_relaxed);
        new_node->last_accessed_us.store(now, std::memory_order_relaxed);
        if (config_.host_ram_budget_bytes > 0) {
            new_node->kv_block = std::move(block);
        }
        new_node->ram_bytes = new_node->compute_ram_bytes();

        split_node->children.push_back(new_node);
        split_node->ram_bytes = split_node->compute_ram_bytes();
        total_ram_bytes_ += split_node->ram_bytes;

        node_map_[new_node->node_id] = new_node;
        total_ram_bytes_ += new_node->ram_bytes;

        if (config_.auto_evict_on_insert) {
            evict_to_budget_locked(0);
        }
        return new_node;
    }

    return current;
}

std::shared_ptr<RadixKvNode> RadixConvCache::insert_sequence(
    const std::vector<int32_t>& tokens,
    bool is_system_prompt,
    bool is_pinned,
    std::shared_ptr<KvBlock> block) {
    return insert_sequence(tokens.data(), tokens.size(), is_system_prompt, is_pinned, std::move(block));
}

void RadixConvCache::update_subtree_hashes_locked(const std::shared_ptr<RadixKvNode>& node) {
    for (const auto& child : node->children) {
        child->block_hash = kv_block_hash(
            node->block_hash,
            child->tokens.data(),
            child->tokens.size(),
            config_.format,
            config_.num_layers,
            config_.num_heads,
            config_.head_dim);
        if (child->kv_block) {
            child->kv_block->header.parent_hash = node->block_hash;
            child->kv_block->header.block_hash = child->block_hash;
        }
        update_subtree_hashes_locked(child);
    }
}

bool RadixConvCache::mark_system_prompt(uint64_t node_id, bool is_system) {
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return false;
    it->second->is_system_prompt.store(is_system, std::memory_order_relaxed);
    return true;
}

bool RadixConvCache::mark_system_prompt(const std::shared_ptr<RadixKvNode>& node, bool is_system) {
    if (!node) return false;
    return mark_system_prompt(node->node_id, is_system);
}

bool RadixConvCache::pin_node(uint64_t node_id, bool pinned) {
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return false;
    it->second->is_pinned.store(pinned, std::memory_order_relaxed);
    return true;
}

bool RadixConvCache::pin_node(const std::shared_ptr<RadixKvNode>& node, bool pinned) {
    if (!node) return false;
    return pin_node(node->node_id, pinned);
}

bool RadixConvCache::unpin_node(uint64_t node_id) {
    return pin_node(node_id, false);
}

bool RadixConvCache::unpin_node(const std::shared_ptr<RadixKvNode>& node) {
    if (!node) return false;
    return unpin_node(node->node_id);
}

bool RadixConvCache::is_pinned(uint64_t node_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return false;
    return it->second->is_pinned.load(std::memory_order_relaxed);
}

bool RadixConvCache::is_system_prompt(uint64_t node_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return false;
    return it->second->is_system_prompt.load(std::memory_order_relaxed);
}

void RadixConvCache::set_host_ram_budget_bytes(size_t bytes) {
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    config_.host_ram_budget_bytes = bytes;
}

void RadixConvCache::set_max_nodes(size_t max_nodes) {
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    config_.max_nodes = max_nodes;
}

size_t RadixConvCache::evict_bytes(size_t bytes_to_free) {
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    size_t freed = 0;
    size_t evicted_count = 0;

    while (freed < bytes_to_free) {
        std::shared_ptr<RadixKvNode> oldest_leaf = nullptr;
        uint64_t oldest_time = UINT64_MAX;

        for (const auto& [id, node] : node_map_) {
            if (node == root_) continue;
            if (!node->children.empty()) continue;
            if (node->is_pinned.load(std::memory_order_relaxed)) continue;
            if (node->is_system_prompt.load(std::memory_order_relaxed)) continue;

            uint64_t access_time = node->last_accessed_us.load(std::memory_order_relaxed);
            if (access_time < oldest_time) {
                oldest_time = access_time;
                oldest_leaf = node;
            }
        }

        if (!oldest_leaf) break;

        auto parent = oldest_leaf->parent.lock();
        if (parent) {
            total_ram_bytes_ -= parent->ram_bytes;
            auto& p_children = parent->children;
            for (auto it = p_children.begin(); it != p_children.end(); ++it) {
                if (*it == oldest_leaf) {
                    p_children.erase(it);
                    break;
                }
            }
            parent->ram_bytes = parent->compute_ram_bytes();
            total_ram_bytes_ += parent->ram_bytes;
        }

        size_t leaf_bytes = oldest_leaf->ram_bytes;
        freed += leaf_bytes;
        if (total_ram_bytes_ >= leaf_bytes) {
            total_ram_bytes_ -= leaf_bytes;
        } else {
            total_ram_bytes_ = 0;
        }

        bytes_evicted_ += leaf_bytes;
        evictions_++;
        node_map_.erase(oldest_leaf->node_id);
        evicted_count++;
    }

    return evicted_count;
}

size_t RadixConvCache::evict_to_budget(size_t additional_bytes) {
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    return evict_to_budget_locked(additional_bytes);
}

size_t RadixConvCache::evict_to_budget_locked(size_t additional_bytes) {
    size_t evicted_count = 0;

    auto over_budget = [&]() {
        bool ram_over = (config_.host_ram_budget_bytes > 0 &&
                         (total_ram_bytes_ + additional_bytes > config_.host_ram_budget_bytes));
        bool nodes_over = (config_.max_nodes > 0 &&
                           node_map_.size() > config_.max_nodes);
        return ram_over || nodes_over;
    };

    while (over_budget()) {
        std::shared_ptr<RadixKvNode> oldest_leaf = nullptr;
        uint64_t oldest_time = UINT64_MAX;

        for (const auto& [id, node] : node_map_) {
            if (node == root_) continue;
            if (!node->children.empty()) continue; // Only evict leaf nodes
            if (node->is_pinned.load(std::memory_order_relaxed)) continue; // Protected
            if (node->is_system_prompt.load(std::memory_order_relaxed)) continue; // Protected

            uint64_t access_time = node->last_accessed_us.load(std::memory_order_relaxed);
            if (access_time < oldest_time) {
                oldest_time = access_time;
                oldest_leaf = node;
            }
        }

        if (!oldest_leaf) {
            // No evictable leaf nodes available
            break;
        }

        // Remove oldest_leaf from parent's children
        auto parent = oldest_leaf->parent.lock();
        if (parent) {
            total_ram_bytes_ -= parent->ram_bytes;
            auto& p_children = parent->children;
            for (auto it = p_children.begin(); it != p_children.end(); ++it) {
                if (*it == oldest_leaf) {
                    p_children.erase(it);
                    break;
                }
            }
            parent->ram_bytes = parent->compute_ram_bytes();
            total_ram_bytes_ += parent->ram_bytes;
        }

        size_t leaf_bytes = oldest_leaf->ram_bytes;
        if (total_ram_bytes_ >= leaf_bytes) {
            total_ram_bytes_ -= leaf_bytes;
        } else {
            total_ram_bytes_ = 0;
        }

        bytes_evicted_ += leaf_bytes;
        evictions_++;
        node_map_.erase(oldest_leaf->node_id);
        evicted_count++;
    }

    return evicted_count;
}

std::shared_ptr<RadixKvNode> RadixConvCache::find_node(uint64_t node_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it != node_map_.end()) {
        return it->second;
    }
    return nullptr;
}

std::vector<std::shared_ptr<RadixKvNode>> RadixConvCache::get_branch_nodes_locked(
    const std::shared_ptr<RadixKvNode>& node) const {
    if (!node) return {};
    std::vector<std::shared_ptr<RadixKvNode>> path;
    auto curr = node;
    while (curr) {
        path.push_back(curr);
        curr = curr->parent.lock();
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<std::shared_ptr<RadixKvNode>> RadixConvCache::get_branch_nodes(
    const std::shared_ptr<RadixKvNode>& node) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    return get_branch_nodes_locked(node);
}

std::vector<std::shared_ptr<RadixKvNode>> RadixConvCache::get_branch_nodes(uint64_t node_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return {};
    return get_branch_nodes_locked(it->second);
}

std::vector<int32_t> RadixConvCache::get_sequence_tokens(const std::shared_ptr<RadixKvNode>& node) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto nodes = get_branch_nodes_locked(node);
    std::vector<int32_t> seq;
    for (const auto& n : nodes) {
        seq.insert(seq.end(), n->tokens.begin(), n->tokens.end());
    }
    return seq;
}

std::vector<int32_t> RadixConvCache::get_sequence_tokens(uint64_t node_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return {};
    auto nodes = get_branch_nodes_locked(it->second);
    std::vector<int32_t> seq;
    for (const auto& n : nodes) {
        seq.insert(seq.end(), n->tokens.begin(), n->tokens.end());
    }
    return seq;
}

std::vector<uint64_t> RadixConvCache::get_branch_block_hashes(
    const std::shared_ptr<RadixKvNode>& node) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto nodes = get_branch_nodes_locked(node);
    std::vector<uint64_t> hashes;
    hashes.reserve(nodes.size());
    for (const auto& n : nodes) {
        if (n != root_) {
            hashes.push_back(n->block_hash);
        }
    }
    return hashes;
}

std::vector<uint64_t> RadixConvCache::get_branch_block_hashes(uint64_t node_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return {};
    auto nodes = get_branch_nodes_locked(it->second);
    std::vector<uint64_t> hashes;
    hashes.reserve(nodes.size());
    for (const auto& n : nodes) {
        if (n != root_) {
            hashes.push_back(n->block_hash);
        }
    }
    return hashes;
}

std::vector<std::shared_ptr<KvBlock>> RadixConvCache::get_branch_blocks(
    const std::shared_ptr<RadixKvNode>& node) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto nodes = get_branch_nodes_locked(node);
    std::vector<std::shared_ptr<KvBlock>> blocks;
    for (const auto& n : nodes) {
        if (n->kv_block) {
            blocks.push_back(n->kv_block);
        }
    }
    return blocks;
}

std::vector<std::shared_ptr<KvBlock>> RadixConvCache::get_branch_blocks(uint64_t node_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return {};
    auto nodes = get_branch_nodes_locked(it->second);
    std::vector<std::shared_ptr<KvBlock>> blocks;
    for (const auto& n : nodes) {
        if (n->kv_block) {
            blocks.push_back(n->kv_block);
        }
    }
    return blocks;
}

bool RadixConvCache::set_kv_block(uint64_t node_id, std::shared_ptr<KvBlock> block) {
    if (config_.host_ram_budget_bytes == 0) {
        return false;
    }
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return false;
    auto node = it->second;
    node->last_accessed_us.store(current_time_us(), std::memory_order_relaxed);
    total_ram_bytes_ -= node->ram_bytes;
    node->kv_block = std::move(block);
    node->ram_bytes = node->compute_ram_bytes();
    total_ram_bytes_ += node->ram_bytes;

    if (config_.auto_evict_on_insert) {
        evict_to_budget_locked(0);
    }
    return true;
}

std::shared_ptr<KvBlock> RadixConvCache::get_kv_block(uint64_t node_id) const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return nullptr;
    it->second->last_accessed_us.store(current_time_us(), std::memory_order_relaxed);
    return it->second->kv_block;
}

bool RadixConvCache::remove_node(uint64_t node_id, bool force) {
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    auto it = node_map_.find(node_id);
    if (it == node_map_.end()) return false;
    auto node = it->second;
    if (node == root_) return false;
    if (!node->children.empty()) return false;
    if (!force) {
        if (node->is_pinned.load(std::memory_order_relaxed)) return false;
        if (node->is_system_prompt.load(std::memory_order_relaxed)) return false;
    }

    auto parent = node->parent.lock();
    if (parent) {
        total_ram_bytes_ -= parent->ram_bytes;
        auto& p_children = parent->children;
        for (auto cit = p_children.begin(); cit != p_children.end(); ++cit) {
            if (*cit == node) {
                p_children.erase(cit);
                break;
            }
        }
        parent->ram_bytes = parent->compute_ram_bytes();
        total_ram_bytes_ += parent->ram_bytes;
    }

    if (total_ram_bytes_ >= node->ram_bytes) {
        total_ram_bytes_ -= node->ram_bytes;
    } else {
        total_ram_bytes_ = 0;
    }

    node_map_.erase(it);
    return true;
}

void RadixConvCache::clear() {
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    node_map_.clear();
    root_->children.clear();
    root_->tokens.clear();
    root_->kv_block.reset();
    root_->is_system_prompt.store(false, std::memory_order_relaxed);
    root_->is_pinned.store(false, std::memory_order_relaxed);
    root_->ram_bytes = root_->compute_ram_bytes();
    node_map_[0] = root_;
    total_ram_bytes_ = root_->ram_bytes;
    next_node_id_ = 1;
}

RadixConvCacheStats RadixConvCache::get_stats() const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    RadixConvCacheStats stats;
    stats.total_nodes = node_map_.size();
    stats.total_ram_bytes = total_ram_bytes_;
    stats.prefix_lookups = prefix_lookups_.load(std::memory_order_relaxed);
    stats.prefix_hits = prefix_hits_.load(std::memory_order_relaxed);
    stats.prefix_misses = prefix_misses_.load(std::memory_order_relaxed);
    stats.evictions = evictions_;
    stats.bytes_evicted = bytes_evicted_;

    uint64_t sys_count = 0;
    uint64_t pin_count = 0;
    for (const auto& [id, node] : node_map_) {
        if (node->is_system_prompt.load(std::memory_order_relaxed)) {
            sys_count++;
        }
        if (node->is_pinned.load(std::memory_order_relaxed)) {
            pin_count++;
        }
    }
    stats.system_prompt_nodes = sys_count;
    stats.pinned_nodes = pin_count;
    return stats;
}

void RadixConvCache::reset_stats() {
    prefix_lookups_.store(0, std::memory_order_relaxed);
    prefix_hits_.store(0, std::memory_order_relaxed);
    prefix_misses_.store(0, std::memory_order_relaxed);
    std::unique_lock<std::shared_mutex> lock(rw_mu_);
    evictions_ = 0;
    bytes_evicted_ = 0;
}

size_t RadixConvCache::total_ram_bytes() const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    return total_ram_bytes_;
}

size_t RadixConvCache::total_nodes() const {
    std::shared_lock<std::shared_mutex> lock(rw_mu_);
    return node_map_.size();
}

} // namespace strata::core
