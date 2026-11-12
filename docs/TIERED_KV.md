# Persistent Tiered KV Cache

The engine provides a three-tier, content-addressable KV cache hierarchy: active slots in GPU VRAM (Tier 0),
a multi-branch radix prefix tree in host RAM (Tier 1), and an asynchronous Direct I/O storage engine on NVMe SSD
(Tier 2).

It replaces single-chain checkpoint retention with a branching radix forest. Multiple system prompts and long
document prefixes remain live simultaneously with zero VRAM penalty, fast differential switching across chat
branches, and zero cold-start recovery across engine restarts.

```
+-------------------------------------------------------------------------+
| Tier 0: GPU VRAM (Active Slots)                                         |
| Fast attention kernels, QSA residency slots, dynamic paging             |
+------------------------------------+------------------------------------+
                                     ^  PCIe DMA Stream Link
                                     v  (Differential Switch < 5 ms)
+------------------------------------+------------------------------------+
| Tier 1: Host Pinned RAM (Multi-Branch Radix Prefix Tree)                |
| Zero VRAM penalty, multi-system prompt branching, LRU RAM eviction     |
+------------------------------------+------------------------------------+
                                     ^  Async Direct I/O (O_DIRECT)
                                     v  (Background Worker Queue)
+------------------------------------+------------------------------------+
| Tier 2: NVMe SSD (Persistent CAS Storage Engine)                        |
| Block files (.blk), manifest index, crash-consistent atomic fsync       |
+-------------------------------------------------------------------------+
```

## Why it is needed

Single-chain KV retention retains checkpoints along one branch of history only. When a client switches from
System Prompt A (e.g. Code Assistant) to System Prompt B (e.g. Creative Writer), the previous prefix is
evicted from VRAM and must be fully recalculated at prompt prefill speed (30K tokens costs ~30 seconds).

The Persistent Tiered KV Cache addresses this:
1. **Multi-Branch Radix Tree**: Multiple divergent system prompts coexist on host RAM simultaneously.
2. **Differential PCIe DMA Context Switching**: When switching contexts, common prefix blocks are retained;
   only divergent blocks are transferred over PCIe DMA with stream synchronization.
3. **Async NVMe Persistence with O_DIRECT**: Cold conversation branches spill to SSD asynchronously without
   polluting the OS Page Cache or system RAM.
4. **Zero Cold-Start**: Engine shutdown flushes the manifest index atomically. Restarting re-indexes existing
   blocks instantly without re-reading payloads or recomputing tokens.

---

## Architecture & Components

### 1. Block-based CAS & Serialization (`include/strata/core/kv_block.hpp`)

- **Fixed-size chunking**: Tokens are partitioned into blocks of 128 tokens (`kKvBlockTokens = 128`).
- **Content-Addressable Storage (CAS)**: Each block hash incorporates its token sequence, format, geometry,
  and parent block hash, forming a deterministic Merkle lineage:
  $$\text{block\_hash} = \text{xxHash64}(\text{parent\_hash} \parallel \text{tokens} \parallel \text{geometry})$$
- **Alignment**: Every serialized buffer is padded to a 4,096-byte multiple (`kKvBlockAlignment`) for Direct I/O.
- **Bit-exact roundtrip**: Preserves raw tensor data bit-for-bit across `kKvInt8`, `kKvHybrid` (K8V4), `kKvQ4`,
  and `kKvF16`.
- **Integrity**: Fast xxHash64 checksum protects against payload corruption and disk faults.

### 2. Async NVMe Storage Engine (`include/strata/platform/nvme_cache.hpp`)

- **Direct I/O (`O_DIRECT` / `FILE_FLAG_NO_BUFFERING`)**: Unbuffered SSD reads and writes bypass the OS file
  cache, keeping system memory free for expert weights and host KV pools.
- **Auto-fallback**: Automatically falls back to buffered I/O if the underlying filesystem does not support
  direct I/O (e.g. `tmpfs` in testing containers).
- **Worker threadpool**: Background worker queue processes reads and writes without blocking inference or token
  generation threads.
- **Manifest index (`.strata_kv_manifest.bin`)**: Binary metadata index recording block hashes, token counts,
  sizes, and access stamps.
- **Crash consistency**: Manifest writes use write-to-temp, `fsync`, and atomic `rename`.
- **LRU budget control**: Evicts oldest unpinned blocks when storage exceeds `--kv-disk-budget`. Pinned system
  prompts are strictly protected.

### 3. Host-Side Multi-Branch Radix Prefix Tree (`include/strata/core/radix_conv_cache.hpp`)

- **Radix tree structure (`RadixKvNode`)**: Nodes store token slices, CAS block hashes, reference counts, and
  pin status.
- **Longest Common Prefix (LCP)**: Traverses tree with shared read-locks (`std::shared_mutex`) for concurrent,
  zero-copy prefix matching.
- **Dynamic edge splitting**: Splits existing edges cleanly when new prompts diverge from an existing prefix,
  updating CAS hashes throughout the subtree.
- **0 VRAM penalty**: Branches are held in host RAM; GPU VRAM only maps the currently active execution context.
- **RAM budget enforcement**: Prunes least-recently-used unpinned leaf nodes when host memory budget is reached.
  The budget is the radix tree's own (`--tiered-kv-ram-mib`, default 2048 MiB), not shared with the parked
  conversation cache.
- **Pin scope**: A `pin=N` request key pins only the radix blocks covering the first N tokens of the
  conversation (128-token blocks; a block straddling the cut counts). The tail past the pin stays evictable.
  Pins are set, never cleared: a shared prefix node may carry another parked conversation's pin, so one
  conversation's park cannot take it away. A stale pin only holds the old prefix blocks, never the tail.

### 4. PCIe DMA Context Switcher & Stream Link (`include/strata/core/context_switcher.hpp`)

- **Differential switching**: Compares active session state with target context, preserving already-resident
  prefix blocks and moving only delta blocks.
- **Recurrence state restoration**: Preserves and restores recurrent states (GDN recurrence tensors, PLE history,
  and QSA indexer tails) bit-for-bit.
- **Stream ordering**: Coordinates DMA copies through CUDA/HIP streams (`cudaStream_t` / `hipStream_t`),
  eliminating race conditions with attention kernels.
- **Fast prefill bypass**: Calculates reusable leading tokens and bypasses prompt prefill computation.
- **Tiered auto-fetch**: If a requested block is absent from host RAM, it is loaded transparently from NVMe
  storage.

### 5. Server API & Engine Lifecycle (`src/program/generate.cpp`, `serve/server.py`)

- **Default Behavior**: Defaults to the standard **Linear Chain KVCache**. The Tiered KV Cache is an experimental feature enabled via the `--exp-tiered-kv` flag.
- **Engine CLI options**:
  - `--exp-tiered-kv [DIR]`: Enables experimental persistent tiered KV cache. If `DIR` is specified, blocks are persisted to SSD via Direct I/O; if omitted, runs in RAM-only mode (Tier 1 Radix tree in host RAM) and logs a warning that cached sessions are volatile and will be lost on engine restart. Tier 1 (RAM-only) is recommended for most workloads to prevent SSD wear.
  - `--kv-disk-cache DIR`: Persistent disk KV cache directory (implies `--exp-tiered-kv`).
  - `--kv-disk-budget N`: Maximum disk budget in MiB (default: 10240, 10 GiB).
  - `--tiered-kv-ram-mib N`: Host RAM budget for the radix tree in MiB (default: 2048, 2 GiB). Its own budget:
    the parked conversation cache (`--conversation-cache-mib`) and the radix tree each get their own, so the
    two together can hold up to the sum. Earlier builds sized the radix tree after `--conversation-cache-mib`
    (4096 with `--exp-tiered-kv`); pass `--tiered-kv-ram-mib 4096` to keep that size.
  - `--kv-multi-system-prompt`: Enables multi-branch radix tree cache in resident `--serve` mode. Part of the experimental tiered KV cache suite (activates `--exp-tiered-kv`).
- **Signal handling**: `SIGINT` and `SIGTERM` handlers flush dirty blocks and synchronize manifest index to disk.
- **Server integration**: `serve/server.py` recognizes system prompt switches, hashes system prompt prefixes,
  passes `kv_branch=<id>` and `kv_multi_sys=1` to the engine, and tracks hit telemetry under `/metrics`.

---

## Usage

### 1. Recommended Operating Mode & SSD Wear Caution

> [!TIP]
> **Use Tier 1 (Host RAM) for most workloads**: Running `--exp-tiered-kv` without a cache directory operates entirely in host memory. This delivers instant differential switching (< 5 ms) and zero prefill penalties across prompt branches without writing to disk.

> [!WARNING]
> **Tier 2 (NVMe SSD) writes consume SSD endurance**: Specifying an SSD directory (`--exp-tiered-kv <DIR>`) writes 128-token CAS blocks to disk via Direct I/O. Under heavy multi-turn conversational workloads or frequent session branching, this generates sustained disk write traffic that can degrade SSD lifespan (TBW). Use Tier 2 only when persistent caching across engine restarts is strictly required.

### 2. Running the Engine Directly

Start resident server with tiered disk cache enabled via `--exp-tiered-kv`:

```bash
# Recommended: Volatile RAM-only mode (Tier 1):
strata --pack <pack_dir> --native <model.gguf> --serve \
       --exp-tiered-kv

# Or Persistent SSD mode (Tier 2, writes to disk):
strata --pack <pack_dir> --native <model.gguf> --serve \
       --exp-tiered-kv /path/to/kv_cache \
       --kv-disk-budget 20480
```

Every park logs the radix tree's state so RAM growth is visible in the engine log:

```
strata serve: conversation cache: parked N tokens in T ms; parked=... bytes=... evictions=... snapshot_bytes=...
              reused_kv_bytes=... radix_bytes=... radix_nodes=... radix_evictions=... radix_pinned=...
```

`radix_bytes` stops at the `--tiered-kv-ram-mib` budget and `radix_evictions` starts climbing once it is full;
`radix_pinned` is the number of pinned nodes (the pinned prefixes).

### 3. Configuring via `serve/server.py`

In your server configuration file (e.g. `strata-config.json`):

```json
{
  "args": [
    "--pack", "/path/to/pack",
    "--native", "/path/to/model.gguf",
    "--exp-tiered-kv"
  ]
}
```

Or pass via server CLI directly:
```bash
# Recommended RAM-only mode:
python3 -m serve.server --config strata-config.json --exp-tiered-kv

# Or Persistent SSD mode:
python3 -m serve.server --config strata-config.json --exp-tiered-kv /var/cache/strata/kv --kv-disk-budget 20480
```

Requests to `/v1/chat/completions` or `/v1/messages` with varying system prompts automatically benefit from
multi-branch caching and fast prefill bypass.

### 4. Monitoring Telemetry

Query cache hit rate and branch metrics:

```bash
curl http://127.0.0.1:8080/metrics
```

Output includes the `tiered_kv` telemetry block:

```json
{
  "tiered_kv": {
    "system_prompt_branches": 4,
    "system_prompt_switches": 18,
    "system_prompt_hits": 14,
    "multi_system_requests": 32
  }
}
```

---

## Test Verification

Run all unit and integration tests:

```bash
# 1. C++ Engine and Tiered KV Cache unit tests
cmake -B build-test -DSTRATA_BUILD_TESTS=ON
cmake --build build-test
ctest --test-dir build-test -R "kv_block|nvme_cache|radix_conv_cache|context_switcher"

# 2. Python Server Integration tests
python3 -m unittest serve.test_server.Cluster5TieredKvCache

# 3. Comprehensive End-to-End Benchmark
./build-test/tiered_kv_experiment
```
