# AGENTS.md

strata-fabric is an independent fork of Niko1221/Strata, stripped to the core: the inference engine plus
its OpenAI- and Anthropic-compatible API server, nothing else. It runs the Qwen3.8-Flash-Next
mixture-of-experts model (and its Coder, Swift 1.5, Unsloth and Gyro variants) on a normal PC: one NVIDIA
or AMD graphics card plus system RAM. The code keeps the name `strata` (the binary, the modules, the CMake
project) so upstream merges stay cheap; only the project identity is strata-fabric.

There are two parts:

- **The engine** (`src/`, `include/`, `CMakeLists.txt`, `cmake/`, `third_party/` for ggml): a C++/CUDA/HIP
  binary. A resident engine speaks a line protocol over stdin/stdout (`--serve` mode: `GEN <max_new> <ids>`
  in, `T <id>`/`DONE` out, batch `BGEN`/`BT`/`BDONE`/`BADM`/`BSTOP`, samples `BSTAT`/`PSTAT`). How it
  works, with every measured number: [docs/DETAILS.md](docs/DETAILS.md) and
  [docs/paper/Strata-Paper.pdf](docs/paper/Strata-Paper.pdf). AMD: [docs/AMD_HIP.md](docs/AMD_HIP.md).
  Multi-GPU: [docs/MULTI_GPU.md](docs/MULTI_GPU.md). Tiered KV: [docs/TIERED_KV.md](docs/TIERED_KV.md).
- **The server** (`serve/server.py`, Python stdlib only): spawns the engine, serves `/v1/chat/completions`,
  `/v1/messages`, `/v1/responses`, images, metrics and an MCP tool hub on 127.0.0.1. It never binds past
  loopback without an API key. There is no web page; any OpenAI/Anthropic client is the front end.

`prepare.py` replaces the old installer: `--build hip|cuda` compiles the engine, `--model NAME` downloads
a pinned model (revision + size + SHA-256, resumable, into `../Strata-data`), builds the tokenizer pack
and the MTP draft, and writes `strata-<name>.json`. `prepare.py --list` prints the catalog. The NVFP4
model sizes are not in this slim catalog.

## Working on the code

- The server's tests need no GPU (a mock engine speaks the engine protocol):
  `python3 -m unittest discover -s serve -p 'test_*.py'`. Run this after every change to `serve/`.
- The engine's own tests: `cmake -B build-test -DSTRATA_BUILD_TESTS=ON && cmake --build build-test`.
- Build every backend a file touches before asking for review of an engine change (HIP is the one verified
  on this machine; CUDA compiles on any machine with nvcc).
- Model tools live in `tools/` (tokenizer, packs, MTP, NVFP4); benchmarks run with `tools/strata_bench.py`.
  `bench/` (micro-benchmarks and results) is local only — it is in `.gitignore`, never commit it. Keep
  measured claims with what they were measured on; make no claim without a measurement.
- Keep the docs' style: plain words, short sentences. The docs kept are DETAILS.md, the paper, AMD_HIP,
  MULTI_GPU, NVFP4 and TIERED_KV; anything else upstream writes stays unmerged (see below).

## Upstream policy (one-way)

Upstream (Niko1221/Strata) is a read-only source of engine updates: `git fetch upstream && git merge
upstream/main`; the push URL is set to `DISABLED`. Never push there, and never squash or rebase across a
merge base. After a merge, the removed paths (apps, Terminal, packaging, sycl, metal, setup, docker, the
web UI, the Terminal/app docs, most of tools/, most of strata_control/) come back as modify/delete
conflicts: resolve each with `git rm`, so the path stays deleted. The engine and server files merge as usual.
