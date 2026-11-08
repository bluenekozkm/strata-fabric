> **strata-fabric** — an independent fork of [Strata](https://github.com/Niko1221/Strata), stripped to the core: the
> inference engine plus its OpenAI- and Anthropic-compatible API server. There are also improvements and customization 
> features tailored for specific environments.
>
> **This fork is work in progress.** It is being actively rebuilt, so a change can break something at any time, and
> the documentation beyond the kept engine docs ([docs/DETAILS.md](docs/DETAILS.md),
> [docs/AMD_HIP.md](docs/AMD_HIP.md), [docs/MULTI_GPU.md](docs/MULTI_GPU.md), [docs/NVFP4.md](docs/NVFP4.md),
> [docs/TIERED_KV.md](docs/TIERED_KV.md), the
> [paper](docs/paper/Strata-Paper.pdf)) is thin. Weigh both before you rely on it.

<h3 align="center">Run a state-of-the-art model on your own gaming PC</h3>

<p align="center">Support for NVIDIA and AMD · Windows, Linux · free and open source</p>

<p align="center"><a href="LICENSE"><img src="https://img.shields.io/badge/license-MIT-blue" alt="MIT license"></a></p>

<p align="center"><a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4"><img src="docs/media/pagoda-preview.webp" width="720" alt="A voxel pagoda garden that Strata's model wrote in one shot"></a><br>
<sub>A voxel pagoda garden, 1 shot prompt running on an RTX 5070 with Strata (IQ3_S, 128K context) ·
<a href="https://github.com/Niko1221/Strata/releases/download/v0.1.10/Pagoda.mp4">full video (49 s)</a></sub></p>

Strata runs **[Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next)**, a 125-billion-parameter model that
normally needs a server, on a normal PC. It chats, writes code, reads pictures and works with your apps and coding
agents. Your chats and files stay on your PC.

**New here?** Check [what you need](#what-you-need), then [install](#install).

- **Runs on a gaming PC.** 32 GB of RAM and graphics memory together is enough to start.
- **Works with your tools.** It speaks the OpenAI and Anthropic APIs, so Claude Code, Codex CLI, OpenCode and other agents connect to it.
- **Reads pictures.** They are on by default.
- **Free.** MIT license, no account, no cloud.
- **Private.** No telemetry. The server listens on `127.0.0.1` and never binds past loopback without an API key.

Engine 0.1.49 (the same engine upstream ships, built from this repository). Every model size with its exact
download size: `python3 prepare.py --list`.

## What you need

**You need at least 32 GB of RAM and graphics memory (VRAM) together.** On a unified-memory PC, that is the whole
memory. More memory gives a bigger model or a faster one.

- A 32 GB PC with a 12-16 GB card runs the Coder.
- 64 GB of RAM runs every size.

| Your computer | How to run it here |
| --- | --- |
| **NVIDIA** GeForce RTX 20, 30, 40, 50 series (12 GB or more) | Build the CUDA engine: `python3 prepare.py --build cuda` |
| **AMD** Radeon RX 6800 and newer, Radeon AI PRO R9700 (12 GB or more) | Build the HIP engine: `python3 prepare.py --build hip`. [AMD cards](docs/AMD_HIP.md) |

Also needed:

- about 80 GB of free disk (an SSD makes the first start much faster)
- Windows 10 / 11 or Linux
- a current graphics driver
- Python 3.10 or newer with the packages in [requirements.txt](requirements.txt) (the server), and the CUDA toolkit
  or the ROCm SDK for the engine build

Two or three cards of the same make can share a model ([multi-GPU](docs/MULTI_GPU.md)). Strata never splits a model across
NVIDIA and AMD cards. Speed estimates for other cards than the two measured ones:
[docs/DETAILS.md](docs/DETAILS.md#other-gpus-estimated).

## Install

Three steps. They all run the same engine and use the same models.

### 1. Build the engine

```sh
python3 prepare.py --build hip     # AMD: RX 6800 and newer, R9700 (needs the ROCm SDK)
python3 prepare.py --build cuda    # NVIDIA: RTX 20 and newer (needs the CUDA toolkit)
```

The build finds your GPU architecture by itself (`--arch` overrides it, for example `gfx1201`) and writes the binary
into `engine/`.

> **Only the AMD build has been tested in this fork so far** (HIP, on this repository's Linux test PC with an RDNA4
> card; the running engine, the server and a real model were exercised end to end on it). The CUDA steps below
> compile wherever `nvcc` exists, but no NVIDIA card has run them here, and no other GPU either. Everything past
> AMD you must verify yourself — and please report what you find in an issue.

The detailed build, spelled out (the steps upstream's [AMD backend doc](docs/AMD_HIP.md#build) gives):

Requirements: a working ROCm driver/runtime, HIP development headers and compiler, hipBLAS and (for tuned dense
prefill) hipBLASLt development files, CMake 3.24+, a C++20 host compiler, and Git. Then:

```sh
cmake -S . -B build-hip \
  -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_ENABLE_CUDA=OFF \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100
cmake --build build-hip --target strata -j2
```

`CMAKE_HIP_ARCHITECTURES` is `gfx1100`, `gfx1101`, `gfx1200`, `gfx1201`, or a list such as `"gfx1100;gfx1201"`
(one binary for both); `gfx1030` (RDNA2) builds too. At startup the engine compares each GPU it uses with the
architectures the binary was compiled for and stops with a clear message instead of failing later with "invalid
device function". If CMake cannot find the HIP compiler, add
`-DCMAKE_HIP_COMPILER=/path/to/rocm/llvm/bin/clang++` (on the Fedora-family test host:
`/usr/lib64/rocm/llvm/bin/clang++`). Use the compiler and libraries from the same ROCm installation, and do not
enable both GPU backends in one build.

For NVIDIA, the same two commands with `-DSTRATA_ENABLE_CUDA=ON -DSTRATA_ENABLE_HIP=OFF` and
`CMAKE_CUDA_ARCHITECTURES` instead (the default is 120, for RTX 50; RTX 20, 30, 40 and 50 build, that is sm_75 and
newer. Below that, only an experimental Pascal/Volta build exists: `-DSTRATA_EXPERIMENTAL_SM60=ON`, which needs a
CUDA 12.x toolkit and is not tested here).

CMake fetches the llama.cpp revision pinned by this repository. For an offline build, point `STRATA_GGML_DIR` at
a checkout of that exact revision; using an arbitrary newer checkout changes the dependency being tested.

### 2. Get a model

#### The easy way:

```sh
python3 prepare.py --list                # the catalog: every size and its download size
python3 prepare.py --model Gyro-S        # download + tokenizer pack + MTP draft + strata-gyro-gyro-s.json
```

The catalog name goes into `--model` exactly as `--list` prints it (`Q2_0`, `IQ2_XS`, `IQ1_M` for the Coder,
`IQ2_XS-swift` for Swift 1.5, `Gyro-S`, `Gyro-M`, the Unsloth sizes, `Q8_0`). Every download is pinned (revision +
size + SHA-256) and continues where it stopped if it is interrupted. A model is a big download (about 40-90 GB).
The files land in the data folder next to the checkout (`../Strata-data` by default, `--data-dir` moves it), so a
fresh checkout never downloads them twice.

#### By hand, from Hugging Face

`prepare.py` is a plain script and does nothing you cannot do yourself. This is the flow the old `setup.sh`
automated, command by command. The worked example below uses **Gyro-S**; swap in any catalog entry the same way.

**a. Download the files.** Every size is one or more GGUF files in a Hugging Face repo, plus — except for the
Coder's second shard, which reuses the original's — one shared vision encoder for pictures:

| Size (family) | Repo | Files |
| --- | --- | --- |
| Q2_0, IQ2_XS, IQ3_XXS, IQ3_S | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `Qwen3.8-Flash-Next-GSQ-RCO-<size>-0000i-of-00002.gguf` (2 shards) + `mmproj-Qwen3.8-Flash-Next-BF16.gguf` |
| IQ1_M (the Coder) | `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` | `…-IQ1_M-0000i-of-00002.gguf` (2 shards); shard 2 is the original's file + the same `mmproj` |
| IQ2_XS-swift, IQ3_S-swift | `ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF` | `Swift-Qwen3.8-Flash-Next-GSQ-RCO-<size>-0000i-of-00002.gguf` + `mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf` |
| Gyro-S, Gyro-M | `agentionai/Qwen3.8-Flash-Next-Gyro-GGUF` | one file per size (Gyro-S: `Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf`) + the original `mmproj` |
| UD-IQ4_XS … Q8_0 | `unsloth/Qwen3.8-Flash-Next-GGUF` | 3 to 6 shards + the original `mmproj` |

The exact file names, the pinned revision of each repo, and every file's size and SHA-256 are the `REVISIONS` and
`PINS` tables at the top of `prepare.py` — take them from there, they are the source of truth. Download each file
in your browser, or:

```sh
huggingface-cli download agentionai/Qwen3.8-Flash-Next-Gyro-GGUF \
  Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf --revision <revision from REVISIONS>
# plain HTTPS works too: https://huggingface.co/<repo>/resolve/<revision>/<file>
```

Put them where the engine looks (the data folder; `--data-dir` moves the whole tree):

```
Strata-data/
  models/
    mmproj-Qwen3.8-Flash-Next-BF16.gguf        # the vision encoder: models/ root, one shared copy
    gyro-Gyro-S/
      Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf     # the model; multi-shard sizes keep all shards here
  packs/                                       # step b writes this
  mtp/                                         # step c writes this
```

Check each file against its pinned SHA-256, then write the `.done` mark — this is the old setup's protocol, and
`prepare.py` honors it, so a file you fetched yourself is never fetched again:

```sh
cd ../Strata-data/models/gyro-Gyro-S
sha256sum Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf          # compare with the pin
echo "sha256 <pin>" > Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf.done
```

**b. Build the pack** (the tokenizer the engine and the server read; needs the pinned llama.cpp for its
`gguf-py`, the commit `LLAMA_PIN` names in `prepare.py`):

```sh
git clone https://github.com/ggml-org/llama.cpp.git third_party/llama.cpp
git -C third_party/llama.cpp checkout <LLAMA_PIN>
export STRATA_GGUF_PY=$PWD/third_party/llama.cpp/gguf-py
python3 tools/strata_tokenizer.py --gguf ../Strata-data/models/gyro-Gyro-S/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf \
  --out ../Strata-data/packs/gyro-gyro-s                  # writes packs/gyro-gyro-s/tokenizer/
```

Optional, for PCs whose RAM does not hold every expert: pack the experts into one file for the low-RAM mode
(`--low-ram` in the easy way does this):

```sh
python3 tools/iq_pack.py --gguf <shard 1> --out ../Strata-data/packs/gyro-gyro-s \
  --experts-bin --compat-bf16        # --compat-bf16 only for the Gyro and Unsloth families
```

**c. Build the MTP draft** (the speculative-decoding helper; only its ~5 GB of tensors are downloaded):

```sh
python3 tools/mtp_fetch.py fetch --out ../Strata-data/mtp
python3 tools/mtp_pack.py --src ../Strata-data/mtp --experts q2_0 --out ../Strata-data/mtp/mtp-q2_0.gguf
python3 tools/mtp_rt.py --gguf ../Strata-data/mtp/mtp-q2_0.gguf --out ../Strata-data/mtp/rt
```

(All three `tools/` commands take the `STRATA_GGUF_PY` environment variable from step b.)

#### The run config

The server starts the engine from a JSON run config — the `strata-<name>.json` file. This is a complete one for
Gyro-S, the same thing `prepare.py` writes (the paths follow the default layout under a checkout at
`/home/you/strata-fabric`):

```json
{
 "exe": "/home/you/strata-fabric/engine/strata",
 "args": [
  "--pack", "/home/you/Strata-data/packs/gyro-gyro-s",
  "--native", "/home/you/Strata-data/models/gyro-Gyro-S/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
  "--ple-gguf", "/home/you/Strata-data/models/gyro-Gyro-S/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
  "--expert-cache", "auto", "--prefill", "auto",
  "--spec", "4", "--spec-min-p", "0.5",
  "--mtp", "/home/you/Strata-data/mtp/rt",
  "--max-context", "131072", "--kv", "int8", "--mmap-experts",
  "--expert-profile", "/home/you/strata-fabric/data/expert-profile.bin",
  "--vision"
 ],
 "cwd": "/home/you/strata-fabric",
 "tokenizer": "/home/you/Strata-data/packs/gyro-gyro-s/tokenizer",
 "model_name": "qwen3.8-flash-next-gyro-gyro-s",
 "port": 8080,
 "vision": {
  "exe": "/home/you/strata-fabric/engine/strata-vision",
  "mmproj": "/home/you/Strata-data/models/mmproj-Qwen3.8-Flash-Next-BF16.gguf",
  "model": "/home/you/Strata-data/models/gyro-Gyro-S/Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf",
  "gpu": false, "max_tokens": 768, "threads": 6
 },
 "backend": "hip",
 "log": "/home/you/Strata-data/strata-gyro-gyro-s.log"
}
```

What the pieces are:

- `exe` is the engine binary from step 1; `cwd` is the checkout. `--pack` and `--native` point at step b's pack
  and the model's first shard. `--ple-gguf` exists only for the Gyro sizes (the n-gram table rides inside the
  Gyro GGUF) — leave it out for every other family.
- `--spec 4 --spec-min-p 0.5 --mtp` are the guess-and-check settings from [How does it work?](#how-does-it-work),
  against step c's draft. `--max-context` is what your RAM allows (131072 on a 64 GB PC). `--kv int8
  --mmap-experts` are the defaults; keep them (`--kv q4s` halves KV memory with searched 4-bit scale precision).
- `--expert-profile` is a shipped ranking of which experts a 12-16 GB card reads most (`data/expert-profile.bin`;
  the Coder uses `data/expert-profile-coder.bin`). The other families set none.
- `model_name` is the name clients see in `/v1/models`. `vision` turns pictures on (`gpu: false` is the CPU
  reader for cards under 24 GB); drop the block to leave pictures off. `backend` (and `lib_dirs`, which
  `prepare.py` adds when it finds the ROCm SDK in a venv) is written for HIP builds; a CUDA build has neither.
- Name the file `strata-<name>.json`, then:

```sh
python3 serve/server.py --engine strata --config strata-gyro-gyro-s.json --port 8080
```

### 3. Start serving

```sh
python3 serve/server.py --engine strata --config strata-gyro-gyro-s.json --port 8080
```

The server spawns the engine and serves the OpenAI and Anthropic APIs, images, metrics and an MCP tool hub on one
port. Point any client at `http://127.0.0.1:8080/v1` (the next section has the exact settings per app).

> While the model starts, your PC can be slow or stop responding for 1-3 minutes (longest the first time). Strata loads
> 35-55 GB into your RAM and locks part of it for the graphics card. Wait, and don't close the window.

The server never listens past loopback unless you give it an API key: `--api-key KEY` (or the `STRATA_API_KEY`
environment variable) plus `--host 0.0.0.0` reaches the network. Every flag: `python3 serve/server.py --help`.

#### Multi-prompt switching: Tiered KV Cache (optional)

By default, the engine uses a standard linear KV cache. If you switch between different agents, tools, or multiple system prompts, enable the experimental tiered KV cache:

```sh
python3 serve/server.py --config strata-gyro-gyro-s.json --port 8080 --exp-tiered-kv
```

(Or put `"--exp-tiered-kv"` in `"args"` inside `strata-<name>.json`.)

- **What it does**: Holds divergent system prompts and conversation branches in a multi-branch radix tree in host RAM (Tier 1). Switching between system prompts takes under 5 ms without recomputing prompt tokens or consuming GPU VRAM ([docs/TIERED_KV.md](docs/TIERED_KV.md)).
- **Recommend Tier 1 (RAM only)**: Run `--exp-tiered-kv` without a path. This keeps all cache branches in host RAM, offering the fastest performance with zero SSD write wear. Cached sessions in RAM are volatile and reset when the server stops.
- **Caution with Tier 2 (SSD)**: If given a directory path (`--exp-tiered-kv /path/to/cache`), the engine persists KV blocks to SSD via Direct I/O. Continuous SSD caching can accelerate drive wear (TBW degradation). Use Tier 2 only when cross-restart cache persistence is strictly necessary.

## Models

One model in several sizes, compressed more or less. Smaller sizes are faster. Larger sizes are a bit smarter.

### Low on memory? Try Gyro

[Gyro](https://huggingface.co/agentionai/Qwen3.8-Flash-Next-Gyro-GGUF) (by agentionai) is the smallest way to run this model.

| Size | Memory it needs (RAM + VRAM) | Download | What it is |
| --- | ---: | ---: | --- |
| **Gyro-S** | 27.6 GB | 58.5 GB | For PCs that fit nothing else. Quality is lower than IQ2_XS. |
| **Gyro-M** | 35 GB | 92.0 GB | About IQ3_XXS quality with about 20% less GPU memory. It writes answers 17-21% slower than Q2_0 on a 12 GB card and on an R9700. |

Gyro runs on NVIDIA and AMD cards (AMD is experimental). Start it with `python3 prepare.py --model Gyro-S`
(or `Gyro-M`).

### The other sizes

| Your memory (RAM + VRAM) | Take | Notes |
| --- | --- | --- |
| **32 GB** | **Gyro-S**, or the **[Coder](docs/DETAILS.md#or-the-coder-half-the-experts-for-code)** for code | The Coder needs 32 GB of RAM and keeps half of the experts, so it is weaker outside code |
| **40 GB** | **[Swift 1.5](docs/DETAILS.md#or-swift-15-a-fine-tune-that-thinks-shorter)** IQ2_XS (39.2 GB), or **Gyro-M** (35 GB) | Swift thinks for a shorter time before it answers. Gyro-M is close to IQ3_XXS in quality, but it writes answers 17-21% slower than Q2_0 |
| **48 GB** | **IQ3_XXS** (47.0 GB) | Slower, better |
| **64 GB** | **IQ3_S** (54.8 GB), the original or Swift 1.5 | The slowest, the best: it matches the full model on the published tests |
| **96 GB or more** | Unsloth's **UD-IQ4_XS** (about 72 GB of RAM) to **Q8_0** (about 136 GB), 4 to 8 bit | Closer to the full model. With less RAM, Strata reads part of them from the SSD, which is much slower |

Less RAM than a row says can still work: Strata switches to its low-RAM mode by itself, and
`python3 prepare.py --model NAME --low-ram` writes the experts into one file for PCs whose RAM does not hold them.
Expect it to be slower, and much slower with a small graphics card.

More choices:

- **[Swift 1.5](docs/DETAILS.md#or-swift-15-a-fine-tune-that-thinks-shorter)** (`--model IQ2_XS-swift` or
  `IQ3_S-swift`) is a fine-tune that thinks for a shorter time before it answers, at about the same quality and RAM.
- **[NVFP4](docs/NVFP4.md)** is NVIDIA's 4-bit version for NVIDIA cards. It needs about 80 GB of RAM to keep every
  expert in RAM. The ready-made NVFP4 packs are not in this fork's slim catalog; the conversion workflow is in the doc.
- **Unsloth's UD quants** (`UD-IQ4_XS`, `UD-Q4_K_XL`, `UD-Q5_K_XL`, `UD-Q6_K_XL`) are regular catalog entries.

Every size with its exact download size: `python3 prepare.py --list`. What each size costs in RAM and how it
behaves: [docs/DETAILS.md](docs/DETAILS.md#which-model).

## How fast is it?

Based on a 12 GB graphics card (RTX 5070, Ryzen 5 7600, 64 GB RAM), measured by the original author with the
upstream release. See results from other cards: [speed of each model](docs/DETAILS.md#speed-measured).

| Model | Answers appear at | Reads your prompt at |
| --- | --- | --- |
| **Q2_0** | `████████████` 94 tokens/s | `████████████` 2,650 tokens/s |
| **IQ2_XS** | `██████████░░` 79 tokens/s | `█████████░░░` 2,090 tokens/s |
| **IQ3_XXS** | `████████░░░░` 62 tokens/s | `████████░░░░` 1,750 tokens/s |
| **IQ3_S** | `███████░░░░░` 53 tokens/s | `███████░░░░░` 1,620 tokens/s |
| **Coder** | `███████░░░░░` 55 tokens/s | `██████████░░` 2,180 tokens/s |

A token is about ¾ of a word, so 60 tokens per second is faster than you can read. "Reads your prompt" is how fast Strata
takes in a long document, code or chat history.

Measured on a 16 GB AMD card (RX 9070 XT, Ryzen 9 3900X, 47 GB RAM):

| Model | Answers appear at | Reads your prompt at |
| --- | --- | --- |
| **Q2_0** | `████████░░░░` 60 tokens/s | `█████░░░░░░░` 1,160 tokens/s |
| **IQ2_XS** | `███████░░░░░` 52 tokens/s | `█████░░░░░░░` 1,110 tokens/s |
| **Coder** | `██████░░░░░░` 44 tokens/s | `██████░░░░░░` 1,420 tokens/s |

The bars use one scale for both tables. A card with more VRAM is faster: an RTX 3090 (24 GB) should write about 100-140
tokens per second. The engine version behind each number is in [DETAILS.md](docs/DETAILS.md#speed-measured).

## Using it

### Connect your apps

| App type | Base URL |
| --- | --- |
| OpenAI-compatible (most apps and agents) | `http://127.0.0.1:8080/v1` (any API key, any model name) |
| Anthropic API (Claude Code and others) | `http://127.0.0.1:8080`, without `/v1`. These apps add `/v1/messages` themselves. Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:8080` |
| OpenAI Responses API (Codex CLI) | `http://127.0.0.1:8080/v1/responses` ([setup](docs/DETAILS.md#the-responses-api-and-codex-cli)) |

On loopback the server accepts any API key and any model name. With `--api-key` set, clients must send it
(`Authorization: Bearer …` for OpenAI-style apps, `x-api-key` for Anthropic-style apps).

### Tools

The model can call tools from your MCP servers through the hub at `/mcp` on the same port: add a server, and its
tools show up in chat completions. Setup and examples: [docs/DETAILS.md](docs/DETAILS.md#tools-from-mcp-servers).

### Pictures

Pictures are on by default for models that can read them. Attach or paste a picture in your client, or send it in the
`messages` of a request. The picture reader runs on the processor when your graphics card has less than 24 GB (a few
seconds per picture, nothing taken from the card's memory) and on the graphics card from 24 GB. The rule is the same
for NVIDIA and AMD. Turn pictures off with `python3 prepare.py --model NAME --no-vision` (it writes the run config
without the vision reader).

The numbers behind the rule: on a Radeon AI PRO R9700 the card's reader took 57 ms for a 260-token picture and the
processor's took 3.1 s. On 16 GB cards (Tesla P100 and RTX A4000, Linux) the card's reader cost 3-5% of text speed,
because it holds about 1.4 GB that the model would use otherwise. Details and measurements:
[docs/DETAILS.md](docs/DETAILS.md#images-vision).

### Good to know

- Choose the thinking level (**off, low, medium or high**) with your app's "reasoning effort". Off is fastest.
- The server answers one request at a time by default. For several at once, start it with `--parallel N`. On a 12 GB
  card this makes each answer slower.
- Strata reads the first message of a chat in full, about 1 minute per 30,000 tokens. Follow-up messages start in seconds.
- Where things are stored: the models and the packs in `../Strata-data`, the engine in `engine/`, one run config
  (`strata-<name>.json`) and one log (`strata-<name>.log`) per model in the checkout. The whole API surface:
  [docs/DETAILS.md](docs/DETAILS.md#using-it).

## Build from source

<details>
<summary>From the repository</summary>

Download the repository (`git clone https://github.com/bluenekozkm/strata-fabric`) or unzip it. Then:

```sh
python3 -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python3 prepare.py --build hip --model Gyro-S    # or --build cuda, and any name from --list
python3 serve/server.py --engine strata --config strata-gyro-gyro-s.json --port 8080
```

Useful flags of `prepare.py`: `--arch gfx1201` (the GPU architecture, auto-detected by default), `--rocm /opt/rocm`
(the ROCm root for the HIP build), `--context 131072` (the max context written into the run config), `--low-ram`,
`--no-vision`, `--port 8080`, `--force` (overwrite an existing run config). On Windows, the HIP SDK setup and the
per-card notes: [docs/AMD_HIP.md](docs/AMD_HIP.md).

The first `--model` download is the long step (about 40-90 GB). It continues where it stopped if it is interrupted,
and a file already checked keeps its mark, so run it again rather than delete anything. Next time, starting the
server downloads nothing.

Next time, run the server command again. It starts right away. Stop it with Ctrl-C (it stops the engine with it).

To measure your PC with the model running, use the benchmark: `.venv/bin/python tools/strata_bench.py` (or
`./bench.sh`). It takes 5-10 minutes and writes a report you can share.

**Optional: more speed on one card when the whole model fits in RAM.** Two opt-in settings by RubenTorney (#1904, #1905).
They help when you run one graphics card and your RAM holds every expert, for example Q2_0 or IQ3_XXS on a 64 GB PC. Open
`strata-<model>.json` in the Strata folder and add:

- `"--adapt-async", "1"` to the `"args"` list. The expert cache then updates on a helper thread instead of pausing the
  answer. Measured on an RTX 5070 (Windows, Q2_0): +4-5% decode, +7-10% on code and long documents. Answers can differ
  slightly from the default and stay the same from run to run.
- `"STRATA_PIN_BY_NEED": "1"` in the `"env"` block (Windows). Windows can lock only about half the RAM for the graphics
  card, and this locks the experts the card needs first. Measured on the RTX 5070: +4.4% prompt reading.

```json
"args": [ ..., "--adapt-async", "1" ],
"env": { "STRATA_PIN_BY_NEED": "1" }
```

Restart the server after the change. To undo, remove the two entries.

</details>

## How does it work?

Models like this one usually run on servers with hundreds of gigabytes of graphics memory. Your graphics card has
12-24 GB. Strata makes the model fit by sharing the work across your whole PC. Picture a library.

- The model is a team of 24,576 small specialists ("experts"), like 24,576 reference books. Each word needs only 10 of them.
- **Your graphics card is the desk.** It is small and fast. The roughly 4,000 books used most often lie open on it.
- **Your RAM is the shelves in the reading room.** All 24,576 books are there. When a word needs a book that is not on the
  desk, your processor reads it from the shelf at once, while the card keeps working.
- **Your SSD is the archive.** It holds a 29 GB lookup table. The model reads a few rows of it per word, so it does not
  need to sit in RAM.

A small helper also guesses the next few words. The big model checks all of them at once and keeps the right ones. You get
the same answer, 1.6-1.8x sooner. Long texts are read in big pieces, up to 8,192 tokens at a time, at over 1,000 tokens
per second.

The longer explanation: [docs/DETAILS.md](docs/DETAILS.md#how-it-works). Every part and its numbers:
[the details](docs/DETAILS.md#how-it-works) and the [paper](docs/paper/Strata-Paper.pdf).

## Something went wrong?

- **My PC froze the first time Strata started.** It loads the model into RAM. Wait, and don't close the window. Still
  frozen after 10 minutes? Restart the PC, close other programs and try again, or pick a smaller size.
- **It stopped while downloading.** Run `python3 prepare.py --model NAME` again. It continues where it stopped.
- **It is very slow, the disk light keeps blinking, or it says "the engine stopped unexpectedly".** Your PC does not have
  enough free RAM. Close other programs (browsers use a lot), or pick a smaller size such as Q2_0 or IQ2_XS.
- **The engine is missing or fails to start.** Build it again: `python3 prepare.py --build hip` (or `--build cuda`).
  The engine's log is `strata-<model>.log` next to the run config; the server prints the engine's first error.
- **It says port 8080 is already in use.** Strata is already running. Look for its window, or pick another port with
  `--port`.

More problems and fixes: [docs/DETAILS.md](docs/DETAILS.md#troubleshooting). Still stuck? Open an
[issue](https://github.com/bluenekozkm/strata-fabric/issues) with your run config and the engine log. For a security
problem, report it privately: [SECURITY.md](SECURITY.md).

## Credits and license

The model is [Qwen3.8-Flash-Next](https://huggingface.co/Qwen/Qwen3.8-Flash-Next) by the Qwen team. It was compressed by
[ISTA-DASLab](https://huggingface.co/ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF), UkisAI (Swift 1.5), agentionai (Gyro)
and Unsloth. Strata uses parts of [llama.cpp / ggml](https://github.com/ggml-org/llama.cpp). All credits:
[docs/DETAILS.md](docs/DETAILS.md#credits-and-licenses). Strata is open source under the [MIT License](LICENSE). A few parts and
every model have their own licenses ([which ones](docs/DETAILS.md#credits-and-licenses)).

## Contributing

This fork keeps only the engine and the API server. After a change to `serve/`, run the server's test suite (it needs
no GPU — a mock engine speaks the engine protocol): `python3 -m unittest discover -s serve -p 'test_*.py'`. Before
asking for review of an engine change, build every backend the file touches (HIP and CUDA).

## Support Strata

Support goes to the **original author of Strata**, not to this fork's maintainer: the engine in this repository is
upstream's work, and this section is kept from the original README with its destination unchanged.

Strata is free and MIT-licensed. The original author, [Niko1221](https://github.com/Niko1221), builds and tests Strata
on whatever hardware is available. If it runs well on your PC, a coffee helps keep that going — more cards tested,
more of what you report fixed. No pressure. Using it and reporting what you find helps just as much.

<a href="https://buymeacoffee.com/strataengine"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" height="50"></a>

Interested in sponsoring Strata? Contact the original author via [GitHub](https://github.com/Niko1221).
