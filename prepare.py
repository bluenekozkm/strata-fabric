"""prepare.py - get a model and an engine ready for strata-fabric's server, without questions.

Replaces the old interactive setup.py with a plain, scriptable flow (this fork keeps only the engine and
its API server; the Terminal, the desktop app, the update feed and the installers are gone):

  python3 prepare.py --list                       the catalog
  python3 prepare.py --model Gyro-M               download + pack + MTP draft + write strata-gyro-gyro-m.json
  python3 prepare.py --build hip                  compile the engine (gfx arch auto-detected, or --arch)
  python3 prepare.py --model Gyro-M --build hip   both

Every download is pinned (revision + size + SHA-256) and resumable, and lands in the data folder NEXT TO
this one (../Strata-data by default, like the old setup) so a fresh checkout never re-downloads.  A file
already checked keeps its <name>.done mark, and marks written by the old setup.py are honored.
"""
from __future__ import annotations

import argparse
import hashlib
import http.client
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
LLAMA_PIN = "3cf03257f219afbe7334045ff7c6a06ac68c627d"        # the llama.cpp commit CMakeLists fetches
LLAMA_DIR = ROOT / "third_party" / "llama.cpp"
EXE = "strata.exe" if os.name == "nt" else "strata"

# ------------------------------------------------------------------------------------------------- the catalog
# Repositories at their pinned revisions.  The sizes and SHA-256 are the Hub's LFS pointers.
HF = "https://huggingface.co"
REVISIONS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "ed59f92082b1e93c0e96d60a8b11aab089b52f09",
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": "99bb8f7f95c7aa7b24a36a7786a4f657b30f5d3d",
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": "5348543e0147355ac9cbcb031184a3546350988e",
    "unsloth/Qwen3.8-Flash-Next-GGUF": "38bb39ee97821de2c9009abb7e93950eec396e66",
    "unsloth/Qwen3.6-35B-A3B-MTP-GGUF": "5bc3e238d916f48a861bac2f8a1990a0e9b7e98d",
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": "1ada08422e404aca51ab9df665f089f34c419c6b",
}
MMPROJ = ("mmproj-Qwen3.8-Flash-Next-BF16.gguf",
          907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0")
MMPROJ_SWIFT = ("mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf",
                907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60")

# name -> family, about, download GB, and the pinned files: HF path -> (bytes, sha256)
CATALOG = {
    "Q2_0": {"family": "qwen", "about": "2-bit, the fastest", "gb": 66.4},
    "IQ2_XS": {"family": "qwen", "about": "2-bit i-quant, a little better quality, close in speed", "gb": 68.0},
    "IQ3_XXS": {"family": "qwen", "about": "3-bit i-quant, better quality, slower", "gb": 75.8},
    "IQ3_S": {"family": "qwen", "about": "3.5-bit i-quant, the best quality, the slowest", "gb": 83.6},
    "IQ1_M": {"family": "coder", "about": "the Coder's only size: half the experts, kept for code and tools", "gb": 58.4},
    "IQ2_XS-swift": {"family": "swift", "title": "Swift 1.5 IQ2_XS", "about": "Swift 1.5, 2-bit i-quant", "gb": 68.2},
    "IQ3_S-swift": {"family": "swift", "title": "Swift 1.5 IQ3_S", "about": "Swift 1.5, 3.5-bit i-quant", "gb": 83.8},
    "Gyro-S": {"family": "gyro", "about": "the smallest option (27.6 GB of VRAM + RAM together)", "gb": 58.5},
    "Gyro-M": {"family": "gyro", "about": "about IQ3_XXS quality in 35 GB of VRAM + RAM together", "gb": 92.0},
    "UD-IQ4_XS": {"family": "unsloth", "about": "~4-bit i-quant (Unsloth Dynamic), three shards", "gb": 93.7,
                  "shards": 3, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00003.gguf"},
    "UD-Q4_K_XL": {"family": "unsloth", "about": "4-bit K-quant experts (Unsloth Dynamic)", "gb": 111.3},
    "UD-Q5_K_XL": {"family": "unsloth", "about": "5-bit K-quant experts, close to Q8_0 in quality", "gb": 158.3,
                   "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf"},
    "UD-Q6_K_XL": {"family": "unsloth", "about": "6-bit K-quant experts", "gb": 169.2,
                   "shards": 6, "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf"},
    "Q8_0": {"family": "unsloth-q8", "about": "8-bit, the reference quality (192 GB PC)", "gb": 188.2},
}
# per family: the repo, the shard file-name pattern, the vision encoder, iq_pack's extra args
FAMILY = {
    "qwen": {"repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
             "shards": 2, "mmproj": MMPROJ, "tag": "", "name": "qwen3.8-flash-next"},
    "coder": {"repo": "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF", "file": "Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
              "shards": 2, "mmproj": MMPROJ, "tag": "coder-", "name": "qwen3.8-flash-next-coder",
              "profile": "expert-profile-coder.bin"},
    "swift": {"repo": "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF", "file": "Swift-Qwen3.8-Flash-Next-GSQ-RCO-{q}-0000{i}-of-00002.gguf",
              "shards": 2, "mmproj": MMPROJ_SWIFT, "tag": "swift-", "name": "swift-1.5",
              "license": "Swift Open License 1.0"},
    "gyro": {"repo": "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF", "file": "Qwen3.8-Flash-Next-{q}.gguf",
             "shards": 1, "mmproj": MMPROJ, "tag": "gyro-", "name": "qwen3.8-flash-next-gyro",
             "pack_args": ["--compat-bf16"], "profile": "expert-profile.bin"},
    "unsloth": {"repo": "unsloth/Qwen3.8-Flash-Next-GGUF", "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00004.gguf",
                "shards": 4, "mmproj": MMPROJ, "tag": "unsloth-", "name": "qwen3.8-flash-next-unsloth",
                "pack_args": ["--compat-bf16"]},
    "unsloth-q8": {"repo": "unsloth/Qwen3.8-Flash-Next-GGUF", "file": "Qwen3.8-Flash-Next-{q}-0000{i}-of-00006.gguf",
                   "shards": 6, "mmproj": MMPROJ, "tag": "unsloth-", "name": "qwen3.8-flash-next-unsloth",
                   "pack_args": ["--compat-bf16"]},
}

# The pinned files: family repo -> file name -> (bytes, sha256).  The sizes and SHA-256 are the Hub's LFS
# pointers of the pinned revisions (the old setup.py's tables, carried over).
PINS = {
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-GGUF": {
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf": (39225954592, "92cee27ae5bbadcd732416a0f7a7f0acc092399dbbe8f5a5efa707c2ec0a49d7"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf": (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf": (54817524224, "4c1eb2ceb4915e1192f4f386021897bde56a97f40a0bb78bb86465e0f7d2aca3"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf": (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00001-of-00002.gguf": (47039860096, "219ea929900dfa9ef091f3aa473fdba6874b65fcb36526d7d851ac9e95856d15"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ3_XXS-00002-of-00002.gguf": (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
        "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00001-of-00002.gguf": (37623740192, "69820c02ec7d0b45ef2ebb19d6620299db749fe2aded7f39f93c6b88b199b720"),
        "Qwen3.8-Flash-Next-GSQ-RCO-Q2_0-00002-of-00002.gguf": (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
        "mmproj-Qwen3.8-Flash-Next-BF16.gguf": (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
    },
    "ukisai/Swift-1.5-Qwen3.8-Flash-Next-GSQ-RCO-GGUF": {
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00001-of-00002.gguf": (39788473344, "ca3b302d3d1eb6eddce3ea4acc79854e6cc5fdc0583e374a6a8b6bb67327743b"),
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ2_XS-00002-of-00002.gguf": (28363693824, "e4996a6de18fd49c58d9a0040fcfd73787db6407c51973f29884bee7b14c9dbb"),
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00001-of-00002.gguf": (44900816736, "46cb3996cd1a17fe8bc5ae1f9bb16869480de5a03f17f868d6047a20a46edd3b"),
        "Swift-Qwen3.8-Flash-Next-GSQ-RCO-IQ3_S-00002-of-00002.gguf": (38842920384, "70ce6fb48a94fe34bd7e9a53a526be68066c64b5a5077cdc82fb31350b66b608"),
        "mmproj-Swift-Qwen3.8-Flash-Next-BF16.gguf": (907543520, "cd1140f4abba943ce8d9fa92fed8407e389954d26907efde7b72153c75bd8a60"),
    },
    "ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF": {
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf": (29608446496, "e11083ba855e7666b48ea3f2db6a9c3a20c18751a012cc24f948de91b7087fad"),
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf": (28800138432, "316b46f3a2dbd68c900f43136ab9449f9dcc3725dfd8c794847c204bc161e113"),
    },
    "agentionai/Qwen3.8-Flash-Next-Gyro-GGUF": {
        "Qwen3.8-Flash-Next-Gyro-S-TQ1_0.gguf": (58487015808, "c6ea1b08ca312f99095b27588ba35976b53f948268e7f91e7761926aa01d9c87"),
        "Qwen3.8-Flash-Next-Gyro-M-TQ2_0.gguf": (91965774272, "7aa36d15131d255179ab6c87fa9c9f81a0b541fc9e81e959d4c3e29eb20feb5f"),
        "mmproj-Qwen3.8-Flash-Next-BF16.gguf": (907543008, "b1a82259702816a5330d7bd7607cd9676b11780e79ff7348c21103ff3ce49bd0"),
    },
    "unsloth/Qwen3.8-Flash-Next-GGUF": {
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00001-of-00004.gguf": (10946624, "4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082"),
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00002-of-00004.gguf": (49859583136, "3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9"),
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00003-of-00004.gguf": (49376141504, "56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3"),
        "Qwen3.8-Flash-Next-UD-Q4_K_XL-00004-of-00004.gguf": (12087983520, "753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a"),
        "Qwen3.8-Flash-Next-UD-IQ4_XS-00001-of-00003.gguf": (10946624, "5ce89370720f8bf90890f439361282104c1aa1482d4013bb9a50923e758e71a4"),
        "Qwen3.8-Flash-Next-UD-IQ4_XS-00002-of-00003.gguf": (49835229856, "577a38a2392b40ca2193cea502e1d92f60b8cd370675d308e0ec21885d9daaa7"),
        "Qwen3.8-Flash-Next-UD-IQ4_XS-00003-of-00003.gguf": (43836407744, "d4634e6d84f0ebb0940be15c90d3790bf6464e3dea3a1cddc567dc0e83ad8833"),
        "Qwen3.8-Flash-Next-UD-Q5_K_XL-00001-of-00006.gguf": (10946618, "610922cdb1afe1094e95d7a3e86b12ee33317ebbf7320ac9344e21b6ae4bb69f"),
        "Qwen3.8-Flash-Next-UD-Q5_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
        "Qwen3.8-Flash-Next-UD-Q5_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
        "Qwen3.8-Flash-Next-UD-Q5_K_XL-00004-of-00006.gguf": (49990245824, "9ee6bbe462e6830864382d629101490625b1c95afa15492b7ad550c7bed2dd13"),
        "Qwen3.8-Flash-Next-UD-Q5_K_XL-00005-of-00006.gguf": (49882416192, "0b2437f661f584030cb5aeb8fb1bb70f6422518a6746bfcc0304167c7b6b8beb"),
        "Qwen3.8-Flash-Next-UD-Q5_K_XL-00006-of-00006.gguf": (3320101792, "55dd96c5ad0863b06b6393151b9ad0ac727b8d58a62e02a296ca5d5ddbfee4a3"),
        "Qwen3.8-Flash-Next-UD-Q6_K_XL-00001-of-00006.gguf": (10946624, "8bdc6bad55b6699f46f60708829932ff4018d7ac9bd309c388534fa31753bdb7"),
        "Qwen3.8-Flash-Next-UD-Q6_K_XL-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
        "Qwen3.8-Flash-Next-UD-Q6_K_XL-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
        "Qwen3.8-Flash-Next-UD-Q6_K_XL-00004-of-00006.gguf": (49814108512, "0ea4b599880a5a52fcd9c188ba1443f646c69c96571904ddbd6a79246a01a997"),
        "Qwen3.8-Flash-Next-UD-Q6_K_XL-00005-of-00006.gguf": (49419317344, "9948e81ae8368144b135a7403d8c0f2cf9a6c2f9f692313865855b2832e696c9"),
        "Qwen3.8-Flash-Next-UD-Q6_K_XL-00006-of-00006.gguf": (14838313984, "e4ae255234f42b18012f6d6eec5bf615b57479b7e0a0233c47436a321ffeec3f"),
        "Qwen3.8-Flash-Next-Q8_0-00001-of-00006.gguf": (10946624, "2dabcbb53ca537a7947bc7d20414fd464eeaf4d66d43021b5b2556cc87544ad2"),
        "Qwen3.8-Flash-Next-Q8_0-00002-of-00006.gguf": (682434912, "494ca4ed3dbf97bc28da88af3890b8877b9032f909812d00c0526a9ca5e91d2e"),
        "Qwen3.8-Flash-Next-Q8_0-00003-of-00006.gguf": (54400261312, "34efd79a80a1ce540a517a5d56171924b66ce1c38b04c904f17ad6d8ef17cf20"),
        "Qwen3.8-Flash-Next-Q8_0-00004-of-00006.gguf": (49446841216, "bfa634025fabbd2658bf7694bc80b90e571699c768723f844c934c7ef06c691a"),
        "Qwen3.8-Flash-Next-Q8_0-00005-of-00006.gguf": (49668930400, "232a8f14cc0fa4262e7efe8593774b136fe40909e39c7a020342ddaa27259a97"),
        "Qwen3.8-Flash-Next-Q8_0-00006-of-00006.gguf": (34015618784, "538a93bca918064983409a41187ad4c68640f9aced6f29564da8f551bf86d7a5"),
    },
}

# --------------------------------------------------------------------------------------------------- the shell
def say(msg: str) -> None:
    print(msg, flush=True)


def fail(msg: str, fix: str = "") -> None:
    say(f"[prepare] {msg}" + (f"\n  {fix}" if fix else ""))
    sys.exit(1)


def ok(msg: str) -> None:
    print(f"[prepare] {msg}", flush=True)


def run(cmd: list[str], env: dict | None = None) -> None:
    r = subprocess.run(cmd, env=env)
    if r.returncode != 0:
        fail(" ".join(str(x) for x in cmd[:4]) + " ... failed", "see its output above")


def data_folder(asked: str | None) -> Path:
    """The data folder NEXT TO this checkout (../Strata-data), so a fresh clone keeps its downloads."""
    if asked:
        p = Path(asked).expanduser().resolve()
    else:
        p = ROOT.parent / "Strata-data"
    p.mkdir(parents=True, exist_ok=True)
    return p


def gguf_py_env() -> dict:
    if not LLAMA_DIR.is_dir():
        say("  fetching the pinned llama.cpp (gguf-py for the pack tools, ggml for the build) ...")
        LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_DIR)])
        run(["git", "-C", str(LLAMA_DIR), "checkout", LLAMA_PIN])
    return dict(os.environ, STRATA_GGUF_PY=str(LLAMA_DIR / "gguf-py"))


# -------------------------------------------------------------------------------------------------- downloading
def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def marked(p: Path, sha: str) -> bool:
    """The .done mark protocol (also the old setup.py's): the mark holds 'sha256 <hex>' after a checked download."""
    try:
        return p.is_file() and f"sha256 {sha}" in p.with_name(p.name + ".done").read_text(errors="replace")
    except OSError:
        return False


def mark(p: Path, sha: str) -> None:
    p.with_name(p.name + ".done").write_text(f"sha256 {sha}\n", encoding="utf-8")


def download(url: str, dst: Path, total: int, sha: str, what: str) -> None:
    """One resumable stream with a Range resume and a progress line; a wrong file is deleted, not kept."""
    if marked(dst, sha):
        ok(f"{what}: already downloaded and checked")
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    part = dst.with_suffix(dst.suffix + ".part")
    have = part.stat().st_size if part.exists() else 0
    for attempt in range(30):
        if total and have >= total:
            break
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "strata-fabric-prepare",
                                                       "Range": f"bytes={have}-"})
            with urllib.request.urlopen(req, timeout=60) as r, open(part, "ab" if have else "wb") as f:
                if have and r.status != 206:                 # the server ignored the range: start over
                    f.seek(0)
                    f.truncate()
                    have = 0
                last = 0.0
                while True:
                    b = r.read(8 << 20)
                    if not b:
                        break
                    f.write(b)
                    have += len(b)
                    if time.time() - last > 2:
                        last = time.time()
                        size = f"{have / 1e9:6.2f} / {total / 1e9:.2f} GB ({100 * have / total:.0f}%)" if total \
                            else f"{have / 1e6:7.1f} MB"
                        print(f"\r  {what}: {size}   ", end="", flush=True)
            print()
            if not total or have >= total:
                break
        except (OSError, http.client.HTTPException) as e:
            print()
            say(f"  download interrupted ({e}); retrying in 10 s ...")
            time.sleep(10)
    if total and part.stat().st_size != total:
        fail(f"{what}: the download is {part.stat().st_size} bytes, the pin says {total}",
             "delete the .part file and run this again")
    got = sha256_of(part)
    if got != sha:
        part.unlink(missing_ok=True)
        fail(f"{what}: SHA-256 mismatch (got {got[:16]}..., the pin says {sha[:16]}...)",
             "the file was deleted; a mirror that ignores range requests is the usual cause — try again")
    part.rename(dst)
    mark(dst, sha)
    ok(f"{what}: downloaded and checked")


# --------------------------------------------------------------------------------------------------- the model
def model_files(name: str) -> tuple[dict, dict, dict, str]:
    """(family dict, the model's files {name: (bytes, sha)}, the shard paths' HF urls, the config tag)."""
    m = CATALOG[name]
    fam = FAMILY[m["family"]]
    if m["family"] == "gyro":
        q = "Gyro-S-TQ1_0" if name == "Gyro-S" else "Gyro-M-TQ2_0"
    else:
        q = name.replace("-swift", "")               # the quant tier is the name (Q2_0, UD-Q4_K_XL, ...)
    files, urls = {}, {}
    pattern = m.get("file", fam["file"])
    for i in range(1, m.get("shards", fam["shards"]) + 1):
        fname = pattern.format(q=q, i=i)
        pin = PINS[fam["repo"]].get(fname)
        if not pin:
            fail(f"{name}: no pin for {fam['repo']}/{fname}", "the catalog in prepare.py needs this file's LFS pointer")
        files[fname] = pin
        urls[fname] = f"{HF}/{fam['repo']}/resolve/{REVISIONS[fam['repo']]}/{fname}"
    mp = fam["mmproj"]
    files[mp[0]] = (mp[1], mp[2])
    urls[mp[0]] = f"{HF}/{fam['repo']}/resolve/{REVISIONS[fam['repo']]}/{mp[0]}"
    tag = fam["tag"] + name
    return fam, files, urls, tag


def prepare_model(a) -> None:
    name = a.model
    if name not in CATALOG:
        fail(f"unknown model {name!r}", "one of: " + ", ".join(sorted(CATALOG)))
    fam, files, urls, tag = model_files(name)
    data = data_folder(a.data_dir)
    models = data / "models" / tag
    shards = []
    for fname, (size, sha) in files.items():
        # the vision encoder lands in models/ root: one copy shared by every model, like the old layout
        dst = (data / "models" / fname) if fname.startswith("mmproj") else (models / fname)
        download(urls[fname], dst, size, sha, fname)
        if not fname.startswith("mmproj"):
            shards.append(dst)
    ok(f"model files: {models}")

    pack = data / "packs" / tag.lower()
    env = gguf_py_env()
    if a.low_ram and not (pack / "experts.bin").exists():
        say(f"  writing the experts into one file for the low-RAM mode (one time, ~{CATALOG[name]['gb']:.0f} GB written) ...")
        run([sys.executable, str(ROOT / "tools" / "iq_pack.py"), "--gguf", str(shards[0]), "--out", str(pack),
             "--experts-bin", *fam.get("pack_args", [])], env=env)
    if not (pack / "tokenizer" / "vocab.json").exists():
        run([sys.executable, str(ROOT / "tools" / "strata_tokenizer.py"), "--gguf", str(shards[0]), "--out", str(pack)],
            env=env)                      # writes <pack>/tokenizer/
    ok(f"pack: {pack}")

    mtp = data / "mtp"
    if not (mtp / "rt" / "experts.bin").exists():
        say("  the MTP draft layer (speculative decoding): only its ~5 GB of tensors are downloaded ...")
        run([sys.executable, str(ROOT / "tools" / "mtp_fetch.py"), "fetch", "--out", str(mtp)], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_pack.py"), "--src", str(mtp), "--experts", "q2_0",
             "--out", str(mtp / "mtp-q2_0.gguf")], env=env)
        run([sys.executable, str(ROOT / "tools" / "mtp_rt.py"), "--gguf", str(mtp / "mtp-q2_0.gguf"),
             "--out", str(mtp / "rt")], env=env)
    ok(f"MTP draft: {mtp / 'rt'}")

    write_config(a, fam, tag, pack, shards, mtp, data)


def write_config(a, fam: dict, tag: str, pack: Path, shards: list[Path], mtp: Path, data: Path) -> None:
    eng = ROOT / "engine" / EXE
    if a.engine_path:
        eng = Path(a.engine_path).resolve()
        if not eng.is_file():
            fail(f"--engine-path {eng} is not a file")
    args = ["--pack", str(pack), "--native", str(shards[0]),
            "--expert-cache", "auto", "--prefill", "auto",
            "--spec", "4", "--spec-min-p", "0.5", "--mtp", str(mtp / "rt"),
            "--max-context", str(a.context), "--kv", "int8", "--mmap-experts"]
    if CATALOG[a.model]["family"] == "gyro":
        args[4:4] = ["--ple-gguf", str(shards[0])]       # the n-gram table rides inside the Gyro GGUF
    profile = fam.get("profile")
    if profile and (ROOT / "data" / profile).exists():
        args += ["--expert-profile", str(ROOT / "data" / profile)]
    cfg = {"exe": str(eng), "args": args, "cwd": str(ROOT), "tokenizer": str(pack / "tokenizer"),
           "model_name": f"{fam['name']}-{tag.lower()}", "port": a.port}
    if fam.get("license"):
        cfg["license"] = fam["license"]
    vision_exe = ROOT / "engine" / ("strata-vision.exe" if os.name == "nt" else "strata-vision")
    mmproj = data / "models" / fam["mmproj"][0]
    if not a.no_vision and vision_exe.exists() and mmproj.exists():
        cfg["vision"] = {"exe": str(vision_exe), "mmproj": str(mmproj), "model": str(shards[0]),
                         "gpu": False, "max_tokens": 768, "threads": 6}
        args.append("--vision")
    if a.backend == "hip":
        cfg["backend"] = "hip"
        dirs = hip_lib_dirs(hip_root(a.rocm))
        if dirs:
            cfg["lib_dirs"] = dirs
    cfg["log"] = str(data / f"strata-{tag.lower()}.log")
    out = ROOT / f"strata-{tag.lower()}.json"
    if out.exists() and not a.force:
        fail(f"{out.name} already exists", "it may hold your tuned arguments; pass --force to overwrite it")
    out.write_text(json.dumps(cfg, indent=1) + "\n", encoding="utf-8")
    ok(f"run config: {out}")
    say(f"""
ready. start the server with:

  {sys.executable} -m serve.server --engine strata --config {out.name} --port {a.port}

then use it as an OpenAI- or Anthropic-compatible API at http://127.0.0.1:{a.port}/v1 .""")


# ------------------------------------------------------------------------------------------------------- build
def hip_root(asked: str | None) -> Path:
    if asked:
        return Path(asked).resolve()
    env = os.environ.get("ROCM_PATH")
    if env and Path(env).is_dir():
        return Path(env)
    hits = sorted(ROOT.glob(".venv/lib*/python*/site-packages/_rocm_sdk_devel"))
    if hits:
        return hits[-1].resolve()
    for cand in sorted(Path("/").glob("opt/rocm*")):
        return cand.resolve()
    if Path("/usr/lib/x86_64-linux-gnu").is_dir() or Path("/usr/lib64").is_dir():
        return Path("/usr")
    fail("no ROCm found", "install ROCm, or point --rocm at it (the .venv's _rocm_sdk_devel works too)")


def hip_lib_dirs(root: Path) -> list[str]:
    dirs = [str(d.resolve()) for d in [root / "lib", *sorted(ROOT.glob(".venv/lib*/python*/site-packages/_rocm_sdk_libraries_*/lib"))]
            if d.is_dir()]
    return dirs


def detect_gfx(backend: str) -> str:
    if backend == "cuda":
        out = subprocess.run(["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
                             capture_output=True, text=True)
        if out.returncode == 0 and out.stdout.strip():
            caps = sorted({x.strip().replace(".", "") for x in out.stdout.splitlines() if x.strip()})
            return ";".join(caps)                        # CMAKE_CUDA_ARCHITECTURES form, e.g. 1200
        return "1200"
    try:
        out = subprocess.run(["rocminfo"], capture_output=True, text=True, timeout=20)
        gfx = sorted({tok for line in out.stdout.splitlines() for tok in line.split() if tok.startswith("gfx")})
        if gfx:
            return ";".join(gfx)
    except (OSError, subprocess.TimeoutExpired):
        pass
    return "gfx1201"


def build_engine(a) -> None:
    backend = a.build
    eng = ROOT / "engine"
    eng.mkdir(exist_ok=True)
    archs = a.arch or detect_gfx(backend)
    if backend == "hip":
        root = hip_root(a.rocm)
        bitcode = next((p for p in (root / "lib" / "llvm" / "amdgcn" / "bitcode", root / "amdgcn" / "bitcode")
                        if p.is_dir()), root / "amdgcn" / "bitcode")
        libs = hip_lib_dirs(root)
        os.environ.update({"HIP_PLATFORM": "amd", "HIP_COMPILER": "clang", "HIP_RUNTIME": "rocclr",
                           "ROCM_PATH": str(root), "HIP_PATH": str(root)})
        os.environ["LD_LIBRARY_PATH"] = os.pathsep.join([*libs, os.environ.get("LD_LIBRARY_PATH", "")]).rstrip(os.pathsep)
        os.environ["PATH"] = os.pathsep.join([str(root / "bin"), str(root / "llvm" / "bin"), os.environ.get("PATH", "")])
        defs = ["-DSTRATA_ENABLE_HIP=ON", "-DSTRATA_ENABLE_CUDA=OFF", "-DSTRATA_BUILD_TESTS=OFF",
                "-DSTRATA_PREFILL_MMQ=ON", "-DSTRATA_Q6K_EXPERTS=ON", "-DSTRATA_MMQ_KQUANTS=ON",
                f"-DCMAKE_HIP_ARCHITECTURES={archs}",
                f"-DCMAKE_HIP_COMPILER={root / 'llvm' / 'bin' / 'clang++'}",
                f"-DCMAKE_HIP_COMPILER_ROCM_ROOT={root}",
                "-DCMAKE_PREFIX_PATH=" + ";".join([str(root), *[str(Path(d).parent) for d in libs[1:]]]),
                f"-DCMAKE_HIP_FLAGS=--rocm-path={root} --rocm-device-lib-path={bitcode}",
                f"-DSTRATA_GGML_DIR={llama_dir_for_build()}"]
        bdir = ROOT / "build-hip"
    else:
        nvcc = shutil.which("nvcc")
        if not nvcc:
            fail("no nvcc on PATH", "install the CUDA toolkit, or point PATH at it")
        defs = ["-DSTRATA_ENABLE_CUDA=ON", "-DSTRATA_BUILD_TESTS=OFF", "-DSTRATA_Q6K_EXPERTS=ON",
                "-DSTRATA_MMQ_KQUANTS=ON", f"-DCMAKE_CUDA_ARCHITECTURES={archs}",
                f"-DCMAKE_CUDA_COMPILER={nvcc}", f"-DSTRATA_GGML_DIR={llama_dir_for_build()}"]
        bdir = ROOT / "build"
    say(f"  configuring and compiling the {backend} engine for {archs} (10-20 minutes the first time) ...")
    run(["cmake", "-S", str(ROOT), "-B", str(bdir), "-G", "Ninja", *defs], env=dict(os.environ))
    run(["cmake", "--build", str(bdir), "--target", "strata"], env=dict(os.environ))
    shutil.copy2(bdir / EXE, eng / EXE)
    m = re.search(r"project\(strata VERSION ([\d.]+)", (ROOT / "CMakeLists.txt").read_text(encoding="utf-8"))
    version = m.group(1) if m else "local"
    stamp = {"source": f"local-{backend}", "backend": backend, "version": version, "archs": archs.split(";")}
    if (eng / (EXE.replace("strata", "strata-vision"))).is_file():
        stamp["vision"] = "cpu"
    if backend == "hip":
        stamp["lib_dirs"] = hip_lib_dirs(hip_root(a.rocm))
    (eng / "BUILD.json").write_text(json.dumps(stamp, indent=1) + "\n", encoding="utf-8")
    ok(f"engine built: {eng / EXE}")
    say("  (the image encoder is a separate build when you want it: cmake -S tools/vision -B build-vision && "
        "cmake --build build-vision --target strata-vision)")


def llama_dir_for_build() -> Path:
    if not LLAMA_DIR.is_dir():
        say("  fetching the pinned llama.cpp (ggml comes from it) ...")
        LLAMA_DIR.parent.mkdir(parents=True, exist_ok=True)
        run(["git", "clone", "https://github.com/ggml-org/llama.cpp.git", str(LLAMA_DIR)])
        run(["git", "-C", str(LLAMA_DIR), "checkout", LLAMA_PIN])
    return LLAMA_DIR


# -------------------------------------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="print the catalog and exit")
    ap.add_argument("--model", help="the catalog name to download and prepare")
    ap.add_argument("--build", choices=("hip", "cuda"), help="also compile the engine into engine/")
    ap.add_argument("--arch", help="the GPU architecture (gfx1201, or 1200 for CUDA); auto-detected by default")
    ap.add_argument("--rocm", help="the ROCm root for the HIP build (default: $ROCM_PATH, the .venv's sdk, /opt/rocm)")
    ap.add_argument("--data-dir", help="the data folder (default: ../Strata-data)")
    ap.add_argument("--context", type=int, default=131072, help="max context for the run config (default 131072)")
    ap.add_argument("--low-ram", action="store_true", help="write the experts into one file (packs/experts.bin) "
                                                          "for PCs whose RAM does not hold them")
    ap.add_argument("--no-vision", action="store_true", help="leave pictures off in the run config")
    ap.add_argument("--engine-path", help="use this engine binary in the run config instead of engine/strata")
    ap.add_argument("--force", action="store_true", help="overwrite an existing strata-<name>.json")
    ap.add_argument("--port", type=int, default=8080, help="the port written into the run config (default 8080)")
    ap.add_argument("--repair", action="store_true", help="repair the environment: reinstall requirements and compile the engine")
    a = ap.parse_args()

    if a.repair:
        say("Repairing environment:")
        pip = [sys.executable, "-m", "pip", "install", "--quiet", "--disable-pip-version-check"]
        if (ROOT / "requirements.txt").is_file():
            say("  reinstalling requirements.txt ...")
            run(pip + ["--force-reinstall", "-r", str(ROOT / "requirements.txt")])
        if (ROOT / "requirements-build.txt").is_file():
            say("  reinstalling requirements-build.txt ...")
            run(pip + ["--force-reinstall", "-r", str(ROOT / "requirements-build.txt")])
        b = a.build or ("hip" if shutil.which("hipcc") or os.environ.get("ROCM_PATH") or (ROOT / "tools" / "hip").is_dir() else "cuda")
        a.build = b
        say(f"  building engine ({b}) ...")
        build_engine(a)
        ok("repair complete")
        return 0

    if a.list or not (a.model or a.build):
        say("strata-fabric catalog (the engine reads the GGUF; the pack holds the tokenizer, and with --low-ram "
            "the experts):\n")
        for name, m in CATALOG.items():
            say(f"  {name:<14} {m['gb']:6.1f} GB  {m['about']}")
        say("\nNVFP4 variants are not in this slim catalog (ready-made packs, NVIDIA-only).")
        if not (a.model or a.build):
            say("\ngive --model NAME and/or --build hip|cuda (see --help).")
        return 0
    if a.build:
        build_engine(a)
    if a.model:
        prepare_model(a)
    return 0


if __name__ == "__main__":
    sys.exit(main())
