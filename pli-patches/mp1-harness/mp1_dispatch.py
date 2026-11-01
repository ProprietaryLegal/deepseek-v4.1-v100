#!/usr/bin/env python3
"""MP1 dispatch check for patches/pending/0003: run the patched launch_tiled on mp1_bench.py's inputs.

  VLLM_DS41_MOE_EXL3_PREFILL_SMEM=0 python mp1_dispatch.py --patched <file.cu> ... --save ref.pt
  VLLM_DS41_MOE_EXL3_PREFILL_SMEM=1 python mp1_dispatch.py --patched <file.cu> ... --compare ref.pt
The knob is read once per process (as in the engine), hence two processes. Prints JSON (numbers only).
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

from mp1_bench import tiles_for, timed

HERE = Path(__file__).resolve().parent


def extract_patched(src: Path, out: Path) -> None:
    text = src.read_text().splitlines(keepends=True)
    first = next(i for i, l in enumerate(text) if l.startswith("#include"))
    stop = next(i for i, l in enumerate(text) if l.startswith("// xg = had128(x[t] * g_suh[e])"))
    body = "".join(text[first:stop])
    assert "VLLM_DS41_MOE_EXL3_PREFILL_SMEM" in body and "void launch_tiled(" in body
    out.write_text(f"// extracted unmodified from {src} lines {first + 1}-{stop}\n#pragma once\n" + body)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patched", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--T", type=int, default=1016)
    ap.add_argument("--dense-min", type=int, default=160)
    ap.add_argument("--iters", type=int, default=50)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--save", type=Path)
    g.add_argument("--compare", type=Path)
    args = ap.parse_args()
    from vllm.models.deepseek_v41.common.contracts import HIDDEN, MOE_INTER, N_EXPERTS, TOP_K
    from vllm.models.deepseek_v41.sm70 import moe_exl3 as mx

    args.build.mkdir(parents=True, exist_ok=True)
    extract_patched(args.patched, args.build / "patched_tiled.cuh")
    (args.build / "mp1_dispatch_ext.cu").write_text((HERE / "mp1_dispatch_ext.cu").read_text())
    ext = load(name="mp1_dispatch_ext", sources=[str(args.build / "mp1_dispatch_ext.cu")],
               build_directory=str(args.build), extra_include_paths=[str(args.build)], extra_cflags=["-O3"],
               extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70"], verbose=False)
    cfg = json.loads(args.plan.read_text())["layers"][str(args.layer)]
    bpw = np.array([x[1] for x in cfg], dtype=np.int64)
    hist = np.load(args.upstream / f"v100/balance/v1-ab24/layer{args.layer:02d}.npy").astype(np.float64)
    counts_hist = hist[1] if hist.ndim == 2 else hist
    owner = mx.assign_experts(bpw / bpw.sum(), counts_hist, 4)
    local = np.flatnonzero(owner == 0)
    kl = bpw[local]
    p = counts_hist / counts_hist.sum()
    rng = np.random.default_rng(args.layer)
    ids = np.stack([rng.choice(N_EXPERTS, size=TOP_K, replace=False, p=p) for _ in range(args.T)])
    loc_of = np.full(N_EXPERTS, -1)
    loc_of[local] = np.arange(len(local))
    lid = loc_of[ids.ravel()]
    tiers, R = tiles_for(np.bincount(lid[lid >= 0], minlength=len(local)), args.dense_min)
    dev = torch.device("cuda")
    gen = torch.Generator(device=dev).manual_seed(args.layer)
    smem = os.environ.get("VLLM_DS41_MOE_EXL3_PREFILL_SMEM", "")
    out = {"layer": args.layer, "knob": smem, "projections": {}}
    ys: dict[str, torch.Tensor] = {}
    ref = torch.load(args.compare) if args.compare else None
    for name, k, n in (("gate", HIDDEN, MOE_INTER), ("down", MOE_INTER, HIDDEN)):
        words = np.array([k * n * int(b) // 32 for b in kl], dtype=np.int64)
        offs = torch.as_tensor(np.concatenate([[0], np.cumsum(words)[:-1]]), dtype=torch.int64, device=dev)
        tr = torch.randint(-2 ** 31, 2 ** 31 - 1, (int(words.sum()),), dtype=torch.int32, device=dev, generator=gen)
        kb = torch.as_tensor(kl, dtype=torch.int32, device=dev)
        x = (torch.randn(max(R, 1), k, device=dev, generator=gen) * 0.5).half()
        res = {}
        for mt, tl in tiers.items():
            if not tl:
                continue
            tiles = torch.as_tensor(tl, dtype=torch.int32, device=dev)
            y = torch.full((R, n), float("nan"), device=dev)
            ext.tiled(y, tr, offs, kb, tiles, x, mt)
            torch.cuda.synchronize()
            covered = torch.zeros(R, dtype=torch.bool, device=dev)
            for _, r0, r, _ in tl:
                covered[r0:r0 + r] = True
            key = f"{name}{mt}"
            ys[key] = y[covered].cpu()
            row = {"tiles": len(tl), "ms": round(timed(lambda: ext.tiled(y, tr, offs, kb, tiles, x, mt), args.iters), 4)}
            if ref is not None:
                row["bit_identical"] = bool(torch.equal(ref["y"][key], ys[key]))
                row["ref_ms"] = ref["ms"][key]
                row["speedup"] = round(ref["ms"][key] / row["ms"], 3)
            res[str(mt)] = row
        out["projections"][name] = res
        del tr, x
        torch.cuda.empty_cache()
    if args.save:
        torch.save({"y": ys, "ms": {f"{p}{m}": r["ms"] for p, v in out["projections"].items() for m, r in v.items()}},
                   args.save)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
