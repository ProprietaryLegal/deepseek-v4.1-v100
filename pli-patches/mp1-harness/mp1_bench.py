#!/usr/bin/env python3
"""MP1 harness: upstream EXL3 tiled kwarp prefill kernel vs the shared-memory-staged variant, one GPU, no checkpoint.

Inputs follow scripts/ds41_moe_prefill_microbench.py: one stage-0 layer's rank-local experts and K (pack plan), routing
drawn from the v1-ab24 histogram (T tokens x top-6), random trellis words (the decode does not depend on the data),
random activations. Rows are compacted by expert and cut into 16 / 32 / 64-row tiles by the engine's tier rule
(<= 16, <= 32, else 64-row tiles; >= dense_min rows go to the dense path and are skipped here). For the gate (k=5120,
n=2304) and down (k=2304, n=5120) projections and each tier: the outputs of both kernels are compared bit for bit, and
each is timed with CUDA events.

  CUDA_VISIBLE_DEVICES=<uuid> python mp1_bench.py --upstream <repo> --plan plan-list.json --build <dir> [--ch 8]
Prints JSON (numbers only).
"""
from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from torch.utils.cpp_extension import load

HERE = Path(__file__).resolve().parent


def extract_upstream(src: Path, out: Path) -> None:
    text = src.read_text().splitlines(keepends=True)
    first = next(i for i, l in enumerate(text) if l.startswith("#include"))
    stop = next(i for i, l in enumerate(text) if l.startswith("// Single block. loc[S]"))
    body = "".join(text[first:stop])
    assert "exl3_tiled_kwarp_kernel" in body and body.count("namespace {") == 1
    out.write_text("// extracted unmodified from " + str(src) + f" lines {first + 1}-{stop}\n#pragma once\n" + body)


def build_ext(upstream: Path, build: Path):
    build.mkdir(parents=True, exist_ok=True)
    extract_upstream(upstream / "csrc/sm70_turbomind/ops/exl3_moe_sm70.cu", build / "upstream_tiled.cuh")
    for f in ("mp1_ext.cu", "mp1_smem.cuh"):
        (build / f).write_text((HERE / f).read_text())
    return load(name="mp1_ext", sources=[str(build / "mp1_ext.cu")], build_directory=str(build),
                extra_include_paths=[str(build)], extra_cflags=["-O3"],
                extra_cuda_cflags=["-O3", "-gencode=arch=compute_70,code=sm_70", "-lineinfo", "-Xptxas=-v"]
                + os.environ.get("MP1_EXTRA_CFLAGS", "").split(),
                verbose=False)


def tiles_for(counts: np.ndarray, dense_min: int) -> tuple[dict[int, list[list[int]]], int]:
    """Engine tier rule over rows compacted by local expert: {16|32|64: [[e, row0, rows, 0], ...]}, total rows."""
    out: dict[int, list[list[int]]] = {16: [], 32: [], 64: []}
    row0 = 0
    for e, r in enumerate(counts.tolist()):
        if r == 0:
            continue
        if dense_min and r >= dense_min:
            pass
        elif r <= 16:
            out[16].append([e, row0, r, 0])
        elif r <= 32:
            out[32].append([e, row0, r, 0])
        else:
            for s in range(0, r, 64):
                out[64].append([e, row0 + s, min(64, r - s), 0])
        row0 += r
    return out, row0


def timed(fn, iters: int) -> float:
    fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(iters):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / iters


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--hist", type=Path, default=None)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--layer", type=int, default=5)
    ap.add_argument("--T", type=int, default=1016)
    ap.add_argument("--tp-rank", type=int, default=0)
    ap.add_argument("--dense-min", type=int, default=160)
    ap.add_argument("--ch", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    from vllm.models.deepseek_v41.common.contracts import HIDDEN, MOE_INTER, N_EXPERTS, TOP_K
    from vllm.models.deepseek_v41.sm70 import moe_exl3 as mx

    ext = build_ext(args.upstream, args.build)
    hist_dir = args.hist or (args.upstream / "v100/balance/v1-ab24")
    cfg = json.loads(args.plan.read_text())["layers"][str(args.layer)]
    bpw = np.array([x[1] for x in cfg], dtype=np.int64)
    hist = np.load(hist_dir / f"layer{args.layer:02d}.npy").astype(np.float64)
    counts_hist = hist[1] if hist.ndim == 2 else hist
    owner = mx.assign_experts(bpw / bpw.sum(), counts_hist, 4)
    local = np.flatnonzero(owner == args.tp_rank)
    kl = bpw[local]
    p = counts_hist / counts_hist.sum()
    rng = np.random.default_rng(args.layer)
    ids = np.stack([rng.choice(N_EXPERTS, size=TOP_K, replace=False, p=p) for _ in range(args.T)])
    loc_of = np.full(N_EXPERTS, -1)
    loc_of[local] = np.arange(len(local))
    lid = loc_of[ids.ravel()]
    counts = np.bincount(lid[lid >= 0], minlength=len(local))
    tiers, R = tiles_for(counts, args.dense_min)
    dev = torch.device("cuda")
    g = torch.Generator(device=dev).manual_seed(args.layer)
    out: dict[str, object] = {"layer": args.layer, "tp_rank": args.tp_rank, "T": args.T, "local_experts": len(local),
                              "rows": int(R), "tiles": {str(m): len(v) for m, v in tiers.items()},
                              "kbits_hist": {int(b): int((kl == b).sum()) for b in np.unique(kl)},
                              "gpu": torch.cuda.get_device_name(), "projections": {}}
    for name, k, n in (("gate", HIDDEN, MOE_INTER), ("down", MOE_INTER, HIDDEN)):
        words = np.array([k * n * int(b) // 32 for b in kl], dtype=np.int64)
        offs = torch.as_tensor(np.concatenate([[0], np.cumsum(words)[:-1]]), dtype=torch.int64, device=dev)
        assert all(int(o) % 4 == 0 for o in offs.tolist())
        tr = torch.randint(-2 ** 31, 2 ** 31 - 1, (int(words.sum()),), dtype=torch.int32, device=dev, generator=g)
        kb = torch.as_tensor(kl, dtype=torch.int32, device=dev)
        x = (torch.randn(max(R, 1), k, device=dev, generator=g) * 0.5).half()
        res: dict[str, object] = {"k": k, "n": n, "trellis_gib": round(tr.numel() * 4 / 2 ** 30, 3), "tiers": {}}
        for mt, tl in tiers.items():
            if not tl:
                continue
            tiles = torch.as_tensor(tl, dtype=torch.int32, device=dev)
            y0 = torch.zeros(R, n, device=dev)
            ext.tiled(y0, tr, offs, kb, tiles, x, mt, 0, 0)
            t0 = timed(lambda: ext.tiled(y0, tr, offs, kb, tiles, x, mt, 0, 0), args.iters)
            row = {"tiles": len(tl), "upstream_ms": round(t0, 4)}
            for ch in args.ch:
                y1 = torch.full((R, n), float("nan"), device=dev)
                ext.tiled(y1, tr, offs, kb, tiles, x, mt, 1, ch)
                torch.cuda.synchronize()
                covered = torch.zeros(R, dtype=torch.bool, device=dev)
                for e, r0, r, _ in tl:
                    covered[r0:r0 + r] = True
                same = bool(torch.equal(y0[covered], y1[covered]))
                maxdiff = float((y0[covered] - y1[covered]).abs().max()) if not same else 0.0
                t1 = timed(lambda: ext.tiled(y1, tr, offs, kb, tiles, x, mt, 1, ch), args.iters)
                row[f"ch{ch}"] = {"ms": round(t1, 4), "bit_identical": same, "max_abs_diff": maxdiff,
                                  "speedup": round(t0 / t1, 3)}
            res["tiers"][str(mt)] = row
        out["projections"][name] = res
        del tr, x
        torch.cuda.empty_cache()
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
