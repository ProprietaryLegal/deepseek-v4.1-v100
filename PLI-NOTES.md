# PLI Labs notes: this engine on a smaller 8× V100 host (2026-11-01, interim)

This branch (`pli-v100-notes-2026-11`) adds only this file and [`pli-patches/`](pli-patches/). `main` in this fork is
identical to [philbert440/deepseek-v4.1-v100](https://github.com/philbert440/deepseek-v4.1-v100) `main`
(`23cec6a9`, tag `ds41-vision-20261025b`).

The engine, launcher and EXL3 franken-v100 pack are Phil Tompkins' work, built on 1CatAI's
[1Cat-vLLM](https://github.com/1CatAI/1Cat-vLLM); the model and the Engram tables are DeepSeek's. Thank you. These
notes record how we run it on a host smaller than the engine's test box and what we measured. Every number is a
finished measurement on one machine; work is ongoing, and final production numbers will follow in
[ProprietaryLegal/deepseek-v4-flash-v100](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100) (full report:
[docs/v4.1-flash-1cat-vllm-2026-11-01.md](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100/blob/v2026.11.01/docs/v4.1-flash-1cat-vllm-2026-11-01.md),
numbers as JSON:
[results.json](https://github.com/ProprietaryLegal/deepseek-v4-flash-v100/blob/v2026.11.01/evidence/v4.1-flash-2026-11-01/results.json)).

## What we run

| | ours | engine default / test box |
|---|---|---|
| GPUs | 8× V100-SXM2-32GB, two NVLink quads | same |
| Layout | PP2 × TP4, one stage per quad, cut at layer 20 | same |
| Drafter | DSpark from the pack, 5 draft tokens | same |
| Host CPU / NUMA | 12-core Threadripper PRO 3945WX, **one NUMA node** | 64 cores |
| Host RAM | 8 × 32 GB DDR4 RDIMM at DDR4-2666, MemTotal ~251 GiB | 372 GB |
| Power and clocks | 180 W per card, graphics clock capped at 1425 MHz, `DS41_HOST_TUNING=0` | launcher locks 1530 MHz with host tuning |
| `--max-model-len` × `--max-num-seqs` | 327,680 × 3 (KV pool 927,004 tokens at boot) | 1,048,576 × 2 |
| Engram tables | NVMe + page cache (`VLLM_DS41_ENGRAM_RESIDENT=0`) in the baseline; resident (`=1`) in the arm below | resident |
| NUMA binding | all ranks on node 0 ([`pli-patches/0001`](pli-patches/0001-numa-single-node.patch)) | nodes 0 0 0 0 1 1 1 1 |
| Other | `--max-num-batched-tokens 1024`, prefix caching on, Engram lookahead 0 | same batched tokens |

Software: Ubuntu 24.04.4, kernel 6.8, driver 580.159.03, CUDA 12.8.93, Python 3.12.3, torch 2.10.0+cu128, the release
wheel `1cat_vllm-1.5.2.dev1149+ds41vision1025b` (sha256 `1a5fc4cd…cacf`), pack @ `99d85a46`, Engram shards from
`deepseek-ai/DeepSeek-V4.1-Flash-1017` @ `1bae0e10`.

We started non-resident because the host had 196 GiB when first installed. After a RAM upgrade we measured resident
Engram, the engine's default.

## Measured (finished, 2026-11-01, DDR4-2666, 1425 MHz cap, 180 W)

Protocol: ~75,000-token prompts from a fixed corpus of public long-form documents, one per seed, 2,048-token output
cap, a fresh `cache_salt` on every request (0 prefix hits). **Cold** = page cache dropped before the request; **warm**
= the same prompt again with a new salt. Six seeds per speed cell; each arm uses its own unseen seeds. Decode beside
prefill = a 3,000-token decode stream with a cold 75K prefill submitted 10 s later. Stress = three ~300,000-token
prompts at once. 95 % confidence intervals on the change. Other CPU workloads ran on the host during both arms; the
GPUs ran only this server.

| | baseline, Engram non-resident (00:36-01:33 UTC) | Engram resident (01:51-02:17 UTC) | change, 95 % CI |
|---|---|---|---|
| cold 75K TTFT, mean of 6 | 46.00 s (sd 6.21, 37.88-52.29) | **26.94 s** | **−41.4 %** (−46.5, −34.8) |
| cold prefill | 1,656 tok/s | 2,785 tok/s | +68.1 % (+51.9, +86.9) |
| warm 75K TTFT, mean of 6 | 27.02 s (sd 0.07) | 27.10 s | +0.3 % (−0.6, +1.7) |
| decode tok/s, thinking off (cold cells / warm cells) | 42.15 / 40.91 | 46.41 / 44.78 | +10.1 / +9.4 % (CI spans 0) |
| decode beside a cold 75K prefill | 18.2 tok/s (32.2 before), longest gap 2.10 s | 29.5 tok/s, longest gap 1.21 s | — |
| that prefill's TTFT | 104.3 s | 53.6 s | −48.6 % |
| 3 × ~300K stress: wall; TTFTs | 476 s; 194 / 339 / 469 s | 368 s; 120 / 241 / 361 s | 0 preemptions in both |
| gates | greedy recording 3 / 3 runs identical; functional check 15 / 15 | greedy 16 / 16 byte-identical to baseline; functional 15 / 15 | |

Baseline, thinking on, warm: TTFT 29.88 s, decode 32.55 tok/s (1.35 accepted per draft; ~2.0 with thinking off).

Resident cost on this host: four stage-0 ranks × 45.78 GiB, loaded one at a time in 20.8-27.3 s each (1.80-2.36 GB/s,
O_DIRECT, 32 threads); about 208 GiB of ~251 GiB held while the server runs; lowest MemAvailable during the arm
32.0 GiB; no swap growth, no OOM kill. Because of the per-rank "tables + 32 GiB" check, a resident start here needs
about 241 GiB available with the server stopped. The functional check covers chat, streaming, JSON mode, a tool round
trip, `tool_choice=required`, thinking on/off/default, five reasoning efforts, a 900-word repetition check and 32K
three-needle recall.

## Patches ([`pli-patches/`](pli-patches/))

Both apply to `ds41-vision-20261025b` (`git apply --check` passes).

- [`0001-numa-single-node.patch`](pli-patches/0001-numa-single-node.patch): `v100/serve_ds41.sh` binds all eight ranks
  to NUMA node 0, for a single-node host. In production use on our machine since 2026-10-31.
- [`0003-mp1-exl3-tiled-smem-staging.patch`](pli-patches/0003-mp1-exl3-tiled-smem-staging.patch),
  **kernel-level only**: the first step of `docs/moe_prefill_persistent.md`. Shared-memory-staged k-tile loads (coalesced
  128-bit loads into a per-warp ring, one chunk ahead) for the 16- and 32-row tiles of the layout-1 EXL3 tiled MoE
  prefill kernel, behind `VLLM_DS41_MOE_EXL3_PREFILL_SMEM=1` (default off; falls back unless the trellis bases are
  16-byte aligned). Decode, MMA order, k summation order and the reduction are unchanged, so outputs are bit-identical.
  Measured 2026-10-31 on one V100 (1425 MHz cap, 180 W, no server, T = 1,016 tokens, TP rank 0, layers 5 and 12, two
  repetitions, through the patched `launch_tiled`): gate −9.1 % to −12.3 %, down −7.5 % to −13.2 % per tiled pass,
  bit-identical every run; the unchanged 64-row path read ±3 % (noise). It changes `csrc/`, so `_C` must be rebuilt.
  Its effect on server prefill has not been measured yet.
- [`mp1-harness/`](pli-patches/mp1-harness/): the single-GPU harness for 0003 (`mp1_bench.py`: upstream vs staged
  kernel, same inputs, bit comparison and CUDA-event timing; `mp1_dispatch.py`: the patched dispatcher, knob off with
  `--save`, knob on with `--compare`). `--plan` wants `{"layers": {"<layer>": [[expert_id, K], ...]}}`; the pack's
  `exl3_moe_plan.json` stores K as a 384-digit string per layer, so convert it first.

(0002, a launcher change for a KV-pool experiment, is not included: its measurement has not finished.)

## Reproduce

```bash
git clone -b ds41-vision-20261025b https://github.com/philbert440/deepseek-v4.1-v100 && cd deepseek-v4.1-v100
git apply /path/to/pli-patches/0001-numa-single-node.patch          # single-NUMA-node hosts only
# install the release wheel and download the pack + Engram shards as in README.md, then in ds41.env:
export DS41_MAX_MODEL_LEN=327680 DS41_MAX_NUM_SEQS_OVERRIDE=3 DS41_HOST_TUNING=0
export VLLM_DS41_ENGRAM_RESIDENT=1                                   # 0 = the non-resident baseline
v100/serve_ds41.sh --dry-run --strict && v100/serve_ds41.sh
```

Cold cells need `sync; echo 3 | sudo tee /proc/sys/vm/drop_caches` before each request; send a fresh `cache_salt` per
request so the prefix cache cannot serve it.

## Hardware notes (observed on our one machine; not about the engine)

- Two host resets left ACPI BERT records of a fatal PCIe Completion Timeout (AER uncorrectable status `0x00004000`) at
  the root port feeding a GPU board: 2026-10-22 (180 W, clocks up to 1477 MHz) and 2026-10-31 (180 W, SM clocks
  uncapped). Per-card `power.draw` readings above the 180 W limit appeared both uncapped (192-232 W) and under the
  1425 MHz cap (196-212 W in 500 ms samples on 2026-11-01). Our inference, not proven: a board drops off its PCIe link
  on load transients. No reset since the 2026-11-01 boot during the arms above.
- At DDR4-3200 one DIMM channel threw four uncorrected ECC errors (three during the first 392 GiB checkpoint load,
  which SIGBUSed two workers). At DDR4-2666 on the same DIMMs: three 200 GiB × 5-pattern probes with 0 mismatches and
  EDAC 0 / 0 through both arms. A mitigation, not a repair.
