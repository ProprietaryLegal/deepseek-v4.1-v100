// MP1 harness extension: today's exl3_tiled_kwarp_kernel (extracted at build time from the upstream
// csrc/sm70_turbomind/ops/exl3_moe_sm70.cu into upstream_tiled.cuh, unmodified) and the shared-memory-staged variant
// (mp1_smem.cuh), launched on identical inputs. Python: mp1_bench.py.
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_fp16.h>
#include <cstdint>

#include "upstream_tiled.cuh"  // opens `namespace {` and leaves it open

#include "mp1_smem.cuh"

template <int MT>
void launch_orig_mt(dim3 grid, cudaStream_t s, const uint32_t* tr, const int64_t* op, const int* kp, const int* tp,
                    const half* x, float* y, int k, int n, int NT) {
  constexpr int PK = MT >= 64 ? EXL3_TILED_KW_P64 : (MT >= 32 ? EXL3_TILED_KW_P32 : EXL3_TILED_KW_P16);
  constexpr int MINB = MT >= 64 ? EXL3_TILED_MINB64 : (MT >= 32 ? EXL3_TILED_MINB32 : EXL3_TILED_MINB16);
  exl3_tiled_kwarp_kernel<MT, kCodebookMul1, PK, MINB><<<grid, kWarps * 32, 0, s>>>(
      tr, op, kp, tp, x, y, k, n, NT, nullptr, nullptr, nullptr, 0);
}

template <int MT, int CH>
void launch_smem_mt(dim3 grid, cudaStream_t s, const uint32_t* tr, const int64_t* op, const int* kp, const int* tp,
                    const half* x, float* y, int k, int n, int NT) {
  constexpr int MINB = MT >= 64 ? EXL3_TILED_MINB64 : (MT >= 32 ? EXL3_TILED_MINB32 : EXL3_TILED_MINB16);
  auto kern = exl3_tiled_kwarp_smem_kernel<MT, kCodebookMul1, CH, MINB>;
  static bool once = false;
  if (!once) {
    cudaFuncSetAttribute(kern, cudaFuncAttributePreferredSharedMemoryCarveout, cudaSharedmemCarveoutMaxShared);
    once = true;
  }
  kern<<<grid, kWarps * 32, 0, s>>>(tr, op, kp, tp, x, y, k, n, NT, nullptr, nullptr, nullptr, 0);
}

}  // namespace (opened by upstream_tiled.cuh)

// variant 0 = upstream tiled kwarp kernel, 1 = MP1 smem-staged. tiles: int32 [NT, 4] (expert, row0, rows, 0).
void tiled(torch::Tensor y, torch::Tensor trellis, torch::Tensor offs, torch::Tensor kbits, torch::Tensor tiles,
           torch::Tensor x, int64_t mt, int64_t variant, int64_t ch) {
  const int k = x.size(1), n = y.size(1), NT = tiles.size(0);
  TORCH_CHECK(k % 64 == 0 && n % 32 == 0, "k % 64, n % 32");
  TORCH_CHECK(reinterpret_cast<uintptr_t>(trellis.data_ptr()) % 16 == 0, "trellis base must be 16 B aligned");
  if (NT == 0) return;
  dim3 grid(n / 32, NT, 1);
  auto s = at::cuda::getCurrentCUDAStream();
  auto tr = reinterpret_cast<const uint32_t*>(trellis.data_ptr());
  auto op = offs.data_ptr<int64_t>();
  auto kp = kbits.data_ptr<int>();
  auto tp = tiles.data_ptr<int>();
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
  auto yp = y.data_ptr<float>();
#define MP1_MT(MTV)                                                                       \
  if (mt == MTV) {                                                                        \
    if (variant == 0) launch_orig_mt<MTV>(grid, s, tr, op, kp, tp, xp, yp, k, n, NT);     \
    else if (ch == 2) launch_smem_mt<MTV, 2>(grid, s, tr, op, kp, tp, xp, yp, k, n, NT);  \
    else if (ch == 4) launch_smem_mt<MTV, 4>(grid, s, tr, op, kp, tp, xp, yp, k, n, NT);  \
    else if (ch == 8) launch_smem_mt<MTV, 8>(grid, s, tr, op, kp, tp, xp, yp, k, n, NT);  \
    else if (ch == 16) launch_smem_mt<MTV, 16>(grid, s, tr, op, kp, tp, xp, yp, k, n, NT); \
    else TORCH_CHECK(false, "ch must be 2, 4, 8 or 16");                                  \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                       \
    return;                                                                               \
  }
  MP1_MT(16)
  MP1_MT(32)
  MP1_MT(64)
#undef MP1_MT
  TORCH_CHECK(false, "mt must be 16, 32 or 64");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tiled", &tiled); }
