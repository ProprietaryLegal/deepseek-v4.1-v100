// MP1 dispatch check: the patched upstream file (pending/0003 applied), extracted at build time from its first
// #include through the end of launch_tiled into patched_tiled.cuh, unmodified. tiled() calls the patched launch_tiled
// (layout 1, mul1) exactly as exl3_moe_tiled does, so VLLM_DS41_MOE_EXL3_PREFILL_SMEM picks the kernel.
#include <torch/extension.h>
#include "patched_tiled.cuh"  // ends after the anonymous namespace that holds launch_tiled

void tiled(torch::Tensor y, torch::Tensor trellis, torch::Tensor offs, torch::Tensor kbits, torch::Tensor tiles,
           torch::Tensor x, int64_t mt) {
  const int k = x.size(1), n = y.size(1), NT = tiles.size(0);
  if (NT == 0) return;
  dim3 grid(n / 32, NT, 1);
  auto s = at::cuda::getCurrentCUDAStream();
  auto tr = reinterpret_cast<const uint32_t*>(trellis.data_ptr());
  auto xp = reinterpret_cast<const half*>(x.data_ptr<at::Half>());
#define MP1_D(MTV)                                                                                      \
  if (mt == MTV) {                                                                                      \
    launch_tiled<MTV, kCodebookMul1>(1, grid, s, tr, offs.data_ptr<int64_t>(), kbits.data_ptr<int>(),   \
                                     tiles.data_ptr<int>(), xp, y.data_ptr<float>(), k, n, NT, nullptr, \
                                     nullptr, nullptr, 0);                                              \
    C10_CUDA_KERNEL_LAUNCH_CHECK();                                                                     \
    return;                                                                                             \
  }
  MP1_D(16)
  MP1_D(32)
  MP1_D(64)
#undef MP1_D
  TORCH_CHECK(false, "mt must be 16, 32 or 64");
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("tiled", &tiled); }
