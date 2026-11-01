// MP1 step 1: shared-memory-staged k-tile loads for the layout-1 (k-split warps) EXL3 tiled prefill
// kernel. Each warp walks the same contiguous k range of its 32-column strip as exl3_tiled_kwarp_kernel, but loads
// CH k-tiles of trellis words at a time with coalesced 128-bit loads (prefetched into registers one chunk ahead)
// into a per-warp shared-memory buffer. Each lane then gathers the same words LaneWords::load would have read with
// __ldg. Decode (mul1/mcg), MMA order and the k summation order are unchanged, so the output must be bit-identical to
// the tiled kernel's. Included after the upstream helpers (Window, LaneWords, decode_tile_sm80, mma884, kWarps, ...)
// inside the same anonymous namespace.
#pragma once

constexpr int kMp1MaxK = 5;  // EXL3 bits per weight in this pack: 2..5

template <int CH>
struct Mp1Ring {
  // per warp: CH k-tiles x 2 tiles x 8K words, sized for the largest K
  static constexpr int kWords = CH * 16 * kMp1MaxK;
};

// Words of k-tile `j` of the staged chunk, as LaneWords::load would read them from global memory.
template <int K>
__device__ __forceinline__ void mp1_gather(LaneWords<K, false>& lw, const uint32_t* __restrict__ ring, int j,
                                           const Window<K, false>& win) {
  constexpr int kTW = 8 * K;
  const uint32_t* blk = ring + j * 2 * kTW;
  const int base = win.ja[0] % kTW, t0 = win.ja[0] - base;
#pragma unroll
  for (int m = 0; m < LaneWords<K, false>::kCount; ++m) lw.w[m] = blk[t0 + (base + m) % kTW];
}

// Chunk [tk, tk + cnt) of this warp's strip: cnt * 2 * kTW contiguous words (16 B aligned) -> registers.
template <int K, int CH>
__device__ __forceinline__ void mp1_fetch(uint4 (&pre)[(CH * 16 * K / 4 + 31) / 32], const uint32_t* __restrict__ sb,
                                          int tk, int cnt) {
  constexpr int kW = 16 * K;  // words per k-tile of a 32-column strip
  constexpr int kL = (CH * kW / 4 + 31) / 32;
  const int lane = threadIdx.x & 31;
  const int n4 = cnt * kW / 4;
  const uint4* src = reinterpret_cast<const uint4*>(sb + static_cast<int64_t>(tk) * kW);
#pragma unroll
  for (int i = 0; i < kL; ++i) {
    const int idx = i * 32 + lane;
    if (idx < n4) pre[i] = __ldg(src + idx);
  }
}

template <int K, int CH>
__device__ __forceinline__ void mp1_store(const uint4 (&pre)[(CH * 16 * K / 4 + 31) / 32], uint32_t* __restrict__ ring,
                                          int cnt) {
  constexpr int kW = 16 * K;
  constexpr int kL = (CH * kW / 4 + 31) / 32;
  const int lane = threadIdx.x & 31;
  const int n4 = cnt * kW / 4;
  uint4* dst = reinterpret_cast<uint4*>(ring);
#pragma unroll
  for (int i = 0; i < kL; ++i) {
    const int idx = i * 32 + lane;
    if (idx < n4) dst[idx] = pre[i];
  }
}

// Same contract as tiled_accum (k-tiles [tk0, tk1) of strip n0, rows of x), words staged through `ring`.
template <int K, int MT, int CB, int CH>
__device__ __forceinline__ void mp1_accum(const uint32_t* __restrict__ tr, const half* __restrict__ x, int rows,
                                          int k, int n0, int tk0, int tk1, float (&d)[MT / 8][8],
                                          uint32_t* __restrict__ ring) {
  constexpr int NG = MT / 8;
  constexpr int kTW = 8 * K;
  constexpr int kL = (CH * 16 * K / 4 + 31) / 32;
  const int lane = threadIdx.x & 31;
  const int q = (lane >> 2) & 3;
  const int nw = 8 * q + (lane >> 4) * 4 + lane % 4;
  const int t = nw >> 4, c = nw & 15;
  const int row = (lane >> 4) * 4 + lane % 4;
  const int ng = (rows + 7) >> 3;
  const int ktiles = k / 16;
  Window<K, false> win(t, c);
  int xo[NG];
#pragma unroll
  for (int g = 0; g < NG; ++g) xo[g] = min(8 * g + row, rows - 1) * k;
#pragma unroll
  for (int g = 0; g < NG; ++g) {
#pragma unroll
    for (int i = 0; i < 8; ++i) d[g][i] = 0.f;
  }
  const uint32_t* sb = tr + static_cast<int64_t>(n0 >> 5) * ktiles * (2 * kTW);
  uint4 pre[kL];
  int cnt = min(CH, tk1 - tk0);
  if (cnt <= 0) return;
  mp1_fetch<K, CH>(pre, sb, tk0, cnt);
  mp1_store<K, CH>(pre, ring, cnt);
  __syncwarp();
  for (int c0 = tk0; c0 < tk1; c0 += CH) {
    const int ncnt = min(CH, tk1 - (c0 + CH));
    if (ncnt > 0) mp1_fetch<K, CH>(pre, sb, c0 + CH, ncnt);   // next chunk in flight during this one
    for (int j = 0; j < cnt; ++j) {
      LaneWords<K, false> lw;
      mp1_gather<K>(lw, ring, j, win);
      uint32_t b[4][2];
      decode_tile_sm80<K, CB>(lw, win, b);
      const half* xk = x + (c0 + j) * 16;
#pragma unroll
      for (int g = 0; g < NG; ++g) {
        if (g < ng) {
          const uint4 lo = *reinterpret_cast<const uint4*>(xk + xo[g]);
          const uint4 hi = *reinterpret_cast<const uint4*>(xk + xo[g] + 8);
          const uint32_t xw[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};
#pragma unroll
          for (int u = 0; u < 4; ++u) {
            uint32_t a[2] = {xw[u], xw[u + 4]};
            mma884(d[g], a, b[u]);
          }
        }
      }
    }
    __syncwarp();
    if (ncnt > 0) {
      mp1_store<K, CH>(pre, ring, ncnt);
      __syncwarp();
    }
    cnt = ncnt;
  }
}

// tiled_kwarp_body with mp1_accum in place of tiled_accum; the cross-warp reduction is the same code, but its
// buffer `red` is the kernel's (one allocation for all K cases; a __shared__ array declared in the inlined body is
// allocated once per K instantiation, 4 x 4 KiB, which cost a block per SM).
template <int K, int MT, int CB, int CH>
__device__ __forceinline__ void mp1_kwarp_body(const uint32_t* __restrict__ tr, const half* __restrict__ x,
                                               float* __restrict__ y, int rows, int k, int n, int n0, int tk0,
                                               int tk1, uint32_t (*rings)[Mp1Ring<CH>::kWords],
                                               float (*red)[8][32]) {
  constexpr int NG = MT / 8;
  const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
  const int q = (lane >> 2) & 3;
  const int ng = (rows + 7) >> 3;
  const int per = (tk1 - tk0) / kWarps;
  float d[NG][8];
  mp1_accum<K, MT, CB, CH>(tr, x, rows, k, n0, tk0 + warp * per, tk0 + (warp + 1) * per, d, rings[warp]);
#pragma unroll
  for (int g = 0; g < NG; ++g) {
    if (g >= ng) break;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      red[warp][(lane & 1) + ((i >> 1) & 1) * 2 + (lane >= 16 ? 4 : 0)]
         [8 * q + ((i >> 2) & 1) * 4 + (lane & 2) + (i & 1)] = d[g][i];
    }
    __syncthreads();
    for (int e = threadIdx.x; e < 256; e += kWarps * 32) {
      const int m = e >> 5, nn = e & 31, r = 8 * g + m;
      if (r < rows) {
        float acc = 0.f;
#pragma unroll
        for (int w = 0; w < kWarps; ++w) acc += red[w][m][nn];
        y[static_cast<int64_t>(r) * n + n0 + nn] = acc;
      }
    }
    __syncthreads();
  }
}

template <int MT, int CB, int CH, int MINB>
__global__ void __launch_bounds__(kWarps * 32, MINB) exl3_tiled_kwarp_smem_kernel(
    const uint32_t* __restrict__ trellis, const int64_t* __restrict__ offs, const int* __restrict__ kbits,
    const int* __restrict__ tiles, const half* __restrict__ x, float* __restrict__ y, int k, int n, int NT,
    const uint32_t* __restrict__ trellis2, const half* __restrict__ x2, float* __restrict__ y2,
    int64_t split_stride) {
  __shared__ __align__(16) uint32_t rings[kWarps][Mp1Ring<CH>::kWords];
  __shared__ float red[kWarps][8][32];
  int tile = blockIdx.y;
  if (tile >= NT) {
    tile -= NT;
    trellis = trellis2;
    x = x2;
    y = y2;
  }
  const int e = tiles[4 * tile], row0 = tiles[4 * tile + 1];
  const int rows = min(tiles[4 * tile + 2], MT);
  if (e < 0 || rows <= 0) return;
  const int n0 = blockIdx.x * 32;
  const uint32_t* tr = trellis + offs[e];
  const half* xt = x + static_cast<int64_t>(row0) * k;
  float* yt = y + blockIdx.z * split_stride + static_cast<int64_t>(row0) * n;
  const int span = k / 16 / gridDim.z;
  const int tk0 = blockIdx.z * span;
#define EXL3_TILED_KWS(KK) \
  mp1_kwarp_body<KK, MT, CB, CH>(tr, xt, yt, rows, k, n, n0, tk0, tk0 + span, rings, red)
  EXL3_K_SWITCH_SM80(kbits[e], EXL3_TILED_KWS)
#undef EXL3_TILED_KWS
}
