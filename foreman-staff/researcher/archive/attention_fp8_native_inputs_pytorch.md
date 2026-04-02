# FP8 Native Inputs via PyTorch — Eliminating Conversion Overhead

**Source:** https://dev-discuss.pytorch.org/t/float8-in-pytorch-1-x/1815 | https://docs.nvidia.com/cuda/cuda-math-api/cuda_math_api/struct____nv__fp8__e4m3.html | https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254
**Relevant to:** attention worker (main/)
**Worker's current problem:** FP8 attention's BF16-to-FP8 conversion adds ~448 ALU instructions per KV block (14:1 ratio vs MMA count), creating dependency chains that increase scoreboard stalls from 12% to 18% and limit SM throughput to 43.8%.

## What This Is

PyTorch 2.2+ natively supports `torch.float8_e4m3fn` tensors. K and V can be pre-quantized to FP8 on the host side before the attention kernel launches, eliminating ALL in-kernel conversion overhead. Combined with NVIDIA's `__nv_fp8_e4m3` CUDA type, custom kernels can accept FP8 inputs directly.

## Why It Matters for Us

This is the attention worker's **#1 next direction** (explicitly listed in agent_state.md as "highest-ROI next step"). The numbers:

- **Current FP8:** 52 us bench, 2.33x SDPA. SM 43.8%, scoreboard 18%.
- **Conversion cost:** ~448 ALU instructions/KV block for K+V BF16→FP8 conversion. ~7.5 us added to theoretical minimum.
- **With native FP8 inputs:**
  - Eliminates ALL 448 conversion instructions per KV block
  - Halves cp.async bandwidth (1 byte/element vs 2 bytes for BF16)
  - Halves shared memory usage (16KB/block → potential 6 blocks/SM)
  - Target: ~44 us (per worker's estimate)

## Key Technique

### Host-side quantization (Python):
```python
# Scale and quantize to FP8 before kernel launch
K_fp8 = (K_bf16 * scale_k).to(torch.float8_e4m3fn)
V_fp8 = (V_bf16 * scale_v).to(torch.float8_e4m3fn)
# Pass to custom kernel — 1 byte per element, contiguous
custom_attention_kernel(Q_bf16, K_fp8, V_fp8, scale_k, scale_v, ...)
```

### CUDA kernel side:
```cpp
// Access raw FP8 data — torch.float8_e4m3fn is 1 byte per element
const uint8_t* K_fp8 = K.data_ptr<uint8_t>();  // or reinterpret as __nv_fp8_e4m3*

// cp.async loads 1 byte per element (half the bandwidth of BF16)
// Use cp.async.cg.shared.global with 16-byte alignment for 16 FP8 values at once

// FP8 MMA: mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32
// Takes uint32 operands packing 4 FP8 values each — same as current kernel
```

### Key NVIDIA types:
- `__nv_fp8_e4m3` — single FP8 value, has `__x` member for raw byte
- `__nv_fp8x2_e4m3` — packed pair
- `__nv_fp8x4_e4m3` — packed quad (fits in uint32 for MMA operand)

### ldmatrix consideration:
- **ldmatrix does NOT directly support FP8** with convenient shapes on sm_120
- Workaround: use regular `ldmatrix` treating the data as 16-bit elements (each "element" is actually 2 packed FP8 values), then the MMA instruction interprets the uint32 registers correctly
- Alternative: direct `lds` loads (4 bytes at a time) bypassing ldmatrix — may be acceptable since FP8 data is half the size

## Caveats

- **API change required:** Kernel signature changes from `(bf16* K, bf16* V)` to `(uint8_t* K_fp8, uint8_t* V_fp8, float scale_k, float scale_v)`. This is a breaking change to the kernel interface.
- **Quantization quality:** Pre-quantizing K/V to FP8 may lose more precision than the current approach of converting within the kernel (where values are fresh from BF16 computation). Per-tensor scaling may need to become per-head or per-block scaling for quality.
- **ldmatrix compatibility:** Without direct FP8 ldmatrix support, the shared memory layout and load pattern may need restructuring. The current ldmatrix_x4 pattern loads 8 BF16 values per thread; with FP8, this becomes 16 values — the swizzle pattern may need adjustment.
- **PyTorch FP8 limitations:** Some operations (like `torch.cat`) don't work on FP8 tensors yet. Need to ensure K/V can be properly shaped/strided before kernel launch.
- **Q stays BF16:** Only K and V benefit from FP8 (Q is loaded once and reused). The kernel must handle mixed-precision inputs.
