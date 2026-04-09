# ComfyUI Render Kernels — Project Specification

## 1. Executive Summary

### Problem Statement
ComfyUI (Melanie's installation at `/users/melanie/ComfyUI/`) uses pure PyTorch
for all GPU compute — zero custom CUDA kernels. On RTX 5090 (sm_120a), this
leaves significant performance on the table: PyTorch's generic cuDNN/SDPA
kernels don't exploit Blackwell's larger shared memory, higher SM count, or
`mma.sync` tensor core throughput. Diffusion inference is latency-sensitive
(users wait for each image) so every percentage point matters.

### Solution
Build 4 custom CUDA kernels targeting the hottest compute paths in the diffusion
pipeline: multi-head attention (30-40% of compute), fused GroupNorm+Linear
(15-20%), weight streaming (10-15% latency), and fused Conv2d+GroupNorm for VAE
decode (5-10%). All kernels must handle variable sizes driven by user-selected
resolution and model architecture.

### Factory Mode

`general_shape_library` — sizes vary with user resolution and model choice.

### Success Metrics
| Axis | Role | Target | Measurement |
|------|------|--------|-------------|
| Correctness | gate | max_err < 1e-3 (fp16/bf16), < 1e-6 (fp64 ref) | vs PyTorch reference |
| Coverage | gate | all shapes in benchmark set pass | shape matrix below |
| Numerical quality | gate | bitwise-identical softmax ordering | attention output comparison |
| Weighted performance | primary | >= 1.2x vs PyTorch SDPA (attn), >= 1.0x others | weighted across benchmark set |
| Complexity | secondary | single .cu per op, Python drop-in | maintenance burden |

### Keep / Discard Rule

- Keep when: >= 2% improvement on weighted benchmark set, no correctness regression
- Discard when: < 2% change (noise), or any shape in coverage matrix fails

## 2. Operations

### Operations Catalog

| Op | Job | Kernel Name | Reference | Target | Status |
|----|-----|-------------|-----------|--------|--------|
| mha | #65 | mha_sm120a | `torch.nn.functional.scaled_dot_product_attention` | >= 1.2x | planning |
| gnorm | #66 | gnorm_linear_sm120a | `F.group_norm` + `nn.Linear` | >= 1.0x | planning |
| wstream | #67 | wstream_sm120a | PyTorch weight casting / comfy-aimdo VBAR | >= 1.0x | planning |
| vae_conv | #68 | vae_conv_gnorm_sm120a | `nn.Conv2d` + `F.group_norm` | >= 1.0x | planning |

**Priority order:** mha first (largest % of compute, most headroom), gnorm second
(fuses two ops into one kernel launch), wstream third (bandwidth optimization),
vae_conv fourth (smaller % but still measurable).

**Eval.sh FUNCTIONS array:**
```bash
FUNCTIONS=(
    "mha|mha_sm120a"
    "gnorm|gnorm_linear_sm120a"
    "wstream|wstream_sm120a"
    "vae_conv|vae_conv_gnorm_sm120a"
)
```

## 3. Tensor Shape Specifications

### 3.1 Multi-Head Attention (Job #65)

**Input:** Q, K, V tensors of shape `[batch, num_heads, seq_len, head_dim]`

**What's fixed per model family:**
| Model | Hidden Dim | Num Heads | Head Dim | Context Dim |
|-------|-----------|-----------|----------|-------------|
| SD 1.5 | 320, 640, 1280 | 8 | 40 | 768 |
| SD 2.x | 320, 640, 1280 | 5, 10, 20 | 64 | 1024 |
| SDXL | 320, 640, 1280 | 5, 10, 20 | 64 | 2048 |
| Flux | 3072 | 24 | 128 | 4096 |
| Flux2 | 3072 | 48 | 64 | 4096 |

**What varies with user resolution (self-attention seq_len):**
| Resolution | Latent | Seq Len (SD/SDXL, /8) | Seq Len (Flux, /16) |
|------------|--------|-----------------------|---------------------|
| 512x512 | 64x64 | 4,096 | 1,024 |
| 768x768 | 96x96 | 9,216 | 2,304 |
| 1024x1024 | 128x128 | 16,384 | 4,096 |
| 1536x1536 | 192x192 | 36,864 | 9,216 |
| 2048x2048 | 256x256 | 65,536 | 16,384 |

**Cross-attention dimensions:**
| Model | Text Seq Len | Text Dim |
|-------|-------------|----------|
| SD 1.5 | 77 | 768 |
| SD 2.x | 77 | 1024 |
| SDXL | 77 | 2048 |
| Flux/Flux2 | ~300 (variable, T5) | 4096 |

**Summary of matmul shapes (QK^T):**
- M (query seq): 1,024 — 65,536 (variable)
- N (key seq): 77 — 65,536 (variable; 77 for cross-attn, same as M for self-attn)
- K (head dim): 40, 64, or 128 (fixed per model)
- Batch*heads: 1-8 batch × 5-48 heads = 5-384

**Precision:** fp16 or bf16 (accumulate in fp32). Softmax in fp32.

**Primary benchmark shapes:**
```
# Self-attention (hot path — quadratic scaling)
batch=1, heads=20, seq=4096,  head_dim=64   # SDXL @ 512px
batch=1, heads=20, seq=16384, head_dim=64   # SDXL @ 1024px
batch=1, heads=24, seq=4096,  head_dim=128  # Flux @ 1024px
batch=1, heads=24, seq=16384, head_dim=128  # Flux @ 2048px

# Cross-attention
batch=1, heads=20, seq_q=16384, seq_kv=77,  head_dim=64   # SDXL cross
batch=1, heads=24, seq_q=4096,  seq_kv=300, head_dim=128  # Flux cross
```

### 3.2 Fused GroupNorm + Linear (Job #66)

**GroupNorm parameters:**
- Groups: always 32 (hardcoded in ComfyUI)
- Channels: 320, 640, 1280 (SD/SDXL UNet), 3072 (Flux), 64-512 (VAE)
- eps: 1e-6, affine=True
- Spatial dims: same as attention seq_len (latent H×W)

**Linear projection (fused after norm):**
- Input: normalized tensor `[batch, channels, H, W]` → reshape to `[batch*H*W, channels]`
- Weight: `[out_features, in_features]` where both are channel dims
- Common sizes: 320→320, 640→640, 1280→1280, 3072→3072 (same-dim projections in transformer blocks)
- Also: 1280→5120 (feedforward up-proj), 5120→1280 (feedforward down-proj)

**Primary benchmark shapes:**
```
# UNet transformer blocks
batch=1, channels=1280, H=128, W=128, groups=32, out=1280  # SDXL 1024px
batch=1, channels=1280, H=64,  W=64,  groups=32, out=1280  # SDXL 512px
batch=1, channels=3072, H=64,  W=64,  groups=32, out=3072  # Flux 1024px
batch=1, channels=1280, H=128, W=128, groups=32, out=5120  # feedforward up

# VAE
batch=1, channels=512, H=64, W=64, groups=32, out=512      # VAE bottleneck
```

### 3.3 Weight Streaming (Job #67)

**Problem:** Large model weights (2-10 GB) must be cast from storage dtype
(fp16/fp8/int8) to compute dtype (fp16/bf16) and streamed to SMs.

**Key dimensions:**
- Weight tensors: linear layers are `[out, in]` where in/out ∈ {320, 640, 1280, 3072, 4096, 5120}
- Largest single weight: 5120×1280 = 6.5M params (26 MB at fp32)
- Total model weights: ~860M params (SD1.5), ~2.6B (SDXL), ~12B (Flux)
- Target bandwidth: approach 1792 GB/s (GDDR7 theoretical peak)

**Strategy:** cp.async double-buffer with XOR swizzle, overlap compute with next-layer prefetch.

**Benchmark:** end-to-end weight transfer latency for one UNet forward pass.

### 3.4 Fused Conv2d + GroupNorm — VAE Decode (Job #68)

**Conv2d parameters:**
- Kernel: always 3×3, stride 1, padding 1 (residual blocks)
- Downsampling: 3×3, stride 2, padding 0 (encoder) — 3-4 layers
- Upsampling: nearest-neighbor interpolate + 3×3 conv (decoder)

**Channel progression (decoder, latent→image):**
```
z_channels → 512 → 512 → 512 → 256 → 128 → 64 → 3 (RGB)
```
Where z_channels = 4 (SD), 16 (Flux), 128 (Flux2).

**Spatial dims (decoder output grows):**
| Resolution | VAE Input | After Upsample 1 | After Upsample 2 | After Upsample 3 | Output |
|------------|-----------|-------------------|-------------------|-------------------|--------|
| 512×512 | 64×64 | 128×128 | 256×256 | 512×512 | 512×512 |
| 1024×1024 | 128×128 | 256×256 | 512×512 | 1024×1024 | 1024×1024 |

**GroupNorm in VAE:** channels ∈ {64, 128, 256, 512}, groups=32, eps=1e-6

**Primary benchmark shapes:**
```
# VAE decode residual blocks (most repeated)
batch=1, in_ch=512, out_ch=512, H=64,  W=64,  kernel=3x3, groups=32
batch=1, in_ch=512, out_ch=256, H=128, W=128, kernel=3x3, groups=32
batch=1, in_ch=256, out_ch=128, H=256, W=256, kernel=3x3, groups=32
batch=1, in_ch=128, out_ch=64,  H=512, W=512, kernel=3x3, groups=32
```

## 4. Architecture

### Compute-bound operations (MMA / tensor core)

**mha (Job #65):** Tiled flash-attention style.
- QK matmul uses `mma.sync` with fp16/bf16 inputs, fp32 accumulators
- Online softmax (numerically stable, single-pass)
- Output projection PV uses same MMA pipeline
- Tile sizes TBD based on head_dim (40/64/128 require different tiling)
- Target: 2-stage pipeline, double-buffer K/V tiles via cp.async

**gnorm_linear (Job #66):** Fuse reduction + elementwise + matmul.
- GroupNorm: parallel reduction per group (mean, variance)
- Normalize + scale/bias in registers
- Feed directly into MMA for the linear projection (no global memory round-trip)
- This eliminates one full read+write of the activation tensor

### Bandwidth-bound operations (vectorized loads + reductions)

**wstream (Job #67):** Pure memory bandwidth optimization.
- cp.async with XOR swizzle for bank-conflict-free shared memory loads
- Double-buffer: load next layer's weights while current layer computes
- Cast fp16→bf16 or int8→fp16 in shared memory during the copy
- Coordinate with model_management.py's existing offload streams

**vae_conv (Job #68):** Mixed compute + bandwidth.
- 3×3 conv is 9× multiply-accumulate per output pixel (compute-bound at high channels)
- GroupNorm after conv is bandwidth-bound (reduction over spatial dims)
- Fusing avoids writing conv output to global memory, then reading it back for norm
- im2col in shared memory → MMA for the matmul view of conv

## 5. File Structure

```
csrc/
├── common/                          # Shared (DO NOT MODIFY — symlinked)
└── comfy_render/
    ├── mha_sm120a.cu                # Multi-head attention
    ├── gnorm_linear_sm120a.cu       # Fused GroupNorm + Linear
    ├── wstream_sm120a.cu            # Weight streaming pipeline
    └── vae_conv_gnorm_sm120a.cu     # Fused Conv2d + GroupNorm

python/blackwell_kernels/
├── comfy_render.py                  # Python wrappers (all 4 ops)
└── __init__.py

tests/test_comfy_render.py           # Correctness tests (all 4 ops)
benchmarks/bench_comfy_render.py     # Benchmarks (all 4 ops)
profiles/profile_comfy_render.py     # ncu profiling (accepts --op <name>)
```

## 6. Testing

### Test Configs

| Op | Test | Config | Tolerance |
|----|------|--------|-----------|
| mha | tiny | batch=1, heads=1, seq=16, dim=64 | 1e-3 rel (fp16) |
| mha | sd15 | batch=1, heads=8, seq=4096, dim=40 | 1e-3 rel |
| mha | sdxl | batch=1, heads=20, seq=16384, dim=64 | 1e-3 rel |
| mha | flux | batch=1, heads=24, seq=4096, dim=128 | 1e-3 rel |
| mha | cross | batch=1, heads=20, seq_q=16384, seq_kv=77, dim=64 | 1e-3 rel |
| gnorm | small | channels=320, H=16, W=16, groups=32 | 1e-3 rel |
| gnorm | sdxl | channels=1280, H=128, W=128, groups=32, out=1280 | 1e-3 rel |
| gnorm | flux | channels=3072, H=64, W=64, groups=32, out=3072 | 1e-3 rel |
| gnorm | ffn_up | channels=1280, H=128, W=128, groups=32, out=5120 | 1e-3 rel |
| vae | bottleneck | in=512, out=512, H=64, W=64, k=3x3, groups=32 | 1e-3 rel |
| vae | upsample | in=256, out=128, H=256, W=256, k=3x3, groups=32 | 1e-3 rel |
| vae | final | in=128, out=64, H=512, W=512, k=3x3, groups=32 | 1e-3 rel |

### Coverage Matrix

**mha:**
- tiny / degenerate: seq=1, seq=16, head_dim=40 (non-power-of-2)
- tile boundaries: seq=63, 64, 65, 127, 128, 129
- rectangular (cross-attn): seq_q=16384, seq_kv=77
- batch: 1, 2, 4, 8
- head_dim: 40, 64, 128 (all three must work)
- causal mask: yes/no

**gnorm:**
- tiny: channels=64, H=1, W=1
- tile boundaries: H*W = 63, 64, 65, 4095, 4096, 4097
- all channel dims: 64, 128, 256, 320, 512, 640, 1280, 3072
- non-square out: 1280→5120, 5120→1280

**vae_conv:**
- tiny: H=1, W=1
- tile boundaries: H=63, 64, 65
- all channel combos: 512→512, 512→256, 256→128, 128→64, 64→3
- stride 1 (residual) and stride 2 (downsample)

## 7. Integration with ComfyUI

**Drop-in replacement strategy:**
- Build as a Python extension module (`blackwell_kernels.comfy_render`)
- Monkey-patch ComfyUI's `comfy/ops.py` at startup via a custom node or hook
- Replace `F.scaled_dot_product_attention` → `comfy_render.mha`
- Replace `F.group_norm` + `nn.Linear` pairs → `comfy_render.gnorm_linear`
- Replace VAE decode conv+norm chains → `comfy_render.vae_conv_gnorm`
- Fallback to PyTorch for unsupported shapes/dtypes

**Location:** `/users/melanie/ComfyUI/custom_nodes/blackwell_kernels/` (symlink to built .so)

## 8. Constraints

- **DO NOT modify `csrc/common/`** — symlinked from common/
- **ALWAYS** use `CUDA_VISIBLE_DEVICES=1` (GPU 1 = air-cooled kernel dev GPU)
- **ALWAYS** use `CUDA_HOME=/usr/local/cuda-13` for builds (CUDA 13.2)
- sm_120a uses `mma.sync`, NOT `tcgen05`
- **Precision:** fp16 and bf16 inputs, fp32 accumulation. No fp8 for now.
- **Causal mask:** mha must support both causal and non-causal modes
- **Memory:** all kernels must work within 32 GB GDDR7 (no host fallback)
