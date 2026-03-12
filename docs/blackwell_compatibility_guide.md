# Blackwell Compatibility Guide — sm_120 Reference

Source: https://docs.nvidia.com/cuda/blackwell-compatibility-guide/

## Compute Capabilities
- **sm_100** — Blackwell datacenter (B200), uses `tcgen05` tensor core ISA
- **sm_120** — Blackwell consumer (RTX 50-series), uses `mma.sync` tensor core ISA
- `sm_100a` / `compute_100a` features are NOT forward/backward compatible

## Binary Compatibility
- Cubin runs on same major revision + same or higher minor revision only
- sm_90 cubin does NOT run on sm_100 (different major)
- PTX compiled for compute_X runs on any higher compute capability
- Hopper PTX (`compute_90a`) explicitly incompatible with Blackwell

## Building for sm_120
```bash
# Requires CUDA Toolkit 13.0+
-gencode=arch=compute_120,code=sm_120
```

## Key Facts for This Project
- We target sm_120 — requires CUDA 13.0+
- sm_120 uses `mma.sync`, NOT `tcgen05` (datacenter sm_100)
- No backward compat with sm_90 cubins — must rebuild from source
