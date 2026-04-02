# 3xTF32 Algorithm for FP32-Accurate GEMM on Tensor Cores (Ootomo & Yokota 2022)

**Source:** https://arxiv.org/abs/2203.03341 (IJHPCA 2022)
**Source:** https://dl.acm.org/doi/10.1145/3624062.3624084 (SC'23 Workshop, ORNL TF32/TF64)
**Relevant to:** numerical/ worker (LU factorization trailing GEMM update)
**Worker's current problem:** TF32 MMA (m16n8k8) is broken on sm_120 due to B fragment diagonal broadcasting. Need an alternative way to get tensor-core-accelerated GEMM with FP32 accuracy.

---

## What This Is

The 3xTF32 algorithm decomposes FP32 matrix multiplication into 3 TF32 tensor core multiplications that together recover full FP32 accuracy. Achieves 1.7x native FP32 SGEMM performance on A100.

**However:** On sm_120, TF32 MMA is broken (B fragment defect). This technique is NOT directly applicable. It is documented here for completeness and because the BF16x9 approach (see companion brief) achieves the same goal more effectively on Blackwell.

---

## The TF32 Decomposition Algorithm

### Step 1: Split FP32 into TF32 + Remainder

TF32 has 10-bit mantissa (vs FP32's 23-bit). The same 8-bit exponent as FP32.

```
For each FP32 value v:
  v_tf32 = round_to_TF32(v)          // truncate mantissa to 10 bits
  delta_v = v - v_tf32                // remainder (fits in TF32 after scaling)
  delta_v_tf32 = round_to_TF32(delta_v * 2^10)  // scaled remainder
```

The reconstruction is:
```
v ≈ v_tf32 + 2^(-10) * delta_v_tf32
```

### Step 2: Matrix Decomposition

```
A = A_tf32 + 2^(-10) * dA_tf32
B = B_tf32 + 2^(-10) * dB_tf32
```

### Step 3: Compute Product

Full expansion:
```
A*B = A_tf32 * B_tf32                           // Term 1: main product
    + 2^(-10) * (dA_tf32 * B_tf32 + A_tf32 * dB_tf32)  // Terms 2-3: corrections
    + 2^(-20) * dA_tf32 * dB_tf32                // Term 4: negligible
```

**Key insight:** Term 4 (dA * dB) has magnitude at least 2^20 times smaller than term 1. Since FP32 mantissa is 23 bits, this term affects at most the LSB. **Dropping it loses negligible accuracy.**

Therefore: **3 MMA calls** instead of 4:
1. C1 = A_tf32 * B_tf32          (main product)
2. C2 = dA_tf32 * B_tf32         (correction 1)
3. C3 = A_tf32 * dB_tf32         (correction 2)
4. Result = C1 + 2^(-10) * (C2 + C3)

All three MMAs use TF32 inputs with FP32 accumulation.

### Performance on A100

- **TF32 method: 33 TFLOPS** (full FP32 exponent range)
- **FP16 method: 51 TFLOPS** (limited exponent range)
- **Native FP32 SGEMM: 19.5 TFLOPS** (theoretical peak)
- **Speedup: 1.7x** over native FP32

### ORNL 3xTF32 (Valero-Lara et al., SC'23)

ORNL's implementation achieved **53 TFLOPS on A100** (2.6x over SGEMM) with slight accuracy relaxation. They also proposed a "TF64" framework that uses 4 TF32 GEMMs to achieve FP64 accuracy:

```
For FP64 → TF32 decomposition:
A_f64 ≈ A1_tf32 + A2_tf32  (each TF32 covers ~10 bits)
B_f64 ≈ B1_tf32 + B2_tf32

A*B ≈ A1*B1 + A1*B2 + A2*B1 + A2*B2  (4 TF32 GEMMs)
```

---

## Why This Does NOT Work on sm_120

**TF32 MMA (m16n8k8) is broken on sm_120.** The B operand has a diagonal broadcast defect where B[k,n] AND B[k+1,n+1] receive the same value. This makes TF32 MMA unusable for general matrix multiplication. Verified empirically with 25+ tests.

The 3xTF32 algorithm requires working TF32 MMA, so it cannot be used on our hardware.

---

## What CAN Be Used: BF16 Equivalents

### Option A: BF16x9 (cuBLAS 13.0+, Blackwell)

Similar decomposition but with BF16 (7-bit mantissa, same exponent as FP32):
```
v = v0_bf16 + 2^(-8) * v1_bf16 + 2^(-16) * v2_bf16
```

Full expansion produces 9 BF16 GEMMs for exact FP32 accuracy. cuBLAS handles this automatically with `CUBLAS_COMPUTE_32F_EMULATED_16BFX9`.

Performance: 3-4x native FP32 on Blackwell.

### Option B: 3xBF16 (Custom Implementation)

Analogous to 3xTF32 but using BF16:
```
A = A_bf16 + 2^(-8) * dA_bf16
B = B_bf16 + 2^(-8) * dB_bf16

A*B ≈ A_bf16 * B_bf16 + 2^(-8) * (dA_bf16 * B_bf16 + A_bf16 * dB_bf16)
```

**3 BF16 MMA calls.** But accuracy is worse than 3xTF32 because BF16 has only 7-bit mantissa (vs TF32's 10-bit). The remainder term covers bits 8-15, so the combined 14-bit effective precision is short of FP32's 23 bits.

**NOT recommended for LU trailing updates** -- insufficient precision.

### Recommendation

Use BF16x9 (Option A) for guaranteed FP32 accuracy with 3-4x speedup. The 9x MMA cost is offset by BF16 tensor core throughput being ~16x native FP32 on Blackwell.

---

## Sources

- [Ootomo & Yokota, 2022 — Recovering Single Precision from Tensor Cores](https://arxiv.org/abs/2203.03341)
- [Valero-Lara et al., TF32/TF64 Frameworks (SC'23)](https://dl.acm.org/doi/10.1145/3624062.3624084)
- [CUTLASS 3xTF32 Discussion](https://github.com/NVIDIA/cutlass/discussions/390)
- [cuBLAS 13.2 FP Emulation Blog](https://developer.nvidia.com/blog/unlocking-tensor-core-performance-with-floating-point-emulation-in-cublas/)
