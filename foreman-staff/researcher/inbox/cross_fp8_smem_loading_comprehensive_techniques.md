# FP8 Shared Memory Loading: Comprehensive Techniques Survey

**Sources:**
- https://research.google/blog/mixed-input-matrix-multiplication-performance-optimizations/ (Google: FragmentShuffler, narrower bitwidth loads vs register-level reordering)
- https://github.com/NVIDIA/cutlass/discussions/911 (CUTLASS: INT8 interleaved layout explanation, row permutation for ldmatrix)
- https://github.com/NVIDIA/cutlass/discussions/647 (CUTLASS: why ldmatrix cannot transpose 8-bit data)
- https://research.colfax-intl.com/adding-fp8-to-flashattention/ (Colfax: FP8 layout conformance with byte_perm + shfl_sync, 1 PFLOP/s FP8 FlashAttention)
- https://hazyresearch.stanford.edu/blog/2024-11-27-tk-fp8 (ThunderKittens: FP8 core matrices are 8x16 elements, swizzle modes)
- https://forums.developer.nvidia.com/t/understanding-cutlass-permuted-shared-memory-layout/303697 (CUTLASS: XOR permutation formula)
- https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/ (Lei Mao: XOR swizzle for 8-bit transpose)
- https://forums.developer.nvidia.com/t/how-to-load-fp8-using-ldmatrix-on-sm120-sm120a/330254 (NVIDIA forum: FP8 ldmatrix on sm_120)

**Relevant to:** attention worker, fused-mlp worker, gemm worker (cross-project)
**Worker's current problem:** Two workers (attention exp 66, fused-mlp FP8 GEMM1) independently discovered that scalar byte/uint16 loads from row-major FP8 shared memory are far slower than ldmatrix+CVT (converting BF16 on the fly). The fundamental issue: ldmatrix is a warp-collective 128-bit optimized instruction, while scalar byte/uint16 loads have bank conflicts at stride-64 and move 8-16 bits per thread per instruction.

---

## EXECUTIVE SUMMARY

This brief consolidates NEW findings from Google's mixed-input GEMM blog, CUTLASS discussions, and Colfax's FP8 FlashAttention that go beyond the existing ldmatrix reinterpret briefs. The key new insights are:

1. **Google identified two strategies for 8-bit fragment loading** -- and measured the performance tradeoff
2. **CUTLASS's INT8 "interleaved layout" is a ROW PERMUTATION** applied at global memory load time, not a smem transpose
3. **ldmatrix fundamentally cannot transpose 8-bit data** -- this is a hardware constraint, not a software limitation
4. **The XOR swizzle adapts naturally to 8-bit elements** -- same formula, different element-size parameter
5. **FP8 needs a 32-element minimum tile width** for proper swizzle alignment (confirmed by ThunderKittens)

---

## FINDING 1: Google's Two Strategies for 8-bit Fragment Loading

Google's mixed-input GEMM optimization blog identifies exactly two strategies for loading INT8/FP8 data from shared memory into MMA-ready register fragments. This is the most authoritative analysis of the tradeoff.

### Strategy 1: Narrower Bitwidth Loads (Direct Scalar)

Threads issue narrow-bitwidth memory loads (8-bit or 16-bit) moving INT8 data from smem directly to registers. This achieves layout conformance immediately -- the data lands in the correct registers for MMA -- but "does not utilize the full shared memory bandwidth."

This is what our workers attempted: scalar byte loads or uint16 loads from FP8 smem. It is correct but slow because the memory system is optimized for 32-bit and 128-bit loads, not 8-bit or 16-bit loads.

### Strategy 2: Wide Load + Register-Level Reordering (ldmatrix Reinterpret)

Use `ldmatrix` (128-bit warp-collective load) to get maximum smem bandwidth, then use `FragmentShuffler` to reorder data within registers to achieve layout conformance. Google's `FragmentShuffler` uses the `prmt` (byte permute) instruction to rearrange bytes, reducing the conversion from ~10 instructions to ~6 instructions per 4xU8 register (1.6x improvement).

**This is why ldmatrix.b16 reinterpret is the winning approach.** It maximizes smem bandwidth and handles layout conformance via cheap PRMT instructions after the load. The ~4-6 PRMT instructions are negligible compared to the bandwidth gain from using ldmatrix.

### Google's `FastNumericArrayConvertor`

Google also optimized the type conversion itself. Their `FastNumericArrayConvertor` operates on "4xU8 in 32-bit registers without unpacking individual 1xU8 values" using the `prmt` instruction to rearrange bytes. This could be relevant if our workers need to convert FP8 values to FP32/BF16 after loading.

---

## FINDING 2: CUTLASS INT8 "Interleaved Layout" Is a Row Permutation, Not a Smem Transpose

The CUTLASS discussion #911 clarifies a commonly misunderstood optimization. CUTLASS's "interleaved layout" for INT8 data is NOT a transpose of the matrix in shared memory. It is a **permutation of rows at global memory load time** so that after ldmatrix loads the data, it lands directly in the correct registers for MMA without inter-thread communication.

Key quotes from the discussion:

> "the interleaving (permutation of rows) was done so that the int8 data ends up in the pattern needed from HMMA after LDSM [ldmatrix]"

> "If we did not interleave the weights this way, we would have to have some communication among the threads to ensure each thread had the correct data from the weight matrix before dequantizing and issuing HMMA."

### What This Means Practically

For INT8 (and by extension FP8) GEMM:
- The B weight matrix is pre-processed ONCE at model load time
- Rows are permuted so that when loaded linearly into smem and then via ldmatrix, the byte distribution across threads matches what MMA expects
- No runtime transpose or shuffle is needed in the kernel
- The interleaving pattern is architecture-specific (different for Volta vs Turing vs Ampere)

### Relevance to Our Workers

For the fused-mlp worker's GEMM1:
- W1 (weight matrix) can be pre-permuted at model load time
- This eliminates ALL runtime overhead for fragment layout conformance
- The permutation is a one-time cost at model initialization

For the attention worker:
- K and V matrices change per forward pass, so pre-permutation is not practical
- The ldmatrix.b16 reinterpret + PRMT fixup approach is needed instead
- However, if K and V arrive in FP8 from a previous layer (native FP8 pipeline), the permutation could be baked into the output layout of the producing kernel

---

## FINDING 3: ldmatrix Cannot Transpose 8-bit Data -- Hardware Constraint

CUTLASS discussion #647 confirms definitively:

> "tensor core instruction is TN layout. ldmatrix can only transpose 16bit data."

This is why CUTLASS does not support NN layout for INT8 tensor core GEMM. The `.trans` modifier on `ldmatrix.m8n8.b16` transposes the 8x8 matrix of 16-bit elements during the load. But when the data is 8-bit, the 16-bit transposition moves PAIRS of bytes, not individual bytes. The result is a transposition at 16-bit granularity, which does NOT correctly transpose the underlying 8-bit elements.

### Implications

1. **ldmatrix.m8n8.x2.trans.b16 on FP8 data:** This transposes at 16-bit granularity. If FP8 is stored row-major with k as fast dimension, the "transpose" swaps k-pairs, not individual k-values. The resulting register layout may still be correct for col-major B operands IF the k-dimension is already the fast dimension in smem.

2. **ldmatrix.m16n16.x1.trans.b8:** This IS a native 8-bit transpose (sm_100+/sm_120a+). But it only exists as `.trans` -- there is no non-transposed version. And it requires the PRMT byte interleave after loading (confirmed by CUTLASS source).

3. **For row-major FP8 data (A operand):** Use `ldmatrix.m8n8.x4.b16` (no `.trans`) and load FP8 pairs as b16 "elements." The non-transposed load preserves byte order within each 16-bit pair.

4. **For column-major FP8 data (B operand, K-major):** Use `ldmatrix.m8n8.x2.trans.b16` on FP8 data. The 16-bit transpose will swap k-pairs, which is correct because consecutive k-values are already adjacent in memory (K-major layout). Each 32-bit register gets `{fp8[k], fp8[k+1], fp8[k+16], fp8[k+17]}` -- the exact pattern needed.

---

## FINDING 4: XOR Swizzle for 8-bit Elements

The XOR swizzle pattern works identically for 8-bit and 16-bit elements, but the address computation must account for the smaller element size. The fundamental formula is:

```
bank = (byte_address / 4) % 32
swizzled_col = col ^ (row % swizzle_period)
```

For 16-bit (BF16) elements:
- Each element is 2 bytes, so element address = col * 2
- bank = (col * 2 / 4) % 32 = (col / 2) % 32
- A row of 64 BF16 elements = 128 bytes = exactly 32 banks

For 8-bit (FP8) elements:
- Each element is 1 byte, so element address = col * 1
- bank = (col * 1 / 4) % 32 = (col / 4) % 32
- A row of 64 FP8 elements = 64 bytes = 16 banks (ONLY HALF the banks are used!)
- A row of 128 FP8 elements = 128 bytes = 32 banks (full bank utilization)

### Bank Conflict Implications

With 64 FP8 elements per row (our current tile width):
- Only 16 banks are accessed per row
- 4 consecutive FP8 bytes map to the same bank (4-way bank conflict for byte loads!)
- This is why scalar byte loads are catastrophically slow

With ldmatrix.b16 reinterpret:
- Loads 16-bit (2-byte) values, accessing every other bank
- Still only 16 unique banks per row of 64 FP8 elements
- But ldmatrix is a warp-collective instruction that loads 128 bits per thread across 8 addresses, so bank conflicts are distributed across the 8 rows being loaded simultaneously

### XOR Swizzle Adaptation

The same `row_idx ^ col_bank_idx` XOR swizzle works for FP8, but:
- `col_bank_idx` should be computed at 4-byte granularity (col / 4) for FP8
- Alternatively, treat FP8 pairs as 16-bit elements and use the existing swizzle unchanged
- The ThunderKittens minimum tile width of 32 FP8 elements ensures at least 8 bank groups are used, enabling the 32-byte swizzle mode

### Recommendation

Use the existing XOR swizzle pattern unchanged. Treat pairs of FP8 bytes as 16-bit elements for swizzle purposes. This is consistent with the ldmatrix.b16 reinterpret approach -- the smem layout is byte-identical to a BF16 layout with half the K-dimension.

---

## FINDING 5: ThunderKittens FP8 Core Matrix Size

ThunderKittens uses 8x16 FP8 elements as its "core matrix" (vs 8x8 for BF16/FP16). This is because m16n8k32 has twice the k-dimension of m16n8k16:

| Precision | MMA shape | Core matrix | Bytes per core | Minimum tile width |
|-----------|-----------|-------------|----------------|-------------------|
| BF16 | m16n8k16 | 8x8 | 128 B | 8 elements |
| FP8 | m16n8k32 | 8x16 | 128 B | 32 elements |

The byte count per core matrix is the same (128 bytes). The difference is only in how many elements that represents. This means:
- Same smem bandwidth per MMA
- Same bank conflict pattern (when using ldmatrix)
- Same swizzle pattern applicability

---

## FINDING 6: The Complete FP8 Loading Pipeline

Combining all findings, the complete pipeline for loading FP8 data from global memory through shared memory into MMA registers is:

### For B operand (weights, can be pre-processed):

```
Global Memory (FP8, pre-permuted layout)
    |
    v  cp.async.cg (16-byte, raw bytes)
    |
Shared Memory (FP8, same byte layout, XOR swizzled)
    |
    v  ldmatrix.m8n8.x2.trans.b16 (treats FP8 pairs as b16)
    |
Registers (2 x uint32 per thread)
    |
    v  [PRMT if byte order needs fixing -- test empirically]
    |
MMA B operand ready
```

### For A operand (activations, per-inference):

```
Global Memory (BF16 or FP8)
    |
    v  cp.async.cg (16-byte, raw bytes)
    |
Shared Memory (BF16 or FP8, XOR swizzled)
    |
    v  OPTION A: ldmatrix.m8n8.x4.b16 (if FP8, treats pairs as b16)
    |  OPTION B: ldmatrix.m8n8.x4.b16 + cvt.e4m3x2 (if BF16, convert after load)
    |
Registers (4 x uint32 per thread)
    |
    v  [PRMT if byte order needs fixing -- test empirically]
    |
MMA A operand ready
```

### For P-to-FP8 A operand (attention kernel, accumulator to next MMA input):

```
Registers (FP32 accumulators, 4 x float32 per thread)
    |
    v  cvt.rn.satfinite.e4m3x2.f32 (pack 2 FP8 values per instruction)
    |
Registers (FP8 packed in uint32, wrong layout for MMA A operand)
    |
    v  __byte_perm (intra-thread byte rearrangement)
    v  __shfl_sync (inter-thread data exchange, within groups of 4)
    |
MMA A operand ready (correct FP8 fragment layout)
```

---

## WHAT'S NEW VS PREVIOUS BRIEFS

| Finding | Previous briefs covered? | New information |
|---------|-------------------------|-----------------|
| ldmatrix.b16 reinterpret for FP8 | YES | No new info, confirmed |
| ldmatrix.m16n16.b8 on sm_120 | YES | No new info, confirmed |
| P-to-FP8 with byte_perm + shfl_sync | YES | No new info, confirmed |
| Google's 2-strategy framework | NO | New -- names the tradeoff explicitly (narrow loads vs wide load + reorder) |
| CUTLASS INT8 row permutation at load time | NO | New -- pre-permutation of weights eliminates runtime overhead |
| ldmatrix cannot transpose 8-bit data (hardware) | NO | New -- explains WHY .trans on b16 is wrong for individual FP8 bytes |
| XOR swizzle adaptation for 8-bit | Partially | New -- explicit bank analysis showing 4-way conflict for byte loads |
| ThunderKittens 8x16 core matrix for FP8 | Partially | New -- explains the 32-element minimum tile width requirement |
| Complete loading pipeline diagram | NO | New -- end-to-end flow from global to MMA for all three cases |

---

## ACTIONABLE RECOMMENDATIONS

### For the attention worker (FP8 kernel, exp 67+):

1. **Do the minimal verification test from the consolidated strategy brief.** Load known FP8 data via ldmatrix.x4.b16, feed to m16n8k32 MMA, check correctness. This is the single highest-value experiment right now.

2. **If byte order is wrong:** Add PRMT fixup. Try `__byte_perm(reg, 0, 0x3120)` or `__byte_perm(reg, 0, 0x2031)` to swap byte pairs within each register. There are only a few plausible permutations to test.

3. **For P-to-FP8 conversion:** Start with the Colfax pattern (byte_perm + shfl_sync). Test on sm_120 -- the mma.sync fragment layout may differ from WGMMA, requiring different selector values. The CuTe layout strides from CUTLASS mma_traits_sm89.hpp are the reference.

### For the fused-mlp worker (FP8 GEMM1):

1. **Pre-permute W1 weights at model load time.** Apply the CUTLASS INT8 interleaved row permutation to the weight matrix. This is a one-time host-side operation that eliminates ALL runtime fragment layout overhead.

2. **Use ldmatrix.b16 reinterpret for FP8 W1.** Same instruction as BF16, just adjust address calculations for sizeof(fp8)=1. The pre-permutation ensures the loaded data is already in the correct MMA fragment layout.

3. **Keep BF16-to-FP8 conversion for A (input X).** The A operand (activations) changes per inference, so pre-permutation is not practical. Continue using ldmatrix for BF16 + cvt.e4m3x2 for conversion.

### For the gemm worker (if pursuing FP8 improvements):

1. **Same approach as fused-mlp.** Pre-permute B matrix weights. Use ldmatrix.b16 reinterpret for both A and B FP8 operands.

2. **Consider the XOR swizzle adaptation.** If tile width is 64 FP8 elements (32 bytes per row), the swizzle covers only 8 bank groups. Consider widening to 128 FP8 elements (64 bytes per row) for full 16 bank groups, or use the pair-based swizzle (treat as 32 16-bit elements).
