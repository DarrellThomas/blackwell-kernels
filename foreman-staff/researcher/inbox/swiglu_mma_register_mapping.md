# SwiGLU: MMA Register Mapping for Column Interleaving on sm_120

**Source:** Bitdefender blog + fal.ai blog + worker's existing mma.sync knowledge
**Relevant to:** fused-mlp worker (Phase 4: SwiGLU)
**Worker's current problem:** Needs to know exactly how interleaved columns map to thread registers after mma.sync, to implement the SwiGLU epilogue correctly.

## The Key Insight: mma.sync Gives You Column Pairs for Free

The m16n8k16 (BF16) and m16n8k32 (FP8) mma.sync instructions produce C/D
fragments where each thread holds elements from two adjacent columns. This is
a fundamental property of the Ampere/Blackwell tensor core layout.

For m16n8k16 BF16 (our primary instruction):
- Each thread in a warp holds **4 output elements** in 2 registers
- These 4 elements span **2 rows x 2 columns** (adjacent column pairs)
- The two columns are always consecutive (e.g., columns 0&1, 2&3, etc.)

When weights are interleaved as `[up_col0, gate_col0, up_col1, gate_col1, ...]`:
- Even columns = up-projection values
- Odd columns = gate values
- Each thread's register pair naturally contains one up value and one gate value
- **No register shuffling needed** -- the hardware gives you the (up, gate) pair

## Epilogue Implementation (Register Level)

After the MMA accumulation loop completes for a tile, each thread has
accumulated results in registers. The SwiGLU epilogue operates entirely
in registers:

```
// For each (up, gate) pair in the thread's accumulators:
// acc[even_col] = up_val (accumulated dot product with W_up column)
// acc[odd_col]  = gate_val (accumulated dot product with W_gate column)

float up_val   = acc[2*i];       // even column
float gate_val = acc[2*i + 1];   // odd column

// SiLU(gate) = gate * sigmoid(gate) = gate / (1 + exp(-gate))
float silu_gate = gate_val / (1.0f + expf(-gate_val));

// SwiGLU output
float result = up_val * silu_gate;
```

On sm_120, `expf` maps to `MUFU.EX2` (approximate exp2) + arithmetic.
The entire epilogue is ~5 instructions per element pair -- negligible
compared to the MMA compute.

## Output Store: Half the Columns

The GEMM produces 2*D_ff columns, but SwiGLU output is only D_ff columns.
Each (up, gate) pair produces one output value.

**Store pattern:**
- The GEMM tile is `BLOCK_M x (2 * BLOCK_N_eff)` in the output
- After the epilogue, you write `BLOCK_M x BLOCK_N_eff` values
- The output column index is `gemm_col / 2` (equivalently `gemm_col >> 1`)
- Global memory store addresses shift accordingly

This means the output tensor is D_ff wide, not 2*D_ff. Memory traffic
for the store phase is halved compared to a standard GEMM epilogue.

## Tile Size Considerations

The worker's existing tiles are 64x64 with BLOCK_K=64 (post-optimization).
For SwiGLU with column interleaving:

- The GEMM dimension is `[M, D] x [D, 2*D_ff]`
- The N-dimension of the GEMM is doubled (2*D_ff instead of D_ff)
- Each 64-column tile produces 32 output columns after gating
- Grid dimension along N = `2*D_ff / 64 = D_ff / 32`

**Compared to two separate GEMMs (gate + up):**
- Two separate GEMMs: grid_N = `2 * (D_ff / 64)` = `D_ff / 32` tiles total
- One interleaved GEMM: grid_N = `2*D_ff / 64` = `D_ff / 32` tiles total
- **Same total compute** -- but one kernel launch instead of two

The real win is eliminating the intermediate writes. With two separate GEMMs,
you write 2*M*D_ff values to global memory (gate tensor + up tensor), then
read them back for the element-wise multiply. With interleaving, the multiply
happens in registers and you write only M*D_ff values.

## Weight Preparation (Python Side)

One-time setup at model load:
```python
# Original LLaMA-style weights:
#   W_gate: [D, D_ff]  -- gate projection
#   W_up:   [D, D_ff]  -- up projection

# Interleaved for fused kernel:
W_combined = torch.empty(D, 2 * D_ff, dtype=torch.bfloat16)
W_combined[:, 0::2] = W_up     # even columns = up
W_combined[:, 1::2] = W_gate   # odd columns = gate
```

**N must be divisible by the tile width** (64 in our case). Since
2*D_ff is always even and D_ff is typically a multiple of 128+ in
LLaMA models (D_ff=11008 for 7B, D_ff=13824 for 13B), this is
always satisfied. The Python-side padding logic already handles
non-tile-aligned dimensions.

## Comparison to Existing v1 Epilogue

The worker's current v1 kernel does:
```
GEMM: [M, D] x [D, D_ff] -> [M, D_ff]   (one projection)
Epilogue: relu_sq(result)                  (in registers)
Store: [M, D_ff]                           (to global memory)
```

SwiGLU changes this to:
```
GEMM: [M, D] x [D, 2*D_ff] -> [M, 2*D_ff]  (both projections, interleaved)
Epilogue: result = up * SiLU(gate)            (in registers, from column pairs)
Store: [M, D_ff]                              (half the GEMM width)
```

The GEMM core is identical -- same mma.sync, same tiling, same double-buffering.
Only the epilogue logic and store pattern change.

## SiLU Instruction Cost on sm_120

`SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))`

In SASS this compiles to approximately:
1. `FNEG` (negate gate_val)
2. `MUFU.EX2` (fast exp2, need to convert base: exp(-x) = exp2(-x * log2(e)))
3. `FMUL` (multiply by log2(e) before EX2)
4. `FADD` (1.0 + exp result)
5. `FRCP` (fast reciprocal, or `FDIV`)
6. `FMUL` (gate_val * sigmoid result)
7. `FMUL` (up_val * silu_gate result)

Total: ~7 FP32 instructions per element pair. At D_ff=3072 (GPT-2 scale),
that's 3072 * 7 = ~21K instructions per row -- trivial compared to the MMA
workload of 2*D*D_ff = 2*768*3072 = ~4.7M multiply-accumulate operations.
