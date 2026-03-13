# PTX Inline Assembly — Register Allocation and Compiler Control

**Purpose:** How to write inline PTX that controls register allocation, prevents spills, and interacts correctly with the CUDA compiler's optimizer.
**Last updated:** 2026-03-13

---

## 1. Constraint Letters (Operand Types)

**Source:** https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html

Each inline PTX operand must specify exactly one constraint letter mapping to a PTX register type:

| Constraint | PTX Type | C Type | Bits | Use |
|------------|----------|--------|------|-----|
| `"h"` | `.u16` | `short` | 16 | Half-precision scalars |
| `"r"` | `.u32` | `unsigned` | 32 | MMA fragment regs, ldmatrix output |
| `"l"` | `.u64` | `unsigned long long` | 64 | Global memory pointers |
| `"q"` | `.u128` | platform-dependent | 128 | Wide loads (platform-dependent) |
| `"f"` | `.f32` | `float` | 32 | FP32 accumulators, MMA C/D operands |
| `"d"` | `.f64` | `double` | 64 | FP64 values |
| `"n"` | immediate | `int` (const) | — | Compile-time constants (cp.async size, wait_group count) |

### Modifier Prefixes

| Modifier | Meaning | Example |
|----------|---------|---------|
| `"="` | Write-only output | `"=r"(out)` |
| `"+"` | Read-write (both input and output) | `"+f"(acc)` |
| (none) | Read-only input | `"r"(in)` |

### Examples for MMA Operands

```c
// MMA: A uses "r" (packed BF16x2 in uint32), C/D use "f" (FP32 accumulators)
asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(D[0]), "=f"(D[1]), "=f"(D[2]), "=f"(D[3])     // 4 x f32 output
    : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),          // 4 x u32 input (packed bf16)
      "r"(B[0]), "r"(B[1]),                                 // 2 x u32 input (packed bf16)
      "f"(C[0]), "f"(C[1]), "f"(C[2]), "f"(C[3]));         // 4 x f32 accumulator in
```

**Critical:** Multiple constraint letters for a single operand are NOT allowed (unlike GCC x86 asm).

---

## 2. volatile vs Non-volatile

### asm volatile

Prevents the compiler from:
- **Deleting** the asm block (even if outputs appear unused)
- **Moving** the asm block relative to other volatile statements
- **Reordering** with respect to other volatile asm blocks

### asm (non-volatile)

The compiler MAY:
- **Reorder** the instruction relative to other non-volatile code
- **Delete** it if outputs are provably unused
- **Duplicate** it if beneficial

### When to Use Which

| Instruction | volatile? | Why |
|-------------|-----------|-----|
| `mma.sync` | **NO** | Let compiler reorder MMAs for scheduling |
| `ldmatrix` | **YES** | Must not move across `__syncthreads()` barriers |
| `cp.async` | **YES** | Must respect commit/wait ordering |
| `cp.async.commit_group` | **YES** | Ordering with cp.async is critical |
| `cp.async.wait_group` | **YES** | Must complete before reading shared memory |

**This is the most important scheduling decision.** Making MMA non-volatile is how spatters.ca achieved 93% peak — the compiler interleaves MMAs with loads from different phases.

### The "memory" Clobber

Adding `"memory"` after a third colon tells the compiler that the asm block may read/write arbitrary memory:

```c
asm volatile("..." ::: "memory");
```

Use sparingly — this prevents the compiler from hoisting loads/stores across the asm block, which can pessimize scheduling.

---

## 3. Temporary Registers Inside asm Blocks

You can declare local PTX registers inside an asm block using `.reg` directives:

```c
asm volatile(
    "{ .reg .u32 t1;\n"               // Local scope with braces
    "  .reg .f32 t2;\n"
    "  cvt.rn.bf16x2.f32 t1, %1, %0;\n"  // Convert two f32 → packed bf16x2
    "  mov.b32 %2, t1;\n"
    "}\n"
    : "=r"(packed)
    : "f"(val0), "f"(val1));
```

### Braces for Local Scope

Wrap `.reg` declarations in `{ ... }` to create a local scope. This prevents "duplicate definition" errors when the function containing the asm is inlined multiple times. Without braces, the compiler would see multiple `.reg .u32 t1` declarations in the same scope.

```c
// WRONG — breaks when inlined
asm volatile(".reg .u32 t1;\n mov.u32 t1, %0;\n" :: "r"(x));

// RIGHT — braces create local scope
asm volatile("{ .reg .u32 t1;\n mov.u32 t1, %0;\n }\n" :: "r"(x));
```

### When to Use Temporary Registers

- **BF16 conversion:** `cvt.rn.bf16x2.f32` needs a temporary for intermediate packed result
- **Address computation:** Computing swizzled addresses with multiple steps
- **Predicated operations:** Need a predicate register (`.pred`)

---

## 4. Register Pressure and the Compiler

### How the Compiler Allocates Registers

1. Inline asm operands consume registers: each `"r"` or `"f"` operand occupies a register
2. The compiler allocates registers for ALL live variables simultaneously
3. Spills happen when live registers exceed the hardware limit (255/thread) or the `launch_bounds` maximum

### Checking Register Usage

```bash
# During build:
CUDA_HOME=/usr/local/cuda-13 nvcc -O2 --ptxas-options=-v ...
# Output shows: "Used N registers, M bytes smem, ..."
```

### launch_bounds for Predictable Allocation

```c
__global__ void __launch_bounds__(256, 2)  // 256 threads, min 2 blocks/SM
my_kernel(...) { ... }
```

- First arg: max threads per block (compiler target)
- Second arg: min blocks per SM (compiler tries to limit registers to fit this many blocks)
- With `__launch_bounds__(128, 3)` on sm_120: compiler targets `64K_regs / (3_blocks × 4_warps × 32_threads) = 170 regs/thread`

### Register Spills vs Occupancy

| Regs/Thread | Warps/SM (128 threads/block) | Blocks/SM | Notes |
|-------------|------|-----------|-------|
| 64 | 32 | 8 | Low per-thread, high occupancy |
| 80 | 24 | 6 | Good balance |
| 96 | 20 | 5 | Reasonable |
| 128 | 16 | 4 | Moderate — sweet spot per tuning guide |
| 145 | 12 | 3 | Our attention kernel — good for sm_120 |
| 170 | 8 | 2 | Getting tight |
| 255 | 6 | 1 | Maximum per-thread, minimum occupancy |

**Our current state:** 145 regs, 0 spills, 3 blocks/SM, 12 warps. This is a good operating point.

---

## 5. CUDA 13 Shared Memory Register Spilling

**Source:** https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling

CUDA 13.0 introduced opt-in register spilling to shared memory instead of L2-backed local memory.

### Enabling

```c
__global__ void kernel(...) {
    asm volatile (".pragma \"enable_smem_spilling\";");
    // ... rest of kernel ...
}
```

### What It Does

When register pressure exceeds the target, the compiler spills to shared memory (~10x lower latency than L2 local memory spills). The compiler automatically manages which values go to smem vs local.

### Limitations

- **Cannot be used with dynamic shared memory** (`extern __shared__`)
- Requires whole-program compilation (`-rdc=false`, the default)
- Incompatible with `-G` (debug mode)
- Must use `__launch_bounds__` for predictable behavior

### When to Use It

If you're at 0 spills and want to ADD more register-hungry optimizations (e.g., register double-buffer of fragments), enabling smem spilling provides a safety net: any marginal spills go to fast smem instead of slow L2.

**However:** For our GEMM kernel at 0.80x cuBLAS, the bottleneck is math_pipe_throttle (scheduling), not register spills. Smem spilling won't help the scheduling problem directly.

---

## 6. Controlling Compiler Register Usage via asm Patterns

### Technique: Force Register Reuse

Instead of letting the compiler allocate separate registers for each operand, reuse the same C variable:

```c
// Separate registers (compiler may allocate 8 regs for fragments)
unsigned a0, a1, a2, a3;
load_matrix_x4(&a0, &a1, &a2, &a3, addr);

// Reuse via array (compiler likely uses 4 regs)
unsigned a[4];
load_matrix_x4(a, addr);
```

### Technique: Union for In-Place Conversion

```c
// Convert FP32 accumulator to BF16 fragment in-place
union { float f; unsigned u; } conv;
conv.f = acc_val;
// BF16 pack: take upper 16 bits of FP32
unsigned bf16_packed = (conv.u >> 16) | ((conv2.u >> 16) << 16);
```

### Technique: Explicit Register Count via launch_bounds

If the compiler allocates too many registers (dropping occupancy), tighten launch_bounds:

```c
// Force compiler to target 128 regs/thread (fits 4 blocks/SM)
__global__ void __launch_bounds__(128, 4) kernel(...) { ... }
```

The compiler will aggressively spill to local memory to stay within this budget. With smem spilling enabled, those spills go to fast shared memory instead.

---

## 7. Large asm Block Best Practices

### String Concatenation for Readability

```c
asm volatile(
    "{\n"
    "  .reg .u32 a0, a1, a2, a3;\n"
    "  .reg .u32 b0, b1;\n"
    "  .reg .f32 d0, d1, d2, d3;\n"
    "\n"
    "  ldmatrix.sync.aligned.x4.m8n8.shared.b16 {a0,a1,a2,a3}, [%4];\n"
    "  ldmatrix.sync.aligned.x2.m8n8.shared.b16 {b0,b1}, [%5];\n"
    "\n"
    "  mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32\n"
    "    {d0,d1,d2,d3}, {a0,a2,a1,a3}, {b0,b1}, {%0,%1,%2,%3};\n"
    //                   ^^ a1/a2 swap for sm_120!
    "\n"
    "  mov.f32 %0, d0;\n"
    "  mov.f32 %1, d1;\n"
    "  mov.f32 %2, d2;\n"
    "  mov.f32 %3, d3;\n"
    "}\n"
    : "+f"(D[0]), "+f"(D[1]), "+f"(D[2]), "+f"(D[3])
    : "r"(smem_addr_a), "r"(smem_addr_b));
```

### Combining ldmatrix + MMA in One asm Block

By putting both the load and the MMA in a single asm block with named internal registers, you:
1. Eliminate the compiler's overhead of managing the intermediate registers
2. Ensure the registers used by ldmatrix are immediately consumed by MMA
3. Prevent the compiler from inserting spill code between load and use

**Tradeoff:** The compiler can no longer reorder the MMA relative to other code. Use this pattern only when you want EXPLICIT control over instruction ordering.

---

## References

- NVIDIA Inline PTX Assembly Guide (CUDA 13.2): https://docs.nvidia.com/cuda/inline-ptx-assembly/index.html
- NVIDIA Shared Memory Register Spilling Blog: https://developer.nvidia.com/blog/how-to-improve-cuda-kernel-performance-with-shared-memory-register-spilling
- PTX ISA 9.2 Reference: https://docs.nvidia.com/cuda/parallel-thread-execution/index.html
- NVIDIA Forum — PTX constraint letters: https://forums.developer.nvidia.com/t/ptx-asm-constraint-letter-for-predicate/283337
