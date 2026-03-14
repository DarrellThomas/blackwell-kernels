# Inline PTX Assembly Guide

**Source:** NVIDIA CUDA Inline PTX Assembly Guide, CUDA 13.2
**Scope:** Complete reference for writing inline PTX in CUDA C++ kernels.

---

## 1. Basic Syntax

### Non-Volatile asm()

```cpp
asm("instruction-string" : outputs : inputs);
```

The compiler may delete or reorder non-volatile asm statements during optimization if it determines the outputs are unused or the reordering is safe.

### Volatile asm()

```cpp
asm volatile("instruction-string" : outputs : inputs);
```

The `volatile` keyword prevents the compiler from deleting or moving the asm statement. Use when:
- The statement has side effects beyond modifying output operands (e.g., writing to shared memory)
- Instruction ordering must be preserved (e.g., barriers, memory fences)
- The statement reads/writes memory not tracked by operands

### Minimal Examples

```cpp
// No operands
asm("membar.gl;");
asm volatile("bar.sync 0;");

// One output
asm volatile("mov.u32 %0, %%clock;" : "=r"(x));

// Output + input
asm("add.s32 %0, %1, %2;" : "=r"(result) : "r"(a), "r"(b));
```

---

## 2. Constraint Letters

| Constraint | PTX Type | C++ Type | Width |
|-----------|----------|----------|-------|
| `"h"` | `.u16` | `uint16_t`, `short` | 16-bit |
| `"r"` | `.u32` | `uint32_t`, `int`, `unsigned` | 32-bit |
| `"l"` | `.u64` | `uint64_t`, `long long`, pointers | 64-bit |
| `"f"` | `.f32` | `float` | 32-bit |
| `"d"` | `.f64` | `double` | 64-bit |
| `"n"` | immediate | compile-time integer constant | N/A |
| `"C"` | string | `constexpr const char[]` | N/A |

**Important:** Only ONE constraint letter per operand. `"rf"` is invalid.

**No 8-bit register constraint exists.** For 8-bit PTX instructions, use 32-bit register constraint:
```cpp
int temp = char_val;
asm("ld.u8 %0, [%1];" : "=r"(temp) : "l"(ptr));
```

---

## 3. Output Operands

### Write-Only (`"="`)

Designates a register that is only written to:
```cpp
asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));
```

### Read-Write (`"+"`)

Designates a register that is both read and written. Required when the output is also used as input, or when the output is conditionally updated:

```cpp
// Increment in-place
asm("add.s32 %0, %0, %1;" : "+r"(i) : "r"(j));

// Conditional update (MUST use "+" since output may not be written)
int y = 0;
asm("{\n\t"
    " .reg .pred p;\n\t"
    " setp.eq.s32 p, %1, 34;\n\t"
    " @p mov.s32 %0, 1;\n\t"
    "}"
    : "+r"(y) : "r"(x));
```

**If you use `"="` for a conditionally-updated output, the compiler assumes the previous value is dead and may not initialize the register correctly.**

---

## 4. Input Operands

Input operands follow the second colon:
```cpp
asm("add.s32 %0, %1, %2;" : "=r"(out) : "r"(in1), "r"(in2));
```

**No output operands (adjacent colons):**
```cpp
asm("st.global.u32 [%0], %1;" :: "l"(ptr), "r"(value));
```

**No input operands (drop trailing colon):**
```cpp
asm("mov.s32 %0, 42;" : "=r"(i));
```

---

## 5. Operand Numbering

Operands are numbered sequentially: `%0` is the first output, `%1` is the second output (or first input if there's only one output), etc.

```cpp
// %0 = output, %1 = first input, %2 = second input
asm("add.s32 %0, %1, %2;" : "=r"(i) : "r"(j), "r"(k));

// Operands can be reused
asm("add.s32 %0, %1, %1;" : "=r"(i) : "r"(k));  // i = k + k

// Multiple outputs: %0=d0, %1=d1, %2=d2, %3=d3, %4=a0, %5=a1...
asm("mma.sync... {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
```

---

## 6. Clobber Lists

### Memory Clobber

Prevents the compiler from caching memory values across the asm statement:
```cpp
asm volatile("..." : outputs : inputs : "memory");
```

Use when:
- The asm block reads/writes memory through pointers not visible as operands
- You need to prevent memory optimizations across the statement

### Example

```cpp
// Store through pointer — compiler can't see the memory write via operands
asm("st.shared.u32 [%0], %1;" :: "r"(smem_addr), "r"(value) : "memory");
```

---

## 7. Register Declarations Inside asm Blocks

Use `.reg` to declare temporary registers within PTX:

```cpp
asm(".reg .u32 t1;\n\t"
    " mul.lo.u32 t1, %1, %1;\n\t"
    " mul.lo.u32 %0, t1, %1;"
    : "=r"(y) : "r"(x));
```

### Avoiding Namespace Conflicts

When a `__device__` function with inline asm is inlined multiple times, temp register names collide. **Always wrap in braces for local scope:**

```cpp
asm("{\n\t"
    " .reg .u32 t1;\n\t"
    " mul.lo.u32 t1, %1, %1;\n\t"
    " mul.lo.u32 %0, t1, %1;\n\t"
    "}"
    : "=r"(y) : "r"(x));
```

Similarly, use braces for local labels to avoid conflicts.

---

## 8. Multiple Instructions

Chain instructions using C string concatenation. Terminate each instruction with `\n\t` for readability:

```cpp
asm("{\n\t"
    " .reg .f32 tmp;\n\t"
    " add.f32 tmp, %1, %2;\n\t"
    " mul.f32 %0, tmp, %3;\n\t"
    "}"
    : "=f"(result) : "f"(a), "f"(b), "f"(c));
```

Both `//` and `/* */` comment styles work within asm strings.

---

## 9. Escaping Special Characters

Use `%%` to access PTX special registers (which use `%` prefix):

```cpp
asm volatile("mov.u32 %0, %%clock;" : "=r"(clock_val));
asm volatile("mov.u32 %0, %%laneid;" : "=r"(lane));
asm volatile("mov.u32 %0, %%tid.x;" : "=r"(tid));
```

---

## 10. Compile-Time String Constraint ("C")

Injects compile-time constant strings directly into PTX. Useful for template-parameterized instruction modifiers:

```cpp
template<> struct RoundMode<RN> {
    static constexpr const char mode[] = ".rn";
};
template<> struct RoundMode<RZ> {
    static constexpr const char mode[] = ".rz";
};

template <int rounding>
__device__ float convert(float a) {
    float result;
    asm("cvt%1.f32.f32 %0, %2;"
        : "=f"(result) : "C"(RoundMode<rounding>::mode), "f"(a));
    return result;
}
```

**Requirements:**
- Must evaluate to address of a `const char[]` with static storage duration
- Array must be compile-time initialized
- Device code only
- No constraint modifiers (no `"=C"` or `"+C"`)

---

## 11. Pointer and Address Handling

On sm_20+, all pointer arguments use generic (flat) addressing. The programmer must use correct memory space qualifiers in PTX:

```cpp
// Global memory access via pointer
asm("ld.global.b32 %0, [%1];" : "=r"(val) : "l"(global_ptr));

// Shared memory — convert generic to shared address first
uint32_t smem_addr = __cvta_generic_to_shared(shared_ptr);
asm("ld.shared.b32 %0, [%1];" : "=r"(val) : "r"(smem_addr));
```

**Key function:** `__cvta_generic_to_shared()` converts a 64-bit generic pointer to a 32-bit shared memory address (use `"r"` constraint for the result, not `"l"`).

---

## 12. Common Patterns for Kernel Development

### MMA with Non-Volatile (Compiler-Reorderable)

```cpp
// Non-volatile: compiler can interleave with other non-volatile ops
asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
    "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
    : "=f"(d0), "=f"(d1), "=f"(d2), "=f"(d3)
    : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
      "r"(b0), "r"(b1),
      "f"(c0), "f"(c1), "f"(c2), "f"(c3));
```

### ldmatrix (Must Be Volatile)

```cpp
// Volatile: shared memory read has side effects not captured by operands
uint32_t addr = __cvta_generic_to_shared(smem_ptr);
asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
    : "=r"(r0), "=r"(r1), "=r"(r2), "=r"(r3)
    : "r"(addr));
```

### cp.async (Must Be Volatile)

```cpp
// Volatile: asynchronous memory operation
uint32_t dst = __cvta_generic_to_shared(dst_smem);
asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n"
    :: "r"(dst), "l"(src_gmem));
```

### Predicated Execution

```cpp
// Only copy if predicate is true
asm volatile("{\n\t"
    " .reg .pred p;\n\t"
    " setp.ne.b32 p, %2, 0;\n\t"
    " @p cp.async.cg.shared.global [%0], [%1], 16;\n\t"
    "}"
    :: "r"(dst_smem), "l"(src_gmem), "r"((int)pred));
```

### BF16x2 Pack with Operand Swap

```cpp
// IMPORTANT: PTX reverses operand order vs C++
// PTX: first source -> HIGH bits, second source -> LOW bits
// To match C++ __floats2bfloat162_rn(lo, hi): swap operands
asm("cvt.rn.bf16x2.f32 %0, %2, %1;\n"  // Note: %2 before %1
    : "=r"(packed) : "f"(f_lo), "f"(f_hi));
```

### Warp Reduction via Shuffles

```cpp
asm("{\n\t"
    " .reg .f32 tmp;\n\t"
    " shfl.sync.bfly.b32 tmp, %0, 1, 31;\n\t"   // c=31, NOT 0x1f1f
    " add.f32 %0, %0, tmp;\n\t"
    " shfl.sync.bfly.b32 tmp, %0, 2, 31;\n\t"
    " add.f32 %0, %0, tmp;\n\t"
    "}"
    : "+f"(val));
```

---

## 13. Error Handling and Debugging

### Common Errors

**Multiple constraint letters:**
```cpp
asm("add.s32 %0, %1, %2;" : "=r"(i) : "rf"(j), "r"(k));
// ERROR: an asm operand may specify only one constraint letter
```

**Non-scalar operands:**
```cpp
int4 i4;
asm("add.s32 %0, %1, %2;" : "=r"(i4) : "r"(j), "r"(k));
// ERROR: an asm operand must have scalar type
```

**Size mismatch:**
```cpp
char ci;
asm("add.s32 %0, %1, %2;" : "=r"(ci) : "r"(j), "r"(k));
// ERROR: asm operand type size(1) does not match constraint 'r'
```

**Type mismatch:**
```cpp
float fi;
asm("add.s32 %0, %1, %2;" : "=r"(fi) : "r"(j), "r"(k));
// ERROR: "r" implies integer, not float — use "f" for floats
```

### Workarounds

```cpp
// For char/short: cast to matching size
int temp;
asm("add.s32 %0, %1, %2;" : "=r"(temp) : "r"((int)char_a), "r"((int)char_b));
char_result = (char)temp;
```

### Debugging Tips

1. **PTX syntax errors appear at ptxas stage, not at nvcc parse time.** The compiler does not parse the template string -- it just substitutes operands and passes to ptxas.

2. **Add `-ptxas-options=-v` to see register usage.** Spills indicate register pressure too high.

3. **Use `cuobjdump -ptx` to inspect generated PTX.** Verify operand substitution is correct.

4. **Use `cuobjdump -sass` to inspect final machine code.** The only ground truth for instruction scheduling.

---

## 14. volatile vs non-volatile Decision Guide

| Operation | Use volatile? | Reason |
|-----------|--------------|--------|
| `mma.sync` | **No** (use plain `asm`) | Compiler can reorder for better scheduling |
| `ldmatrix` | **Yes** | Reads shared memory not tracked by operands |
| `cp.async` | **Yes** | Asynchronous memory side effects |
| `bar.sync` | **Yes** | Synchronization must not be reordered |
| `cp.async.commit_group` | **Yes** | Controls async group boundaries |
| `cp.async.wait_group` | **Yes** | Must wait before accessing data |
| `membar` | **Yes** | Memory ordering fence |
| `shfl.sync` | Can be either | If used in reduction, non-volatile is fine |
| Pure arithmetic | **No** | Let compiler optimize freely |
| `st.shared` / `ld.shared` | **Yes** | Memory side effects |
| `mov %%clock` | **Yes** | Reading hardware counter |

**Rule of thumb:** Use `asm volatile` for anything that touches memory or synchronization. Use plain `asm` for pure computation (MMA, arithmetic) to give the compiler scheduling freedom.

---

## 15. Operand Limit Warning (sm_120 Specific)

**Monolithic asm blocks with more than ~50 output operands using `"+f"` constraint produce silently wrong results on nvcc/CUDA 13.** The threshold is between 34 (works) and 66 (broken). This is likely a ptxas register allocator limitation -- too many simultaneously-live in/out operands cause silent register misassignment.

The kernel compiles, runs, and may even appear faster (because it computes garbage with fewer dependencies), but produces incorrect output.

**Workaround:** Keep individual asm blocks under 40 operands. For larger PTX sequences, use the salykova approach: all registers PTX-managed with `.reg`, no C++ operand interface, addresses computed inside PTX from a single base pointer.

---

## 16. Complete Reference Example

```cpp
__device__ void example_mma_with_load(
    float *d, const void *smem_a, const void *smem_b, float *c)
{
    uint32_t a0, a1, a2, a3, b0, b1;

    // Load A fragment (volatile — shared memory read)
    uint32_t addr_a = __cvta_generic_to_shared(smem_a);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %2, %1, %3}, [%4];\n"
        : "=r"(a0), "=r"(a1), "=r"(a2), "=r"(a3)
        : "r"(addr_a));

    // Load B fragment (volatile — shared memory read)
    uint32_t addr_b = __cvta_generic_to_shared(smem_b);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b0), "=r"(b1)
        : "r"(addr_b));

    // MMA (non-volatile — let compiler schedule freely)
    asm("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0, %1, %2, %3}, "
        "{%4, %5, %6, %7}, "
        "{%8, %9}, "
        "{%10, %11, %12, %13};\n"
        : "=f"(d[0]), "=f"(d[1]), "=f"(d[2]), "=f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3),
          "r"(b0), "r"(b1),
          "f"(c[0]), "f"(c[1]), "f"(c[2]), "f"(c[3]));
}
```
