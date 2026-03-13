// Copyright (c) 2026 Darrell Thomas. MIT License. See LICENSE file.
//
// Hand-scheduled PTX: exp2f + sum + pack_P + load_V + PV_MMA + shuffle
// For D=64, BKV=64, WARP_Q_TILES=1 (the primary benchmark config).
//
// Schedule: interleave exp2f[kc+1] with MMA[kc] across kc boundaries.
// Two P register sets (pa/pb) enable overlap without WAR hazards.
// Sum shuffles deferred to kc=3 epilogue, overlapped with final MMA.

#pragma once
#include <cstdint>

// Operand map (84 total):
//   Outputs (+f): O[0..7][0..3] = %0-%31, rsum0=%32, rsum1=%33, S[0..7][0..3]=%34-%65
//   Inputs  (f):  nm0=%66, nm1=%67
//   Inputs  (r):  va[0][0..3]=%68-%71, va[1][0..3]=%72-%75, va[2][0..3]=%76-%79, va[3][0..3]=%80-%83

// Helper macros for readability
#define PTX_EX2(s, nm) \
    "sub.f32 " s ", " s ", " nm ";\n" \
    "ex2.approx.f32 " s ", " s ";\n"

#define PTX_MMA(d0,d1,d2,d3, a0,a1,a2,a3, b0,b1) \
    "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 " \
    "{" d0 "," d1 "," d2 "," d3 "}, " \
    "{" a0 "," a1 "," a2 "," a3 "}, " \
    "{" b0 "," b1 "}, " \
    "{" d0 "," d1 "," d2 "," d3 "};\n"

#define PTX_LDSM_TRANS(v, addr) \
    "ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {" v "}, [" addr "];\n"

// NOTE: cvt.rn.bf16x2.f32 d, a, b puts a→HIGH, b→LOW.
// To match pack_bf16x2(lo, hi) = __floats2bfloat162_rn(lo, hi),
// we swap: cvt d, hi, lo → d.hi=hi, d.lo=lo.
#define PTX_CVT_BF16X2(dst, lo, hi) \
    "cvt.rn.bf16x2.f32 " dst ", " hi ", " lo ";\n"

#define PTX_SHFL_XOR(dst, src, lane) \
    "shfl.sync.bfly.b32 " dst ", " src ", " #lane ", 31, 0xFFFFFFFF;\n"

__device__ __forceinline__ void ptx_fused_softmax_pv_d64(
    float (&O)[8][4],
    float (&S)[8][4],
    float &rsum0, float &rsum1,
    float new_max0, float new_max1,
    const uint32_t (&va)[4][4])
{
    asm volatile(
    "{\n"
    // Register declarations
    ".reg .b32 pa0, pa1, pa2, pa3;\n"   // P set A (kc=0,2)
    ".reg .b32 pb0, pb1, pb2, pb3;\n"   // P set B (kc=1,3)
    ".reg .b32 v0, v1, v2, v3;\n"       // V fragment
    ".reg .f32 sum0, sum1, st;\n"        // sum accumulators + shuffle temp
    "\n"
    "mov.f32 sum0, 0f00000000;\n"
    "mov.f32 sum1, 0f00000000;\n"
    "\n"

    // ============================================================
    // kc=0 PROLOGUE: exp2f S[0..1], sum, pack P_A, load V[0][0]
    // ============================================================
    PTX_EX2("%34", "%66")   // S[0][0] = exp2f(S[0][0] - nm0)
    PTX_EX2("%35", "%66")   // S[0][1]
    PTX_EX2("%36", "%67")   // S[0][2] (row1, uses nm1)
    PTX_EX2("%37", "%67")   // S[0][3]
    PTX_EX2("%38", "%66")   // S[1][0]
    PTX_EX2("%39", "%66")   // S[1][1]
    PTX_EX2("%40", "%67")   // S[1][2]
    PTX_EX2("%41", "%67")   // S[1][3]
    // sum for kc=0
    "add.f32 sum0, %34, %35;\n"
    "add.f32 sum1, %36, %37;\n"
    "add.f32 sum0, sum0, %38;\n"
    "add.f32 sum0, sum0, %39;\n"
    "add.f32 sum1, sum1, %40;\n"
    "add.f32 sum1, sum1, %41;\n"
    // pack P_A from S[0..1]
    PTX_CVT_BF16X2("pa0", "%34", "%35")  // pa0 = {S[0][0], S[0][1]}
    PTX_CVT_BF16X2("pa1", "%36", "%37")  // pa1 = {S[0][2], S[0][3]}
    PTX_CVT_BF16X2("pa2", "%38", "%39")  // pa2 = {S[1][0], S[1][1]}
    PTX_CVT_BF16X2("pa3", "%40", "%41")  // pa3 = {S[1][2], S[1][3]}
    // load V[kc=0][nc_pair=0]
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%68")
    // MMA O[0..1] += P_A * V[0][0]
    PTX_MMA("%0","%1","%2","%3",     "pa0","pa1","pa2","pa3", "v0","v1")
    PTX_MMA("%4","%5","%6","%7",     "pa0","pa1","pa2","pa3", "v2","v3")
    "\n"

    // ============================================================
    // kc=0 CONT + kc=1 EXP2F: interleave exp2f[S2..3] with MMA[kc=0]
    // ============================================================
    // exp2f S[2] (kc=1, first half) — overlaps with MMA pipeline
    PTX_EX2("%42", "%66")   // S[2][0]
    PTX_EX2("%43", "%66")   // S[2][1]
    PTX_EX2("%44", "%67")   // S[2][2]
    PTX_EX2("%45", "%67")   // S[2][3]
    // load V[kc=0][nc_pair=1], MMA O[2..3]
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%69")
    PTX_MMA("%8","%9","%10","%11",   "pa0","pa1","pa2","pa3", "v0","v1")
    PTX_MMA("%12","%13","%14","%15", "pa0","pa1","pa2","pa3", "v2","v3")
    // exp2f S[3] (kc=1, second half)
    PTX_EX2("%46", "%66")   // S[3][0]
    PTX_EX2("%47", "%66")   // S[3][1]
    PTX_EX2("%48", "%67")   // S[3][2]
    PTX_EX2("%49", "%67")   // S[3][3]
    // sum for kc=1
    "add.f32 sum0, sum0, %42;\n"
    "add.f32 sum0, sum0, %43;\n"
    "add.f32 sum1, sum1, %44;\n"
    "add.f32 sum1, sum1, %45;\n"
    "add.f32 sum0, sum0, %46;\n"
    "add.f32 sum0, sum0, %47;\n"
    "add.f32 sum1, sum1, %48;\n"
    "add.f32 sum1, sum1, %49;\n"
    // pack P_B from S[2..3]
    PTX_CVT_BF16X2("pb0", "%42", "%43")
    PTX_CVT_BF16X2("pb1", "%44", "%45")
    PTX_CVT_BF16X2("pb2", "%46", "%47")
    PTX_CVT_BF16X2("pb3", "%48", "%49")
    // load V[kc=0][nc_pair=2], MMA O[4..5]
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%70")
    PTX_MMA("%16","%17","%18","%19", "pa0","pa1","pa2","pa3", "v0","v1")
    PTX_MMA("%20","%21","%22","%23", "pa0","pa1","pa2","pa3", "v2","v3")
    // load V[kc=0][nc_pair=3], MMA O[6..7] — kc=0 done
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%71")
    PTX_MMA("%24","%25","%26","%27", "pa0","pa1","pa2","pa3", "v0","v1")
    PTX_MMA("%28","%29","%30","%31", "pa0","pa1","pa2","pa3", "v2","v3")
    "\n"

    // ============================================================
    // kc=1 MMA + kc=2 EXP2F
    // ============================================================
    // load V[kc=1][nc_pair=0], MMA O[0..1] with P_B
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%72")
    PTX_MMA("%0","%1","%2","%3",     "pb0","pb1","pb2","pb3", "v0","v1")
    PTX_MMA("%4","%5","%6","%7",     "pb0","pb1","pb2","pb3", "v2","v3")
    // exp2f S[4] (kc=2, first half)
    PTX_EX2("%50", "%66")
    PTX_EX2("%51", "%66")
    PTX_EX2("%52", "%67")
    PTX_EX2("%53", "%67")
    // load V[kc=1][nc_pair=1], MMA O[2..3]
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%73")
    PTX_MMA("%8","%9","%10","%11",   "pb0","pb1","pb2","pb3", "v0","v1")
    PTX_MMA("%12","%13","%14","%15", "pb0","pb1","pb2","pb3", "v2","v3")
    // exp2f S[5] (kc=2, second half)
    PTX_EX2("%54", "%66")
    PTX_EX2("%55", "%66")
    PTX_EX2("%56", "%67")
    PTX_EX2("%57", "%67")
    // sum for kc=2
    "add.f32 sum0, sum0, %50;\n"
    "add.f32 sum0, sum0, %51;\n"
    "add.f32 sum1, sum1, %52;\n"
    "add.f32 sum1, sum1, %53;\n"
    "add.f32 sum0, sum0, %54;\n"
    "add.f32 sum0, sum0, %55;\n"
    "add.f32 sum1, sum1, %56;\n"
    "add.f32 sum1, sum1, %57;\n"
    // pack P_A from S[4..5] (reuse P_A, kc=0 done)
    PTX_CVT_BF16X2("pa0", "%50", "%51")
    PTX_CVT_BF16X2("pa1", "%52", "%53")
    PTX_CVT_BF16X2("pa2", "%54", "%55")
    PTX_CVT_BF16X2("pa3", "%56", "%57")
    // load V[kc=1][nc_pair=2..3], MMA O[4..7]
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%74")
    PTX_MMA("%16","%17","%18","%19", "pb0","pb1","pb2","pb3", "v0","v1")
    PTX_MMA("%20","%21","%22","%23", "pb0","pb1","pb2","pb3", "v2","v3")
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%75")
    PTX_MMA("%24","%25","%26","%27", "pb0","pb1","pb2","pb3", "v0","v1")
    PTX_MMA("%28","%29","%30","%31", "pb0","pb1","pb2","pb3", "v2","v3")
    "\n"

    // ============================================================
    // kc=2 MMA + kc=3 EXP2F
    // ============================================================
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%76")
    PTX_MMA("%0","%1","%2","%3",     "pa0","pa1","pa2","pa3", "v0","v1")
    PTX_MMA("%4","%5","%6","%7",     "pa0","pa1","pa2","pa3", "v2","v3")
    // exp2f S[6]
    PTX_EX2("%58", "%66")
    PTX_EX2("%59", "%66")
    PTX_EX2("%60", "%67")
    PTX_EX2("%61", "%67")
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%77")
    PTX_MMA("%8","%9","%10","%11",   "pa0","pa1","pa2","pa3", "v0","v1")
    PTX_MMA("%12","%13","%14","%15", "pa0","pa1","pa2","pa3", "v2","v3")
    // exp2f S[7]
    PTX_EX2("%62", "%66")
    PTX_EX2("%63", "%66")
    PTX_EX2("%64", "%67")
    PTX_EX2("%65", "%67")
    // sum for kc=3
    "add.f32 sum0, sum0, %58;\n"
    "add.f32 sum0, sum0, %59;\n"
    "add.f32 sum1, sum1, %60;\n"
    "add.f32 sum1, sum1, %61;\n"
    "add.f32 sum0, sum0, %62;\n"
    "add.f32 sum0, sum0, %63;\n"
    "add.f32 sum1, sum1, %64;\n"
    "add.f32 sum1, sum1, %65;\n"
    // pack P_B from S[6..7]
    PTX_CVT_BF16X2("pb0", "%58", "%59")
    PTX_CVT_BF16X2("pb1", "%60", "%61")
    PTX_CVT_BF16X2("pb2", "%62", "%63")
    PTX_CVT_BF16X2("pb3", "%64", "%65")
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%78")
    PTX_MMA("%16","%17","%18","%19", "pa0","pa1","pa2","pa3", "v0","v1")
    PTX_MMA("%20","%21","%22","%23", "pa0","pa1","pa2","pa3", "v2","v3")
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%79")
    PTX_MMA("%24","%25","%26","%27", "pa0","pa1","pa2","pa3", "v0","v1")
    PTX_MMA("%28","%29","%30","%31", "pa0","pa1","pa2","pa3", "v2","v3")
    "\n"

    // ============================================================
    // kc=3 MMA + DEFERRED SUM SHUFFLE
    // ============================================================
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%80")
    PTX_MMA("%0","%1","%2","%3",     "pb0","pb1","pb2","pb3", "v0","v1")
    PTX_MMA("%4","%5","%6","%7",     "pb0","pb1","pb2","pb3", "v2","v3")
    // shuffle sum0 (overlaps with MMA pipeline)
    PTX_SHFL_XOR("st", "sum0", 1)
    "add.f32 sum0, sum0, st;\n"
    PTX_SHFL_XOR("st", "sum0", 2)
    "add.f32 sum0, sum0, st;\n"
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%81")
    PTX_MMA("%8","%9","%10","%11",   "pb0","pb1","pb2","pb3", "v0","v1")
    PTX_MMA("%12","%13","%14","%15", "pb0","pb1","pb2","pb3", "v2","v3")
    // shuffle sum1
    PTX_SHFL_XOR("st", "sum1", 1)
    "add.f32 sum1, sum1, st;\n"
    PTX_SHFL_XOR("st", "sum1", 2)
    "add.f32 sum1, sum1, st;\n"
    // update rsum
    "add.f32 %32, %32, sum0;\n"
    "add.f32 %33, %33, sum1;\n"
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%82")
    PTX_MMA("%16","%17","%18","%19", "pb0","pb1","pb2","pb3", "v0","v1")
    PTX_MMA("%20","%21","%22","%23", "pb0","pb1","pb2","pb3", "v2","v3")
    PTX_LDSM_TRANS("v0,v1,v2,v3", "%83")
    PTX_MMA("%24","%25","%26","%27", "pb0","pb1","pb2","pb3", "v0","v1")
    PTX_MMA("%28","%29","%30","%31", "pb0","pb1","pb2","pb3", "v2","v3")

    "}\n"

    // === OPERAND LIST ===
    // Outputs: O[0..7][0..3] (%0-%31), rsum (%32-%33), S[0..7][0..3] (%34-%65)
    : "+f"(O[0][0]), "+f"(O[0][1]), "+f"(O[0][2]), "+f"(O[0][3]),   // %0-3
      "+f"(O[1][0]), "+f"(O[1][1]), "+f"(O[1][2]), "+f"(O[1][3]),   // %4-7
      "+f"(O[2][0]), "+f"(O[2][1]), "+f"(O[2][2]), "+f"(O[2][3]),   // %8-11
      "+f"(O[3][0]), "+f"(O[3][1]), "+f"(O[3][2]), "+f"(O[3][3]),   // %12-15
      "+f"(O[4][0]), "+f"(O[4][1]), "+f"(O[4][2]), "+f"(O[4][3]),   // %16-19
      "+f"(O[5][0]), "+f"(O[5][1]), "+f"(O[5][2]), "+f"(O[5][3]),   // %20-23
      "+f"(O[6][0]), "+f"(O[6][1]), "+f"(O[6][2]), "+f"(O[6][3]),   // %24-27
      "+f"(O[7][0]), "+f"(O[7][1]), "+f"(O[7][2]), "+f"(O[7][3]),   // %28-31
      "+f"(rsum0),                                                    // %32
      "+f"(rsum1),                                                    // %33
      "+f"(S[0][0]), "+f"(S[0][1]), "+f"(S[0][2]), "+f"(S[0][3]),   // %34-37
      "+f"(S[1][0]), "+f"(S[1][1]), "+f"(S[1][2]), "+f"(S[1][3]),   // %38-41
      "+f"(S[2][0]), "+f"(S[2][1]), "+f"(S[2][2]), "+f"(S[2][3]),   // %42-45
      "+f"(S[3][0]), "+f"(S[3][1]), "+f"(S[3][2]), "+f"(S[3][3]),   // %46-49
      "+f"(S[4][0]), "+f"(S[4][1]), "+f"(S[4][2]), "+f"(S[4][3]),   // %50-53
      "+f"(S[5][0]), "+f"(S[5][1]), "+f"(S[5][2]), "+f"(S[5][3]),   // %54-57
      "+f"(S[6][0]), "+f"(S[6][1]), "+f"(S[6][2]), "+f"(S[6][3]),   // %58-61
      "+f"(S[7][0]), "+f"(S[7][1]), "+f"(S[7][2]), "+f"(S[7][3])    // %62-65
    // Inputs: new_max (%66-%67), V addresses (%68-%83)
    : "f"(new_max0),                                                  // %66
      "f"(new_max1),                                                  // %67
      "r"(va[0][0]), "r"(va[0][1]), "r"(va[0][2]), "r"(va[0][3]),   // %68-71
      "r"(va[1][0]), "r"(va[1][1]), "r"(va[1][2]), "r"(va[1][3]),   // %72-75
      "r"(va[2][0]), "r"(va[2][1]), "r"(va[2][2]), "r"(va[2][3]),   // %76-79
      "r"(va[3][0]), "r"(va[3][1]), "r"(va[3][2]), "r"(va[3][3])    // %80-83
    );
}

#undef PTX_EX2
#undef PTX_MMA
#undef PTX_LDSM_TRANS
#undef PTX_CVT_BF16X2
#undef PTX_SHFL_XOR
