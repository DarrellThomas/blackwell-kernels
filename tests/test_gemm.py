"""Test BF16 GEMM kernel against PyTorch reference."""

import torch


def test_bf16_gemm():
    """Compare custom GEMM output against torch.mm reference."""
    torch.manual_seed(42)
    M, K, N = 256, 512, 256
    device = "cuda:0"

    A = torch.randn(M, K, device=device, dtype=torch.bfloat16)
    B = torch.randn(K, N, device=device, dtype=torch.bfloat16)

    ref = torch.mm(A.float(), B.float())

    from blackwell_kernels import bf16_gemm

    out = bf16_gemm(A, B)

    torch.testing.assert_close(out, ref, rtol=1e-3, atol=1e-3)
    print("PASS: bf16_gemm")


if __name__ == "__main__":
    test_bf16_gemm()
