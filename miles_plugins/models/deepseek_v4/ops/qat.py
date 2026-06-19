import torch

from tile_kernels.quant import per_token_cast_back

from .kernel.act_quant import act_quant, fp4_act_quant


def fp8_simulate(x: torch.Tensor, block_size: int):
    """Simulate per-token FP8 (E4M3) cast + dequant with UE8M0 scaling.

    Both the cast (via :func:`act_quant`) and the cast-back step are routed
    through ``deepseek-ai/TileKernels`` so we share the same FP8 kernels with
    the rest of the DeepSeek stack.
    """
    x_c = x.contiguous()
    # Force fp32 scale storage: act_quant's auto dtype picks float8_e8m0fnu on
    # Blackwell+DeepGEMM, but per_token_cast_back below only accepts int32/fp32
    # scales. scale_fmt="ue8m0" still rounds the values to powers of two.
    y, scale = act_quant(x_c, block_size, "ue8m0", scale_dtype=torch.float32)

    N = x_c.size(-1)
    y_flat = y.view(-1, N)
    scale_flat = scale.reshape(y_flat.size(0), N // block_size).contiguous()

    out_flat = per_token_cast_back((y_flat, scale_flat), "bf16" if x.dtype == torch.bfloat16 else "fp32", block_size)
    return out_flat.view_as(x_c).to(x.dtype)


class DeepSeekV4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=128):
        return fp8_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


fp8_simulate_qat = DeepSeekV4LinearQATFunc.apply


def mxfp4_simulate(x: torch.Tensor, block_size: int = 32):
    """Simulate MXFP4 (E2M1) cast + dequant with ``1 x block_size`` UE8M0 (pow2) scaling.

    Mirrors :func:`fp8_simulate` but for FP4: a single fused on-device kernel
    (``fp4_act_quant`` with ``inplace=True``) rounds each value onto the E2M1 grid
    and dequantizes back to ``x.dtype``. We operate on ``x.clone()`` so the op is
    non-mutating (safe inside an autograd Function / straight-through estimator).
    """
    return fp4_act_quant(x.clone(), block_size, inplace=True).to(x.dtype)


class DeepSeekV4MXFP4LinearQATFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, kv, block_size=32):
        return mxfp4_simulate(kv, block_size)

    @staticmethod
    def backward(ctx, grad_kv):
        return grad_kv, None


mxfp4_simulate_qat = DeepSeekV4MXFP4LinearQATFunc.apply
