"""DeepSeek-V4-Flash 4-layer smoke test for the mixed NVFP4 + blockwise-FP8
recipe (Blackwell only): NVFP4 routed experts (trainer TE recipe override,
rollout modelopt-FP4 via flashinfer_trtllm_routed with per-token activations,
NVFP4 weight-update quantizer), while shared experts and selected attention
projections remain blockwise FP8 and the explicit carve-outs remain BF16/F32.
"""

import os

from scripts.run_deepseek_v4 import (
    ScriptArgs,
    _prepare_download,
    _prepare_single,
    _prepare_spmd,
    _train,
)
from tests.ci.ci_register import register_cuda_ci

register_cuda_ci(est_time=2400, suite="stage-c-8-gpu-b200", labels=["model-scripts"])


def _args() -> ScriptArgs:
    return ScriptArgs(
        model_name="DeepSeek-V4-Flash-FP8-4layer",
        task="gsm8k",
        enable_eval=False,
        num_nodes=1,
        num_gpus_per_node=8,
        hardware="B200",
        nvfp4_experts=True,
        skip_saving=True,
        use_fault_tolerance=False,
        extra_args=(
            "--ci-test "
            "--check-weight-update-allow-quant-error "
            "--ci-disable-logprobs-checker "
            "--disable-weights-backuper "
            "--num-rollout 2 "
        ),
    )


def prepare(args: ScriptArgs):
    _prepare_download(args)
    _prepare_single(args)  # FP8 -> BF16 cast, then BF16 -> NVFP4+blockFP8
    _prepare_spmd(args)
    if args.hf_checkpoint is None:
        # Rollout serves the mixed NVFP4 + blockwise-FP8 checkpoint.
        args.hf_checkpoint = f"{args.model_local_dir}/{args.nvfp4_name}"


def execute(args: ScriptArgs):
    _train(args)


if __name__ == "__main__":
    args = _args()
    prepare(args)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(args)
