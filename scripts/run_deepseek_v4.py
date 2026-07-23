"""
DeepSeek V4 training script.

Supports:
  - DeepSeek-V4-Flash-FP8         Public FP8 repackage of deepseek-ai/DeepSeek-V4-Flash
                                  (sgl-project/DeepSeek-V4-Flash-FP8, 291B, 43 layers).
                                  Verified full-model profiles: 8 nodes x 8 GPUs on H200
                                  or 8 nodes x 4 GPUs on GB300.
  - DeepSeek-V4-Flash-FP8-4layer  4-layer prune of the above for single-node
                                  smoke testing. **Cannot generate meaningful output -
                                  pipeline-only sanity check.**
  - DeepSeek-V4-Pro-FP8           Verified profile: 32 nodes x 8 GPUs on H200.

Usage patterns:

  1. One-shot full pipeline (download + convert + train):
       python scripts/run_deepseek_v4.py full-train \
           --model-name DeepSeek-V4-Flash-FP8-4layer \
           --num-nodes 1 --num-gpus-per-node 8

  2. Individual steps (download -> FP8->BF16 -> BF16->torch_dist -> rsync -> train):
       python scripts/run_deepseek_v4.py prepare-download --model-name DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-single   --model-name DeepSeek-V4-Flash-FP8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py prepare-spmd     --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 8 --num-gpus-per-node 8
       python scripts/run_deepseek_v4.py prepare-cp       --model-name DeepSeek-V4-Flash-FP8
       python scripts/run_deepseek_v4.py train            --model-name DeepSeek-V4-Flash-FP8 \
           --num-nodes 8 --num-gpus-per-node 8 \
           --hf-checkpoint /root/models/DeepSeek-V4-Flash-FP8

  3. NVFP4 routed experts (Blackwell only). Adds a BF16 -> NVFP4+blockFP8 rollout
     checkpoint conversion in prepare; the trainer overrides routed-expert GEMMs
     to the NVFP4 recipe; weight updates quantize experts to NVFP4:
       python scripts/run_deepseek_v4.py full-train \
           --model-name DeepSeek-V4-Flash-FP8-4layer \
           --num-nodes 1 --num-gpus-per-node 8 --nvfp4-experts
     With explicit `train`, point --hf-checkpoint at the converted directory
     ({model_dir}/{model_name}-NVFP4).
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import typer

import miles.utils.external_utils.command_utils as U

app = typer.Typer()

_DEFAULT_MODEL_ORG = {
    "DeepSeek-V4-Flash-FP8": "sgl-project",
    # 4-layer prune of sgl-project/DeepSeek-V4-Flash-FP8.
    "DeepSeek-V4-Flash-FP8-4layer": "Pinaster",
    "DeepSeek-V4-Pro-FP8": "sgl-project",
}

_MEGATRON_MODEL_TYPE = {
    "DeepSeek-V4-Flash-FP8": "deepseek-v4-flash",
    "DeepSeek-V4-Flash-FP8-4layer": "deepseek-v4-flash-4layer",
    "DeepSeek-V4-Pro-FP8": "deepseek-v4-pro",
}

_PRO_MODEL_NAMES = ("DeepSeek-V4-Pro-FP8",)
_BLACKWELL_HARDWARE = ("B200", "B300", "GB200", "GB300")

_DSV4_FP8_TE_PRECISION_CONFIG = """
configs:
  bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe: {}
matchers:
  dsa_indexer_weights_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.indexer.linear_weights_proj"
    config: "bf16"
""".strip()

# NVFP4-experts variant: the global recipe stays blockwise FP8 (attention +
# shared experts, matching the rollout's blockfp8 layers); the routed-expert
# grouped GEMMs are overridden to the NVFP4 recipe (matching the rollout's
# modelopt-FP4 experts); the DSA indexer weights_proj stays BF16.
# Must stay aligned with tools/convert_hf_to_nvfp4_blockfp8.py and
# processors/quantizer_nvfp4_blockfp8.py.
_DSV4_NVFP4_TE_PRECISION_CONFIG = """
configs:
  bf16:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe: {}
  blockfp8:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe:
      fp8_quantization_recipe: "blockwise"
  nvfp4:
    transformer_engine_config_type: "TEQuantizationParams"
    training_recipe:
      fp4_quantization_recipe: "nvfp4"
matchers:
  dsa_indexer_weights_proj_bf16:
    type: "glob"
    enabled: true
    pattern: "*.self_attention.indexer.linear_weights_proj"
    config: "bf16"
  routed_experts_fc1_nvfp4:
    type: "glob"
    enabled: true
    pattern: "*.mlp.experts.linear_fc1"
    config: "nvfp4"
  routed_experts_fc2_nvfp4:
    type: "glob"
    enabled: true
    pattern: "*.mlp.experts.linear_fc2"
    config: "nvfp4"
  shared_experts_fc1_blockfp8:
    type: "glob"
    enabled: true
    pattern: "*.mlp.shared_experts.linear_fc1"
    config: "blockfp8"
  shared_experts_fc2_blockfp8:
    type: "glob"
    enabled: true
    pattern: "*.mlp.shared_experts.linear_fc2"
    config: "blockfp8"
""".strip()

# Trainer-side NVFP4 knobs for the RL-aligned recipe: 1D (1x16) weight blocks,
# no RHT, no stochastic rounding, row-scaled (per-token) activations, exact
# math — mirroring the rollout side's per-token FlashInfer quantization
# (SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION=1 + disable-fast-math).
_NVFP4_TRAIN_ENV_VARS = {
    "NVTE_NVFP4_DISABLE_2D_QUANTIZATION": "1",
    "NVTE_NVFP4_DISABLE_RHT": "1",
    "NVTE_NVFP4_DISABLE_STOCHASTIC_ROUNDING": "1",
    "NVTE_NVFP4_ROW_SCALED_ACTIVATION": "1",
    "NVTE_BACKWARD_OVERRIDE": "dequantized",
    "NVTE_USE_FAST_MATH": "0",
}


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    mode: Literal["normal", "debug_minimal"] = "debug_minimal"
    run_id: str = U.create_run_id()
    model_org: str = ""
    model_name: Literal[
        "DeepSeek-V4-Flash-FP8",
        "DeepSeek-V4-Flash-FP8-4layer",
        "DeepSeek-V4-Pro-FP8",
    ] = "DeepSeek-V4-Flash-FP8"

    task: Literal["dapo_aime", "gsm8k"] = "dapo_aime"
    enable_eval: bool = True

    hf_checkpoint: str | None = None
    data_dir: str = "/root/datasets"
    model_dir: str = "/root/models"
    # Defaults to model_dir. Set explicitly when shared NFS -> per-node local NVMe copy is needed.
    model_local_dir: str | None = None
    save_dir: str = "/root/models"
    megatron_path: str = "/root/Megatron-LM"

    # performance configs
    num_gpus_per_node: int = 8
    hardware: Literal["auto", "H100", "H200", "B200", "B300", "GB200", "GB300"] = "auto"
    # use colocate by default. will switch to disaggregated mode when 0 < rollout_num_nodes < num_nodes
    rollout_num_nodes: int = 0
    colocate: bool = field(init=False)
    actor_num_nodes: int = field(init=False)
    actor_num_gpus_per_node: int = field(init=False)
    rollout_num_gpus: int = field(init=False)
    enable_mtp: bool = False
    optimizer_offload: bool = True
    use_fault_tolerance: bool = True
    cp_size: int = 1

    # debug configs
    dump_details: bool = False
    debug_train_run_id: str | None = None
    debug_train_rollout_id: str | None = None
    debug_data_root: str = "/root/shared_data"
    skip_saving: bool = False

    # precision configs
    enable_r3: bool = True
    train_deterministic: bool = True
    # Megatron-side training precision: blockwise FP8 128x128 GEMMs when True
    # (Hopper: fp32 scales; Blackwell: pow2 scales, MXFP8-emulated), BF16 when False.
    # Rollout always serves the source FP8 checkpoint either way.
    fp8_training: bool = True
    # NVFP4 routed experts (Blackwell only): trainer runs routed-expert GEMMs in
    # NVFP4 (row-scaled activations, 1x16 weight blocks) on top of the blockwise
    # FP8 recipe; rollout serves the mixed NVFP4+blockFP8 checkpoint produced by
    # tools/convert_hf_to_nvfp4_blockfp8.py with flashinfer_trtllm_routed MoE and
    # per-token FP4 activations; weight updates quantize routed experts to NVFP4
    # and everything else to blockwise FP8 (quantizer_nvfp4_blockfp8).
    nvfp4_experts: bool = False
    enable_mis: bool = False

    # pass any extra sglang/miles/megatron args through `--extra-args '--your-arg'`
    extra_args: str = ""

    def __post_init__(self):
        if not self.model_org:
            self.model_org = _DEFAULT_MODEL_ORG[self.model_name]
        if self.model_local_dir is None:
            self.model_local_dir = self.model_dir
        if self.model_name in _PRO_MODEL_NAMES:
            self.enable_r3 = False
        if self.nvfp4_experts:
            assert self.fp8_training, "nvfp4_experts needs fp8_training (blockwise FP8 is the base recipe)"
        assert self.rollout_num_nodes >= 0
        assert self.rollout_num_nodes < self.num_nodes
        self.colocate = self.rollout_num_nodes == 0
        self.actor_num_nodes = self.num_nodes - self.rollout_num_nodes
        self.actor_num_gpus_per_node = self.num_gpus_per_node
        if self.colocate:
            self.rollout_num_gpus = self.num_nodes * self.num_gpus_per_node
        else:
            self.rollout_num_gpus = self.rollout_num_nodes * self.num_gpus_per_node

    @property
    def megatron_model_type(self):
        return _MEGATRON_MODEL_TYPE[self.model_name]

    @property
    def torch_dist_name(self):
        return f"{self.model_name}_torch_dist"

    @property
    def bf16_name(self):
        if self.model_name == "DeepSeek-V4-Pro-FP8":
            return "DeepSeek-V4-Pro-BF16"
        return f"{self.model_name}-bf16"

    @property
    def nvfp4_name(self):
        return f"{self.model_name}-NVFP4"


def _is_blackwell(args: ScriptArgs) -> bool:
    if args.hardware != "auto":
        return args.hardware in _BLACKWELL_HARDWARE

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("Cannot auto-detect hardware because CUDA is not available. Pass --hardware explicitly.")
    major, _minor = torch.cuda.get_device_capability()
    return major >= 10


def _download_dataset(args: ScriptArgs):
    """Download the task-specific dataset(s)."""
    match args.task:
        case "dapo_aime":
            U.hf_download_dataset("zhuzilin/dapo-math-17k", data_dir=args.data_dir)
            U.hf_download_dataset("zhuzilin/aime-2024", data_dir=args.data_dir)
        case "gsm8k":
            U.hf_download_dataset("zhuzilin/gsm8k", data_dir=args.data_dir)


def _hf_checkpoint_path(args: ScriptArgs) -> str:
    """Resolve hf_checkpoint path: explicit override wins, else {model_dir}/{model_name}."""
    return args.hf_checkpoint or f"{args.model_dir}/{args.model_name}"


def _ensure_4layer_model_type(args: ScriptArgs):
    """Undo the old deepseek_ref workaround for local 4-layer prunes."""
    if args.model_name != "DeepSeek-V4-Flash-FP8-4layer":
        return
    cfg = Path(_hf_checkpoint_path(args)) / "config.json"
    if not cfg.exists():
        return
    text = cfg.read_text()
    if '"model_type": "deepseek_ref"' in text:
        cfg.write_text(text.replace('"model_type": "deepseek_ref"', '"model_type": "deepseek_v4"'))
        print(f"[patch] {cfg}: model_type deepseek_ref -> deepseek_v4")


def _prepare_download(args: ScriptArgs):
    """Download HF checkpoint + task dataset. Idempotent: hf skips existing blobs."""
    U.exec_command(f"mkdir -p {args.model_dir} {args.data_dir}")
    # Only download if the user has NOT supplied a pre-existing checkpoint dir.
    # (prepare_single / train with --hf-checkpoint bypass this.)
    if args.hf_checkpoint is None:
        dest = f"{args.model_dir}/{args.model_name}"
        U.exec_command(f"hf download {args.model_org}/{args.model_name} " f"--local-dir {dest}")
    _ensure_4layer_model_type(args)
    _download_dataset(args)


@app.command()
@U.dataclass_cli
def prepare_download(args: ScriptArgs):
    """Download HF checkpoint + dataset from HuggingFace. Run on one node (shared NFS)."""
    _prepare_download(args)


def _convert_nvfp4_blockfp8(args: ScriptArgs):
    """BF16 -> mixed NVFP4 (routed experts) + blockwise-FP8 rollout checkpoint."""
    path_dst = f"{args.model_dir}/{args.nvfp4_name}"
    sentinel = Path(path_dst) / "model.safetensors.index.json"
    if sentinel.exists():
        print(f"convert_hf_to_nvfp4_blockfp8 skip {path_dst} since {sentinel} exists")
        return

    # NVTE_USE_FAST_MATH=0 pins TE's exact-math quantization (its default), so
    # cold-start NVFP4 bytes match the weight-update quantizer bit-for-bit.
    U.exec_command(
        "NVTE_USE_FAST_MATH=0 "
        f"python {U.repo_base_dir}/tools/convert_hf_to_nvfp4_blockfp8.py "
        f"--model-dir {args.model_dir}/{args.bf16_name} "
        f"--save-dir {path_dst} "
    )


def _prepare_single(args: ScriptArgs):
    _download_dataset(args)

    src = _hf_checkpoint_path(args)
    U.fp8_cast_bf16(
        path_src=src,
        path_dst=f"{args.model_dir}/{args.bf16_name}/",
    )
    if args.nvfp4_experts:
        _convert_nvfp4_blockfp8(args)


@app.command()
@U.dataclass_cli
def prepare_single(args: ScriptArgs):
    """FP8 -> BF16 cast for Megatron (+ NVFP4 rollout ckpt with --nvfp4-experts). One node."""
    _prepare_single(args)


def _prepare_spmd(args: ScriptArgs):
    is_4layer = args.model_name == "DeepSeek-V4-Flash-FP8-4layer"
    actor_num_nodes = args.actor_num_nodes
    actor_num_gpus_per_node = args.actor_num_gpus_per_node
    extra_args = "--expert-tensor-parallel-size 1 --context-parallel-size 1 "
    if actor_num_nodes == 1 and is_4layer:
        extra_args += (
            "--tensor-model-parallel-size 1 " "--pipeline-model-parallel-size 1 " "--expert-model-parallel-size 1 "
        )
    elif actor_num_nodes == 8 and args.model_name == "DeepSeek-V4-Flash-FP8":
        extra_args += (
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 4 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
        )
    elif actor_num_nodes == 32 and actor_num_gpus_per_node == 8 and args.model_name == "DeepSeek-V4-Pro-FP8":
        extra_args += (
            "--tensor-model-parallel-size 8 "
            "--pipeline-model-parallel-size 8 "
            "--expert-model-parallel-size 32 "
            "--decoder-first-pipeline-num-layers 7 "
            "--decoder-last-pipeline-num-layers 6 "
            "--make-vocab-size-divisible-by 32 "
        )
    else:
        raise NotImplementedError(
            f"No verified SPMD conversion config for {args.model_name} "
            f"({actor_num_nodes} actor nodes x {actor_num_gpus_per_node} GPUs/node). "
            f"Please specify your conversion parallel config in `run_deepseek_v4.py`."
        )

    num_gpus_for_convert = actor_num_gpus_per_node
    if is_4layer:
        num_gpus_for_convert = min(num_gpus_for_convert, 4)

    U.convert_checkpoint(
        model_name=args.model_name,
        hf_checkpoint=f"{args.model_dir}/{args.bf16_name}",
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=num_gpus_for_convert,
        multinode=True if actor_num_nodes > 1 else False,
        num_nodes=actor_num_nodes,
        extra_args=extra_args,
        dir_dst=f"{args.model_dir}",
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def prepare_spmd(args: ScriptArgs):
    _prepare_spmd(args)


@app.command()
@U.dataclass_cli
def prepare_cp(args: ScriptArgs):
    _prepare_cp(args)


def _prepare_cp(args: ScriptArgs):
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.torch_dist_name}",
        path_dst=f"{args.model_local_dir}/{args.torch_dist_name}",
        num_nodes=args.num_nodes,
    )
    U.rsync_simple(
        path_src=f"{args.model_dir}/{args.model_name}",
        path_dst=f"{args.model_local_dir}/{args.model_name}",
        num_nodes=args.num_nodes,
    )
    if args.nvfp4_experts:
        U.rsync_simple(
            path_src=f"{args.model_dir}/{args.nvfp4_name}",
            path_dst=f"{args.model_local_dir}/{args.nvfp4_name}",
            num_nodes=args.num_nodes,
        )


def _get_parallel_config(args: ScriptArgs) -> str:
    """Return parallel config args for tested GPU configurations.

    Only includes configurations that have been verified to work.
    Raises NotImplementedError for untested configurations.
    """
    actor_num_nodes = args.actor_num_nodes
    actor_num_gpus_per_node = args.actor_num_gpus_per_node
    total_gpus = actor_num_nodes * actor_num_gpus_per_node

    # Single-node smoke-test configs
    if actor_num_nodes == 1:
        return (
            f"--tensor-model-parallel-size {actor_num_gpus_per_node} "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--expert-model-parallel-size {actor_num_gpus_per_node} "
            "--expert-tensor-parallel-size 1 "
        )

    # GB300: 4 GPUs/node
    if actor_num_gpus_per_node == 4:
        if total_gpus == 32:  # 8 nodes x 4 GPUs
            return (
                "--tensor-model-parallel-size 2 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 8 "
                "--decoder-first-pipeline-num-layers 4 "
                "--decoder-last-pipeline-num-layers 3 "
                "--context-parallel-size 2 "
                "--expert-model-parallel-size 4 "
                "--expert-tensor-parallel-size 1 "
            )

    # H200: 8 GPUs/node
    if actor_num_gpus_per_node == 8:
        if total_gpus == 64:  # 8 nodes x 8 GPUs
            return (
                "--tensor-model-parallel-size 8 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 8 "
                "--decoder-first-pipeline-num-layers 4 "
                "--decoder-last-pipeline-num-layers 3 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 8 "
                "--expert-tensor-parallel-size 1 "
            )
        elif total_gpus == 256:  # 32 nodes x 8 GPUs (Pro)
            return (
                "--tensor-model-parallel-size 8 "
                "--sequence-parallel "
                "--pipeline-model-parallel-size 8 "
                "--decoder-first-pipeline-num-layers 7 "
                "--decoder-last-pipeline-num-layers 6 "
                "--context-parallel-size 1 "
                "--expert-model-parallel-size 32 "
                "--expert-tensor-parallel-size 1 "
            )

    raise NotImplementedError(
        f"No pre-set parallel config for {total_gpus} GPUs. "
        f"Please specify your parallel config in `run_deepseek_v4._get_parallel_config`."
    )


def _train(args: ScriptArgs):
    print(f"[precision] fp8_training={args.fp8_training} nvfp4_experts={args.nvfp4_experts}")
    print(
        f"running on {args.num_nodes} nodes "
        f"({args.actor_num_nodes} actor nodes x {args.actor_num_gpus_per_node} GPUs/node, "
        f"{args.rollout_num_gpus} rollout GPUs, colocate={args.colocate})"
    )
    if args.nvfp4_experts:
        assert _is_blackwell(args), "nvfp4_experts requires Blackwell (SM100+) hardware"
        # The experimental FP4 C4 indexer (--enable-deepseek-v4-fp4-indexer,
        # default off in sglang) must stay off: the NVFP4-experts recipe keeps
        # the indexer on the FP8 path (wq_b blockfp8 GEMM + FP8 index scoring,
        # weights_proj BF16), matching the trainer, which has no FP4-indexer
        # counterpart.
        assert (
            "enable-deepseek-v4-fp4-indexer" not in args.extra_args
        ), "the experimental sglang FP4 indexer is not supported with nvfp4_experts"
    _ensure_4layer_model_type(args)

    load_save_path = f"{args.save_dir}/{args.run_id}/checkpoints"
    ckpt_args = f"--hf-checkpoint {args.hf_checkpoint} " f"--ref-load {args.model_local_dir}/{args.torch_dist_name} "
    if not args.skip_saving:
        ckpt_args += (
            f"--load {load_save_path} " f"--save {load_save_path} " "--save-interval 20 " "--save-retain-interval 20 "
        )

    rollout_args = (
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 3000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-temperature 0.8 "
        "--num-steps-per-rollout 1 "
        "--balance-data "
    )

    if args.mode != "debug_minimal":
        rollout_args += (
            "--over-sampling-batch-size 512 "
            "--dynamic-sampling-filter-path miles.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        )

    eval_args = ""
    if args.enable_eval:
        eval_args += "--eval-interval 20 " "--eval-top-p 0.7 "

    match args.task:
        case "dapo_aime":
            rollout_args += (
                f"--prompt-data {args.data_dir}/dapo-math-17k/dapo-math-17k.jsonl "
                "--input-key prompt "
                f"--rollout-max-response-len 4096 "
                """--apply-chat-template-kwargs '{"thinking_mode":"thinking"}' """
            )
            eval_args += (
                f"--eval-prompt-data aime {args.data_dir}/aime-2024/aime-2024.jsonl "
                "--n-samples-per-eval-prompt 8 "
                "--eval-max-response-len 4096 "
            )
        case "gsm8k":
            rollout_args += (
                f"--prompt-data {args.data_dir}/gsm8k/train.parquet "
                "--input-key messages "
                "--rollout-max-response-len 256 "
            )
            eval_args += (
                f"--eval-prompt-data gsm8k {args.data_dir}/gsm8k/test.parquet "
                "--n-samples-per-eval-prompt 1 "
                "--eval-max-response-len 256 "
            )

    perf_args = _get_parallel_config(args)

    perf_args += (
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if args.optimizer_offload:
        optimizer_args += (
            "--optimizer-cpu-offload " "--use-precision-aware-optimizer " "--overlap-cpu-optimizer-d2h-h2d "
        )

    if args.model_name == "DeepSeek-V4-Pro-FP8":
        sglang_world_size = 32
        sglang_tp_size = 32
        sglang_dp_size = 32
        sglang_ep_size = 32
    else:
        sglang_world_size = 4
        sglang_tp_size = 4
        sglang_dp_size = 1
        sglang_ep_size = 4
    # Explicit rollout kernel backends per recipe (fp8 gemm / moe runner / moe
    # a2a always set together). Notes:
    #   - blockfp8 dense GEMMs (attention + shared experts) use fp8-gemm "auto"
    #     in every recipe: sglang's dispatch_w8a8_block_fp8_linear auto-selects
    #     deep_gemm / flashinfer_trtllm / cutlass by hardware -- the verified
    #     default of all existing V4 FP8 runs (unlike MXFP8, whose dispatch
    #     stays on triton unless flashinfer_cutlass is forced explicitly).
    #   - a FlashInfer TRT-LLM MoE runner requires a2a "none": the fused
    #     routed kernel does its own token dispatch, DeepEP is not used.
    if args.nvfp4_experts:
        # NVFP4 routed experts run on the FlashInfer TRT-LLM routed MoE kernel
        # (the only backend with per-token FP4 activation scales).
        sglang_fp8_gemm_backend = "auto"
        sglang_moe_runner_backend = "flashinfer_trtllm_routed"
        sglang_moe_a2a_backend = "none"
    elif args.model_name == "DeepSeek-V4-Pro-FP8":
        sglang_fp8_gemm_backend = "auto"
        sglang_moe_runner_backend = "deep_gemm"
        sglang_moe_a2a_backend = "deepep"
    else:
        sglang_fp8_gemm_backend = "auto"
        sglang_moe_runner_backend = "auto"
        sglang_moe_a2a_backend = "none"
    sglang_args = (
        f"--rollout-num-gpus-per-engine {sglang_world_size} "
        f"--sglang-fp8-gemm-backend {sglang_fp8_gemm_backend} "
        f"--sglang-moe-runner-backend {sglang_moe_runner_backend} "
        f"--sglang-moe-a2a-backend {sglang_moe_a2a_backend} "
        f"--sglang-tp-size {sglang_tp_size} "
        f"--sglang-dp-size {sglang_dp_size} "
        f"--sglang-ep-size {sglang_ep_size} "
        "--router-health-success-threshold 1 "
        "--router-health-check-interval-secs 15 "
        "--router-health-failure-threshold 40 "  # TODO improve
    )
    if sglang_moe_a2a_backend == "deepep":
        sglang_args += "--sglang-deepep-mode low_latency "
    if args.nvfp4_experts:
        sglang_args += "--sglang-dsa-topk-backend flashinfer "
        sglang_args += "--sglang-disable-shared-experts-fusion "
    if args.model_name == "DeepSeek-V4-Pro-FP8":
        sglang_args += "--sglang-enable-dp-attention " "--sglang-cuda-graph-max-bs 8 "
    if args.enable_mtp:
        sglang_args += (
            "--sglang-speculative-algorithm EAGLE "
            "--sglang-speculative-num-steps 3 "
            "--sglang-speculative-eagle-topk 1 "
            "--sglang-speculative-num-draft-tokens 4 "
        )
    extra_env_vars = {
        "SGLANG_SKIP_CHECKPOINT_LOAD_CHECK": "1",
        "SGLANG_DSV4_FP4_EXPERTS": "1" if args.nvfp4_experts else "0",
        "SGLANG_HEALTH_CHECK_TIMEOUT": "120",
        "SGLANG_DG_CACHE_DIR_PER_PROCESS": "1",
        # wo_a is a deliberate high-precision carve-out in both the plain-FP8
        # and mixed NVFP4+blockFP8 launchers.
        "SGLANG_OPT_FP8_WO_A_GEMM": "0",
        # Quantize the SWA KV cache from bf16-rounded norm outputs (the
        # two-step _compute_kv_bf16 + store_cache path) instead of the fused
        # kernel's fp32 registers, matching the trainer-side QAT regardless of
        # expert recipe. Verified: L0 attention-core diff 1.34e-2 -> 2.0e-3,
        # rollout/train logprob diff 0.140 -> 0.121. FUSED_STORE_CACHE=0 pins
        # the store to the bitwise-verified non-fused quantizer.
        "SGLANG_DSV4_USE_BF16_KV_QUANT_SOURCE": "1",
        "SGLANG_OPT_USE_FUSED_STORE_CACHE": "0",
    }
    if args.nvfp4_experts:
        extra_env_vars |= {
            # Rollout quantizes MoE activations per token, matching the trainer's
            # NVTE_NVFP4_ROW_SCALED_ACTIVATION=1 (the checkpoint's static
            # input_scale is neutralized to ones by sglang in this mode).
            "SGLANG_FLASHINFER_NVFP4_PER_TOKEN_ACTIVATION": "1",
            # FlashInfer FP4 quantization defaults to fast math; disable both
            # aliases so rollout rounding matches TE's exact-math quantization.
            "TRTLLM_DISABLE_FP4_QUANT_FAST_MATH": "1",
            "FLASHINFER_DISABLE_FP4_QUANT_FAST_MATH": "1",
            "SGLANG_DSA_FUSE_TOPK": "0",
            "SGLANG_DSA_TOPK_FLASHINFER_TIE_BREAK": "large",
            # miles injects this by default (NVSHMEM's internal NCCL conflicts
            # with ours under CUDA graphs); pinned explicitly per the DSA
            # alignment reference config.
            "NVSHMEM_DISABLE_NCCL": "1",
        }
    if args.model_name == "DeepSeek-V4-Pro-FP8":
        if sglang_moe_a2a_backend == "deepep":
            extra_env_vars["SGLANG_DEEPEP_NUM_MAX_DISPATCH_TOKENS_PER_RANK"] = "256"
        extra_env_vars["SGLANG_JIT_DEEPGEMM_PRECOMPILE"] = "0"

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--attention-softmax-in-fp32 "
        f"--update-weight-buffer-size {1 * 1024 ** 3} "
        f"--actor-num-nodes {args.actor_num_nodes} "
        f"--actor-num-gpus-per-node {args.actor_num_gpus_per_node} "
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--train-memory-margin-bytes 3221225472 "
        "--sglang-mem-fraction-static 0.7 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--model-name deepseekv4 "  # for mbridge load
        "--qkv-format bshd "
        "--moe-router-freeze-gate "
        "--freeze-e-score-correction-bias "
        "--rollout-health-check-interval 300 "
        "--rollout-health-check-timeout 300 "
    )
    if args.colocate:
        misc_args += "--colocate "
    else:
        misc_args += f"--rollout-num-gpus {args.rollout_num_gpus} "

    if args.dump_details:
        misc_args += f"--dump-details {args.debug_data_root}/{args.run_id}/dump_details "

    if args.enable_mis:
        misc_args += (
            "--use-tis "
            "--custom-config-path examples/train_infer_mismatch_helper/mis.yaml "
            "--custom-tis-function-path examples.train_infer_mismatch_helper.mis.compute_mis_weights_with_cp "
        )

    if args.nvfp4_experts:
        misc_args += "--miles-dsa-topk-backend flashinfer "

    if args.use_fault_tolerance:
        misc_args += "--use-fault-tolerance "

    if args.debug_train_run_id is not None:
        if args.debug_train_rollout_id is None:
            args.debug_train_rollout_id = 1
        misc_args += (
            f"--load-debug-rollout-data "
            f"{args.debug_data_root}/{args.debug_train_run_id}/dump_details/rollout_data/{args.debug_train_rollout_id}.pt "
        )
        misc_args += "--debug-train-only "

    if args.enable_r3:
        misc_args += "--use-rollout-routing-replay "

    if args.train_deterministic:
        misc_args += "--deterministic-mode "
        extra_env_vars |= {
            "NCCL_ALGO": "Ring",
            "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        }

    if args.fp8_training:
        misc_args += "--transformer-impl transformer_engine " "--bf16 " "--fp8-format e4m3 " "--fp8-recipe blockwise "
        # On Blackwell, TE emulates the blockwise recipe with MXFP8, which requires pow2 scales.
        fp32_scales = "0" if _is_blackwell(args) else "1"
        train_env_vars = {"NVTE_FP8_BLOCK_SCALING_FP32_SCALES": fp32_scales}
        if args.nvfp4_experts:
            train_env_vars |= _NVFP4_TRAIN_ENV_VARS
        misc_args += f"--train-env-vars '{json.dumps(train_env_vars, separators=(',', ':'))}' "
        # Keep the DSA indexer weights_proj (a TELinear) in BF16 on the trainer: blockwise
        # fp8 on weights_proj is numerically unstable, so override it back to BF16 via TE.
        # With nvfp4_experts, additionally override the routed-expert grouped GEMMs to NVFP4.
        if args.nvfp4_experts:
            te_precision_config = _DSV4_NVFP4_TE_PRECISION_CONFIG
        else:
            te_precision_config = _DSV4_FP8_TE_PRECISION_CONFIG
        if "--te-precision-config-file" not in args.extra_args:
            misc_args += f"--te-precision-config-file " f"{U.save_to_temp_file(te_precision_config, 'yaml')} "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__, run_id=args.run_id)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{sglang_args} "
        f"{misc_args} "
        f"{args.extra_args} "
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        extra_env_vars={**extra_env_vars},
        megatron_path=args.megatron_path,
    )


@app.command()
@U.dataclass_cli
def train(args: ScriptArgs):
    """Run training. Assumes data/model/torch_dist are already prepared on {model_local_dir}."""
    _train(args)


@app.command()
@U.dataclass_cli
def full_train(args: ScriptArgs):
    _prepare_download(args)

    bf16_dir = Path(f"{args.model_dir}/{args.bf16_name}")
    bf16_sentinel = bf16_dir / "model.safetensors.index.json"
    if not bf16_sentinel.exists():
        _prepare_single(args)
    else:
        print(f"[full_train] Skipping FP8->BF16 cast: {bf16_sentinel} already exists.")
        if args.nvfp4_experts:
            # BF16 cast may predate the NVFP4 conversion; this self-skips when done.
            _convert_nvfp4_blockfp8(args)

    torch_dist_dir = Path(f"{args.model_dir}/{args.torch_dist_name}")
    torch_dist_sentinel = torch_dist_dir / "latest_checkpointed_iteration.txt"
    if not torch_dist_sentinel.exists():
        _prepare_spmd(args)
    else:
        print(f"[full_train] Skipping BF16->torch_dist conversion: {torch_dist_sentinel} already exists.")

    if args.model_local_dir != args.model_dir:
        _prepare_cp(args)
    else:
        print(f"[full_train] Skipping rsync: model_local_dir == model_dir ({args.model_dir})")

    if args.hf_checkpoint is None:
        # With NVFP4 experts the rollout serves the converted mixed checkpoint;
        # otherwise it serves the source FP8 checkpoint.
        rollout_ckpt_name = args.nvfp4_name if args.nvfp4_experts else args.model_name
        args.hf_checkpoint = f"{args.model_local_dir}/{rollout_ckpt_name}"

    _train(args)


if __name__ == "__main__":
    app()
