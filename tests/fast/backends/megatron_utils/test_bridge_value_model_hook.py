"""Tests for the bridge value-model hook in bridge_lora_helpers.

``_make_value_model_hook`` swaps a bridge-built model's language-model head for
a scalar value head. Its body had no coverage, which let a constructor mismatch
go unnoticed: it passed ``sequence_parallel=`` to ``LinearForLastLayer``, whose
``__init__`` takes a keyword-only ``config`` and no ``sequence_parallel``. Any
run that reached the hook raised ``TypeError``. These tests exercise the hook
directly so the call site cannot drift from the constructor again.
"""

from tests.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=60, suite="stage-a-cpu", labels=[])


import ast
import inspect
import sys
import types
from pathlib import Path

import pytest

_MISSING = object()


class _RecordingValueHead:
    """Stand-in for ``LinearForLastLayer`` carrying its real signature.

    The keyword-only ``config`` is what makes this test meaningful: passing any
    other keyword, or omitting ``config``, raises ``TypeError`` exactly as the
    real class would.
    """

    def __init__(self, input_size: int, output_size: int, *, config, bias: bool = True) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.config = config
        self.bias = bias


class _FakeModelChunk:
    def __init__(self, sequence_parallel: bool = False) -> None:
        self.config = types.SimpleNamespace(sequence_parallel=sequence_parallel)
        self.output_layer = "original-lm-head"


@pytest.fixture(scope="module")
def helpers_module():
    """Stub the megatron surface the hook touches and import the helpers once.

    Module scope matters: torch cannot survive being evicted from sys.modules
    and re-imported, so the stubs are installed and reverted a single time.
    """
    installed: dict[str, object] = {}

    def install(name: str, module) -> None:
        installed[name] = sys.modules.get(name, _MISSING)
        sys.modules[name] = module

    def stub(name: str, attrs: dict | None = None, is_package: bool = False):
        module = types.ModuleType(name)
        if is_package:
            module.__path__ = []
        for attr_name, value in (attrs or {}).items():
            setattr(module, attr_name, value)
        install(name, module)
        return module

    parallel_state = types.SimpleNamespace()
    try:
        stub("megatron", is_package=True)
        stub("megatron.core", is_package=True)
        stub("megatron.core.utils", {"get_attr_wrapped_model": lambda *a, **k: None})
        install("megatron.core.parallel_state", parallel_state)

        from miles.backends.megatron_utils import bridge_lora_helpers

        # The hook imports LinearForLastLayer lazily from the sibling module, so
        # swapping the module in sys.modules is enough to intercept it.
        stub(
            "miles.backends.megatron_utils.model_provider",
            {"LinearForLastLayer": _RecordingValueHead},
        )
        yield types.SimpleNamespace(helpers=bridge_lora_helpers, parallel_state=parallel_state)
    finally:
        for name, previous in installed.items():
            if previous is _MISSING:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


@pytest.fixture
def hook_env(helpers_module):
    """Reset the stubbed topology to a single non-virtual pipeline stage."""
    helpers_module.parallel_state.get_pipeline_model_parallel_world_size = lambda: 1
    helpers_module.parallel_state.get_virtual_pipeline_model_parallel_world_size = lambda: None
    helpers_module.parallel_state.is_pipeline_last_stage = lambda **kwargs: True
    return helpers_module


def test_value_head_replaces_output_layer(hook_env):
    """The hook installs a scalar value head on the last pipeline stage."""
    hook = hook_env.helpers._make_value_model_hook(4096)
    chunk = _FakeModelChunk()

    hook([chunk])

    assert isinstance(chunk.output_layer, _RecordingValueHead)
    assert chunk.output_layer.input_size == 4096
    assert chunk.output_layer.output_size == 1


def test_value_head_receives_the_chunk_config(hook_env):
    """``config`` comes from the model chunk, which is where sequence_parallel lives."""
    hook = hook_env.helpers._make_value_model_hook(2048)
    chunk = _FakeModelChunk(sequence_parallel=True)

    hook([chunk])

    assert chunk.output_layer.config is chunk.config
    assert chunk.output_layer.config.sequence_parallel is True


def test_hook_takes_only_hidden_size(hook_env):
    """The factory must not resurrect a redundant sequence_parallel parameter."""
    params = inspect.signature(hook_env.helpers._make_value_model_hook).parameters

    assert list(params) == ["hidden_size"]


def test_hook_skips_non_last_pipeline_stages(hook_env):
    """With virtual pipelining, only last-stage chunks get a value head."""
    hook_env.parallel_state.get_pipeline_model_parallel_world_size = lambda: 2
    hook_env.parallel_state.get_virtual_pipeline_model_parallel_world_size = lambda: 2
    hook_env.parallel_state.is_pipeline_last_stage = lambda ignore_virtual, vp_stage: vp_stage == 1

    hook = hook_env.helpers._make_value_model_hook(1024)
    chunks = [_FakeModelChunk(), _FakeModelChunk()]

    hook(chunks)

    assert chunks[0].output_layer == "original-lm-head"
    assert isinstance(chunks[1].output_layer, _RecordingValueHead)


def test_value_head_kwargs_match_the_real_constructor(hook_env):
    """Guard against drift: bind the hook's kwargs to the real LinearForLastLayer.

    The other tests use a stand-in class, so they would keep passing if the real
    constructor changed. This one parses the real signature out of
    ``model_provider.py`` and binds against it, with no megatron import needed.
    """
    import miles.backends.megatron_utils as megatron_utils_pkg

    source_path = Path(megatron_utils_pkg.__file__).parent / "model_provider.py"
    tree = ast.parse(source_path.read_text())
    init = next(
        node
        for cls in ast.walk(tree)
        if isinstance(cls, ast.ClassDef) and cls.name == "LinearForLastLayer"
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == "__init__"
    )
    parameters = [
        inspect.Parameter(arg.arg, inspect.Parameter.POSITIONAL_OR_KEYWORD) for arg in init.args.args[1:]
    ] + [
        inspect.Parameter(
            arg.arg,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if default is None else ast.literal_eval(default),
        )
        for arg, default in zip(init.args.kwonlyargs, init.args.kw_defaults, strict=True)
    ]

    hook = hook_env.helpers._make_value_model_hook(512)
    chunk = _FakeModelChunk()
    hook([chunk])
    head = chunk.output_layer

    inspect.Signature(parameters).bind(input_size=head.input_size, output_size=head.output_size, config=head.config)


def test_lora_mxfp8_config_reaches_bridge_provider(helpers_module):
    provider = types.SimpleNamespace(fp8=None, fp8_recipe=None, fp8_param=False)
    args = types.SimpleNamespace(fp8_param_gather=True, fp8="e4m3", fp8_recipe="mxfp8")

    helpers_module.helpers._configure_lora_mxfp8_provider(provider, args)

    assert provider.fp8 == "e4m3"
    assert provider.fp8_recipe == "mxfp8"
    assert provider.fp8_param is True


def test_lora_mxfp8_rejects_bridge_without_native_primary_support(helpers_module):
    provider = types.SimpleNamespace(fp8=None, fp8_recipe=None)
    args = types.SimpleNamespace(fp8_param_gather=True, fp8="e4m3", fp8_recipe="mxfp8")

    with pytest.raises(RuntimeError, match="does not expose fp8_param"):
        helpers_module.helpers._configure_lora_mxfp8_provider(provider, args)


def test_lora_adapter_uses_no_backup_tms_region(helpers_module, monkeypatch):
    marker = object()
    calls = []
    saver = types.SimpleNamespace(region=lambda **kwargs: calls.append(kwargs) or marker)
    monkeypatch.setitem(
        sys.modules,
        "torch_memory_saver",
        types.SimpleNamespace(torch_memory_saver=saver),
    )
    args = types.SimpleNamespace(fp8_param_gather=True, offload_train=True)

    assert helpers_module.helpers._lora_adapter_allocation_region(args) is marker
    assert calls == [{"tag": "lora_adapter", "enable_cpu_backup": False}]


class _FakeQuantizedParam:
    def __init__(self, *, requires_grad: bool, init_value) -> None:
        self.requires_grad = requires_grad
        self._init_value = init_value
        self.clear_calls = 0

    def get_high_precision_init_val(self):
        return self._init_value

    def clear_high_precision_init_val(self):
        self._init_value = None
        self.clear_calls += 1


class _FakeParamModel:
    def __init__(self, params, modules=()) -> None:
        self._params = params
        self._modules = modules

    def parameters(self):
        return iter(self._params)

    def modules(self):
        return iter(self._modules)


def test_frozen_te_init_copy_is_cleared_once(helpers_module):
    frozen = _FakeQuantizedParam(
        requires_grad=False,
        init_value=types.SimpleNamespace(numel=lambda: 8, element_size=lambda: 2),
    )
    trainable = types.SimpleNamespace(requires_grad=True)
    model = _FakeParamModel([frozen, frozen, trainable])

    count, num_bytes = helpers_module.helpers._clear_frozen_high_precision_init_values([model])

    assert (count, num_bytes) == (1, 16)
    assert frozen.clear_calls == 1


def test_trainable_native_quantized_parameter_is_rejected(helpers_module):
    trainable = _FakeQuantizedParam(
        requires_grad=True,
        init_value=types.SimpleNamespace(numel=lambda: 4, element_size=lambda: 2),
    )

    with pytest.raises(RuntimeError, match="only BF16/FP32 adapter parameters"):
        helpers_module.helpers._clear_frozen_high_precision_init_values([_FakeParamModel([trainable])])


def test_missing_native_quantized_base_is_rejected(helpers_module):
    model = _FakeParamModel([types.SimpleNamespace(requires_grad=False)])

    with pytest.raises(RuntimeError, match="did not construct any native quantized"):
        helpers_module.helpers._clear_frozen_high_precision_init_values([model])


def test_lora_base_router_bias_is_frozen(helpers_module):
    router = types.SimpleNamespace(frozen_expert_bias=False, expert_bias=object())
    model = _FakeParamModel([], [router])

    assert helpers_module.helpers._freeze_lora_base_expert_bias([model]) == 1
    assert router.frozen_expert_bias is True
