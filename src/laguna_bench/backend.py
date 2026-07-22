from __future__ import annotations

import importlib.metadata
import importlib.util
import hashlib
import json
import sys
import time
import urllib.request
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass
class Generation:
    text: str
    prompt_tokens: int = 0
    generation_tokens: int = 0
    prompt_tps: float = 0.0
    generation_tps: float = 0.0
    peak_memory_gb: float = 0.0
    finish_reason: str | None = None
    cached_tokens: int = 0
    elapsed_seconds: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class MLXBackend:
    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        seed: int = 20260721,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        from mlx_vlm import load

        self.model_id = model_id
        self.revision = revision
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        started = time.perf_counter()
        self.model, self.processor = load(model_id, revision=revision, lazy=False)
        self.load_seconds = time.perf_counter() - started
        self.tokenizer = getattr(self.processor, "tokenizer", self.processor)
        self.snapshot = _snapshot_metadata(model_id, revision)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        prompt_cache_state: Any | None = None,
        generation_options: dict[str, Any] | None = None,
    ) -> Generation:
        from mlx_vlm import generate

        template_args: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools:
            template_args["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(messages, **template_args)
        started = time.perf_counter()
        generation_args: dict[str, Any] = {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "seed": self.seed,
        }
        if prompt_cache_state is not None:
            generation_args["prompt_cache_state"] = prompt_cache_state
        if generation_options:
            generation_args.update(generation_options)
        result = generate(
            self.model,
            self.processor,
            prompt=prompt,
            max_tokens=max_tokens,
            verbose=False,
            **generation_args,
        )
        elapsed = time.perf_counter() - started
        return Generation(
            text=result.text,
            prompt_tokens=result.prompt_tokens,
            generation_tokens=result.generation_tokens,
            prompt_tps=result.prompt_tps,
            generation_tps=result.generation_tps,
            peak_memory_gb=result.peak_memory,
            finish_reason=result.finish_reason,
            cached_tokens=result.cached_tokens,
            elapsed_seconds=elapsed,
        )

    @staticmethod
    def new_prompt_cache():
        from mlx_vlm import PromptCacheState

        return PromptCacheState()

    def metadata(self) -> dict[str, Any]:
        config = getattr(self.model, "config", None)
        quantization = getattr(config, "quantization", None) or getattr(config, "quantization_config", None)
        return {
            "model_id": self.model_id,
            "engine": "mlx-vlm",
            "requested_revision": self.revision,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.snapshot,
            "load_seconds": self.load_seconds,
            "model_type": getattr(config, "model_type", None),
            "max_position_embeddings": getattr(config, "max_position_embeddings", None),
            "quantization": _jsonable(quantization),
            "mlx_version": _version("mlx"),
            "mlx_vlm_version": _version("mlx-vlm"),
            "transformers_version": _version("transformers"),
        }


class MLXLMCustomBackend:
    """MLX-LM backend for conversions that ship a reviewed model loader.

    The loader is imported from the pinned Hugging Face snapshot without
    copying it into the installed ``mlx_lm`` package.
    """

    def __init__(
        self,
        model_id: str,
        revision: str | None = None,
        loader_file: str = "laguna.py",
        seed: int = 20260721,
        temperature: float = 0.0,
        top_p: float = 1.0,
    ):
        from huggingface_hub import snapshot_download
        from mlx_lm import load

        local_model = Path(model_id)
        if not local_model.is_dir() and revision is None:
            raise ValueError("mlx-lm-custom requires --revision for a Hugging Face model")
        self.model_id = model_id
        self.revision = revision
        self.loader_file = loader_file
        self.seed = seed
        self.temperature = temperature
        self.top_p = top_p
        if local_model.is_dir():
            loader_snapshot = local_model
        else:
            loader_snapshot = Path(
                snapshot_download(
                    model_id,
                    revision=revision,
                    allow_patterns=["config.json", loader_file],
                )
            )
        loader = _register_custom_mlx_model(loader_snapshot, loader_file)
        self.loader_sha256 = hashlib.sha256(loader.read_bytes()).hexdigest()
        started = time.perf_counter()
        self.model, self.tokenizer, self.config = load(
            model_id,
            revision=revision,
            tokenizer_config={"fix_mistral_regex": True},
            lazy=False,
            return_config=True,
        )
        self.load_seconds = time.perf_counter() - started
        self.snapshot = _snapshot_metadata(model_id, revision)

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        prompt_cache_state: Any | None = None,
        generation_options: dict[str, Any] | None = None,
    ) -> Generation:
        import mlx.core as mx
        from mlx_lm import stream_generate
        from mlx_lm.sample_utils import make_sampler

        template_args: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": False,
        }
        if tools:
            template_args["tools"] = tools
        prompt = self.tokenizer.apply_chat_template(messages, **template_args)
        options = dict(generation_options or {})
        temperature = options.pop("temperature", self.temperature)
        top_p = options.pop("top_p", self.top_p)
        options["sampler"] = make_sampler(temp=temperature, top_p=top_p)
        if prompt_cache_state is not None:
            options["prompt_cache"] = prompt_cache_state
        mx.random.seed(self.seed)
        started = time.perf_counter()
        chunks = list(
            stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt,
                max_tokens=max_tokens,
                **options,
            )
        )
        elapsed = time.perf_counter() - started
        if not chunks:
            return Generation(text="", elapsed_seconds=elapsed)
        final = chunks[-1]
        return Generation(
            text="".join(chunk.text for chunk in chunks),
            prompt_tokens=final.prompt_tokens,
            generation_tokens=final.generation_tokens,
            prompt_tps=final.prompt_tps,
            generation_tps=final.generation_tps,
            peak_memory_gb=final.peak_memory,
            finish_reason=final.finish_reason,
            elapsed_seconds=elapsed,
        )

    def new_prompt_cache(self):
        from mlx_lm.models.cache import make_prompt_cache

        return make_prompt_cache(self.model)

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "engine": "mlx-lm-custom",
            "requested_revision": self.revision,
            "temperature": self.temperature,
            "top_p": self.top_p,
            **self.snapshot,
            "load_seconds": self.load_seconds,
            "model_type": self.config.get("model_type"),
            "max_position_embeddings": self.config.get("max_position_embeddings"),
            "quantization": _jsonable(self.config.get("quantization")),
            "loader_file": self.loader_file,
            "loader_sha256": self.loader_sha256,
            "fix_mistral_regex": True,
            "mlx_version": _version("mlx"),
            "mlx_lm_version": _version("mlx-lm"),
            "transformers_version": _version("transformers"),
        }


def _register_custom_mlx_model(snapshot: Path, loader_file: str) -> Path:
    config = json.loads((snapshot / "config.json").read_text())
    model_type = config["model_type"]
    loader = snapshot / loader_file
    if not loader.is_file():
        raise FileNotFoundError(f"Custom MLX loader not found: {loader}")
    module_name = f"mlx_lm.models.{model_type}"
    spec = importlib.util.spec_from_file_location(module_name, loader)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import custom MLX loader: {loader}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    try:
        spec.loader.exec_module(module)
    except Exception:
        sys.modules.pop(module_name, None)
        raise
    if not hasattr(module, "Model") or not hasattr(module, "ModelArgs"):
        sys.modules.pop(module_name, None)
        raise ImportError(f"Custom MLX loader must define Model and ModelArgs: {loader}")
    return loader


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool, list, dict)):
        return value
    if hasattr(value, "to_dict"):
        return value.to_dict()
    try:
        return json.loads(json.dumps(value))
    except (TypeError, ValueError):
        return str(value)


def _snapshot_metadata(model_id: str, revision: str | None) -> dict[str, Any]:
    path: Path | None = None
    try:
        from huggingface_hub import snapshot_download

        path = Path(snapshot_download(model_id, revision=revision, local_files_only=True))
    except (OSError, ValueError):
        # Model loaders may deliberately fetch only inference files. Recent
        # huggingface_hub versions reject such a locally usable snapshot when
        # metadata files are absent, so resolve the cache ref directly.
        try:
            from huggingface_hub.constants import HF_HUB_CACHE

            repo = Path(HF_HUB_CACHE) / f"models--{model_id.replace('/', '--')}"
            ref = revision or "main"
            ref_path = repo / "refs" / ref
            commit = ref_path.read_text().strip() if ref_path.is_file() else ref
            candidate = repo / "snapshots" / commit
            if candidate.is_dir():
                path = candidate
        except OSError:
            path = None

    if path is None:
        return {"resolved_revision": None, "snapshot_path": None, "model_bytes": None}
    path = path.absolute()
    files = [item for item in path.rglob("*") if item.is_file()]
    return {
        "resolved_revision": path.name if path.parent.name == "snapshots" else None,
        "snapshot_path": str(path),
        "model_bytes": sum(item.stat().st_size for item in files),
    }


class OpenAIBackend:
    """Backend for llama-server and other local OpenAI-compatible engines."""

    def __init__(
        self,
        model_id: str,
        base_url: str,
        seed: int = 20260721,
        api_key: str | None = None,
        revision: str | None = None,
        model_file: Path | None = None,
    ):
        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.seed = seed
        self.api_key = api_key
        self.revision = revision
        self.model_file = model_file
        self.load_seconds = 0.0
        self._request("GET", "/models")

    def generate(
        self,
        messages: list[dict[str, Any]],
        *,
        max_tokens: int,
        tools: list[dict[str, Any]] | None = None,
        prompt_cache_state: Any | None = None,
    ) -> Generation:
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": _openai_messages(messages),
            "temperature": 0.0,
            "max_tokens": max_tokens,
            "seed": self.seed,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        if tools:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        started = time.perf_counter()
        response = self._request("POST", "/chat/completions", payload)
        elapsed = time.perf_counter() - started
        choice = response["choices"][0]
        message = choice["message"]
        text = message.get("content") or ""
        for call in message.get("tool_calls") or []:
            function = call.get("function", {})
            try:
                arguments = json.loads(function.get("arguments") or "{}")
            except json.JSONDecodeError:
                arguments = {}
            text += _native_tool_call(function.get("name", ""), arguments)
        usage = response.get("usage") or {}
        timings = response.get("timings") or {}
        return Generation(
            text=text,
            prompt_tokens=usage.get("prompt_tokens", timings.get("prompt_n", 0)),
            generation_tokens=usage.get("completion_tokens", timings.get("predicted_n", 0)),
            prompt_tps=timings.get("prompt_per_second", 0.0),
            generation_tps=timings.get("predicted_per_second", 0.0),
            finish_reason=choice.get("finish_reason"),
            cached_tokens=usage.get("prompt_tokens_details", {}).get("cached_tokens", 0),
            elapsed_seconds=elapsed,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "engine": "openai-compatible",
            "base_url": self.base_url,
            "resolved_revision": self.revision,
            "model_file": self.model_file.name if self.model_file else None,
            "model_bytes": self.model_file.stat().st_size if self.model_file else None,
            "load_seconds": self.load_seconds,
        }

    def _request(self, method: str, path: str, payload: dict[str, Any] | None = None) -> dict[str, Any]:
        data = json.dumps(payload).encode() if payload is not None else None
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = urllib.request.Request(
            self.base_url + path,
            data=data,
            method=method,
            headers=headers,
        )
        with urllib.request.urlopen(request, timeout=3600) as response:
            return json.loads(response.read())


def _openai_messages(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = json.loads(json.dumps(messages))
    last_call_id: str | None = None
    for message_index, message in enumerate(normalized):
        for call_index, call in enumerate(message.get("tool_calls") or []):
            call.setdefault("id", f"call_{message_index}_{call_index}")
            last_call_id = call["id"]
            function = call.get("function", {})
            if isinstance(function.get("arguments"), dict):
                function["arguments"] = json.dumps(function["arguments"], separators=(",", ":"))
        if message.get("role") == "tool" and "tool_call_id" not in message and last_call_id:
            message["tool_call_id"] = last_call_id
    return normalized


def _native_tool_call(name: str, arguments: dict[str, Any]) -> str:
    parts = [f"<tool_call>{name}"]
    for key, value in arguments.items():
        rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
        parts.append(f"<arg_key>{key}</arg_key><arg_value>{rendered}</arg_value>")
    parts.append("</tool_call>")
    return "".join(parts)


class ScriptedBackend:
    """Small deterministic backend used by the harness tests."""

    def __init__(self, responses: list[str]):
        self.responses = iter(responses)
        self.load_seconds = 0.0

    def generate(self, messages, *, max_tokens, tools=None, prompt_cache_state=None) -> Generation:
        return Generation(text=next(self.responses), generation_tokens=1, generation_tps=1.0)

    @staticmethod
    def new_prompt_cache():
        return None

    def metadata(self) -> dict[str, Any]:
        return {"model_id": "scripted", "engine": "scripted", "load_seconds": 0.0}
