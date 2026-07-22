# Laguna S 2.1 MLX benchmark

This repository contains a reproducible local harness for comparing Laguna S 2.1 MLX quantizations on Apple Silicon. Each run records the raw output and task score along with token and prefill throughput, peak MLX memory, load time, package versions, model revision, and machine metadata.

This work was sponsored by [DWS LLC](https://dws.rip).

On the M5 Max used for these tests, the smallest quant, `mlx-community/Laguna-S-2.1-oQ2e`, running in-process through `mlx-vlm` was the fastest option. It passed all six tasks and finished the suite faster than the tested llama.cpp, DFlash, official NVFP4 MLX, and serial oMLX paths. The measurements are in `BENCHMARK_RESULTS.md`.

## Clone and run

The helper creates the environment and uses the fastest quant from our tests by default:

```bash
git clone <this-repository-url> laguna-s21-bench
cd laguna-s21-bench
scripts/laguna.sh download
scripts/laguna.sh prompt 'Write a Python LRU cache with tests'
```

Start an interactive chat or a local OpenAI-compatible server:

```bash
scripts/laguna.sh chat
scripts/laguna.sh server
```

The server binds to `127.0.0.1:8080`. Set `LAGUNA_MODEL`, `LAGUNA_HOST`, or `LAGUNA_PORT` to change the defaults. For example:

```bash
LAGUNA_MODEL=mlx-community/Laguna-S-2.1-oQ3e scripts/laguna.sh server
```

The suite has six fixed tasks:

| Kind | Tier | What it measures |
|---|---|---|
| generation | small | exact structured-output instruction following |
| generation | medium | single-function Python synthesis against hidden tests |
| generation | large | larger algorithm/module synthesis against hidden tests |
| agentic | small | inspect files, derive an answer, and write an artifact |
| agentic | medium | diagnose and repair a tested Python bug |
| agentic | large | implement a multi-file feature and satisfy tests |

Generation uses greedy decoding (`temperature=0`) with a fixed MLX seed. Each agent task starts in a fresh temporary workspace. The agent has access only to an allowlist of tools, and the harness records every turn. Task prompts and tests are versioned in source. Here, "deterministic" means that the decoding settings and fixtures are repeatable. Metal kernels and new model or runtime revisions can still change the results.

`--agent-prompt-cache` enables experimental MLX KV reuse. It improves throughput within each turn, but mlx-vlm 0.6.6 changed a greedy Laguna trajectory during testing because Laguna mixes global caches with rotating sliding-window caches. The option stays off by default until cold and cached runs produce exactly the same output.

## Setup

```bash
uv sync --extra dev --python 3.13 --locked
```

Laguna support currently comes from `mlx-vlm`, even though these are text-only models.

## Run

List tasks and the curated quant ladder:

```bash
uv run --frozen laguna-bench list
```

Run the complete suite on the smallest quant:

```bash
uv run --frozen laguna-bench run \
  --model mlx-community/Laguna-S-2.1-oQ2e \
  --output results
```

Run a cheap smoke test first:

```bash
uv run --frozen laguna-bench run \
  --model mlx-community/Laguna-S-2.1-oQ2e \
  --task generation-small \
  --output results
```

Every run creates `results/<timestamp>-<model>/run.json`, a readable `summary.md`, the raw generations, and the final agent workspaces. A failed task counts as a benchmark result and does not cause the CLI itself to fail. A model or runtime failure does.

Compare the latest complete run for every tested model:

```bash
uv run --frozen laguna-bench compare --output results
```

Run the standardized short profile for a new quant or the complete context and hyperparameter matrix:

```bash
uv run --frozen laguna-bench sweep --model mlx-community/Laguna-S-2.1-oQ3e --profile quant
scripts/laguna.sh community
```

The full profile covers 256 to 262,144 input tokens, decodes from 64 to 1,024 tokens, three sampling configurations, uniform and TurboQuant KV cache options, a fixed KV window, and several prefill chunk sizes. The 256K case takes several minutes on the reference M5 Max. To export the catalog, task results, and performance records to one CSV, run:

```bash
uv run --frozen laguna-bench export --output results
```

The output is `results/laguna_s21_full_results.csv`. Raw generations and error details remain in the adjacent JSON artifacts.

The harness can also benchmark a local `llama-server` (including Poolside's DFlash build) through its OpenAI-compatible endpoint:

```bash
uv run --frozen laguna-bench run \
  --engine openai \
  --base-url http://127.0.0.1:8080/v1 \
  --model laguna \
  --output results
```

## Quant ladder for this 128 GB Mac

Sizes are repository payloads observed on 2026-07-21 and should be refreshed before downloading:

1. `mlx-community/Laguna-S-2.1-oQ2e`: 33.74 GiB, calibrated 2.70 bpw
2. `mlx-community/Laguna-S-2.1-oQ3e`: 45.86 GiB, faster decode but lower quality in this suite
3. `poolside/Laguna-S-2.1-NVFP4-mlx`: 66.97 GiB, official testing-only build; functional but slow on this runtime
4. `poolside/Laguna-S-2.1-GGUF` Q4_K_M: 70.01 GiB, functional through Poolside's llama.cpp branch

The Vontra and pipenetwork 4-bit MLX conversions both fail to load in mlx-vlm 0.6.6, each because of a different router or quantization incompatibility. We did not download their 6-bit derivatives after those failures. The 116.34 GiB 8-bit conversions leave too little headroom for weights, the KV cache, and macOS on a 128 GB machine. The CSV still includes every variant and records each failure or omission explicitly.

## Safety

The agent can list, read, and write files only inside its disposable task workspace. It can also invoke the fixture's fixed pytest command. Generation graders execute model-produced Python in a temporary directory with a timeout, but this does not provide a hardened OS sandbox. Run only models you trust, and keep sensitive data on the machine backed up.
