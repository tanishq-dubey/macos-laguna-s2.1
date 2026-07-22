# Laguna S 2.1 MLX benchmark

This repository contains a reproducible local harness for comparing Laguna S 2.1 MLX quantizations on Apple Silicon. Each run records the raw output and task score along with token and prefill throughput, peak MLX memory, load time, package versions, model revision, and machine metadata.

This work was sponsored by [DWS LLC](https://dws.rip).

On the M5 Max used for these tests, the smallest quant, `mlx-community/Laguna-S-2.1-oQ2e`, running in-process through `mlx-vlm` was the fastest option. It passed all six tasks and finished the suite faster than the tested llama.cpp, DFlash, official NVFP4 MLX, and serial oMLX paths. The measurements are in `BENCHMARK_RESULTS.md`.

## Results at a glance

These results were measured on a 128 GB Apple M5 Max using macOS 27.0, Python 3.13.12, MLX 0.32.0, and mlx-vlm 0.6.6. The score combines three generation tasks and three agentic coding tasks. Throughput is weighted across the generation tasks.

| Quant | Overall score | Generation | Agentic | Generation tok/s | Peak MLX GB | Suite time |
|---|---:|---:|---:|---:|---:|---:|
| `mlx-community/Laguna-S-2.1-oQ2e` | 1.000 | 1.000 | 1.000 | 40.85 | 37.77 | 87.38s |
| `mlx-community/Laguna-S-2.1-oQ3e` | 0.625 | 0.417 | 0.833 | 48.58 | 50.69 | 84.79s |
| `poolside/Laguna-S-2.1-NVFP4-mlx` | 0.875 | 0.750 | 1.000 | 7.25 | 73.47 | 301.77s |

The oQ2e conversion was the clear default for this machine. It passed all 38 hidden assertions while using about 38 GB of peak MLX memory. oQ3e decoded faster in the controlled profile, but its exact-format and implementation errors lowered its task score. The official NVFP4 MLX build was functional but much slower in this runtime.

Long-context retrieval also passed at every tested size through 256K tokens on oQ2e:

| Prompt tokens | Prefill tok/s | Decode tok/s | Peak MLX GB | Retrieval |
|---:|---:|---:|---:|---:|
| 1,016 | 1411.42 | 60.86 | 37.23 | pass |
| 16,376 | 1613.07 | 52.48 | 38.46 | pass |
| 65,528 | 1088.75 | 32.03 | 41.19 | pass |
| 131,064 | 771.80 | 23.48 | 45.09 | pass |
| 262,136 | 566.31 | 12.16 | 52.87 | pass |

For a local coding setup, use oQ2e through in-process mlx-vlm with temperature 0, top-p 1, the default unquantized KV cache, and a prefill step of 2048. Staying at or below 64K gives a much better latency balance. The 128K and 256K cases fit and passed retrieval, but decode speed fell sharply.

The full [benchmark report](BENCHMARK_RESULTS.md) includes the standardized quant profile, sampling search, KV cache and prefill search, engine bake-off, revisions, and compatibility failures. The machine-readable source is [the combined CSV](results/laguna_s21_full_results.csv).

## Clone and run

The helper creates the environment and uses the fastest quant from our tests by default:

```bash
git clone https://github.com/tanishq-dubey/macos-laguna-s2.1.git laguna-s21-bench
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

## Contributing results from your Mac

Results from other Apple Silicon machines are welcome. The harness records the chip, memory, macOS version, model revision, and runtime versions, then merges your rows into the committed CSV without removing earlier community results.

First, fork the repository on GitHub and clone your fork. Install [uv](https://docs.astral.sh/uv/) if needed, then create a branch and sync the locked environment:

```bash
git clone https://github.com/<your-user>/macos-laguna-s2.1.git
cd macos-laguna-s2.1
git switch -c results/<model>-<chip>
uv sync --extra dev --python 3.13 --locked
```

The reference oQ2e model is a 33.74 GiB download and peaked near 38 GB in the short suite. Make sure your Mac has enough unified memory and free disk space before starting. Download the default model and reproduce the six quality tasks plus the standardized quant profile:

```bash
scripts/laguna.sh download
scripts/laguna.sh bench
```

To test a different conversion, set `LAGUNA_MODEL` for both commands:

```bash
export LAGUNA_MODEL=mlx-community/Laguna-S-2.1-oQ3e
scripts/laguna.sh download
scripts/laguna.sh bench
```

For the complete context, sampling, KV cache, and prefill matrix, run the community profile. It reaches 256K input tokens and can take several minutes per long case on the reference M5 Max:

```bash
LAGUNA_MODEL=mlx-community/Laguna-S-2.1-oQ2e scripts/laguna.sh community
```

Review the comparison, refresh the merged CSV, and run the tests before committing:

```bash
uv run --frozen laguna-bench compare --output results
uv run --frozen laguna-bench export --output results
uv run --frozen pytest -q
git diff -- results/laguna_s21_full_results.csv
```

Raw run directories stay local because they are large and can contain machine-specific paths. Add the combined CSV to your pull request. Update `BENCHMARK_RESULTS.md` only when your run adds a finding that needs explanation.

```bash
git add results/laguna_s21_full_results.csv
git commit -m "Add <chip> results for <model>"
git push -u origin HEAD
gh pr create --fill
```

In the pull request, briefly mention your Mac model, chip, unified memory, macOS version, and whether the run was plugged in and otherwise idle. Do not edit or delete existing CSV rows by hand. Load failures are useful results too: the harness records them as failure rows so compatibility gaps remain visible.

## Safety

The agent can list, read, and write files only inside its disposable task workspace. It can also invoke the fixture's fixed pytest command. Generation graders execute model-produced Python in a temporary directory with a timeout, but this does not provide a hardened OS sandbox. Run only models you trust, and keep sensitive data on the machine backed up.
