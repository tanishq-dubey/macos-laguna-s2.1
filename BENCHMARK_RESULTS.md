# Laguna S 2.1 benchmark results

These results were measured on July 21 and 22, 2026, using a 128 GB Apple M5 Max MacBook Pro with macOS 27.0. The canonical harness ran Python 3.13.12, MLX 0.32.0, mlx-vlm 0.6.6, and MLX-LM 0.31.3. It used greedy decoding with seed `20260721` and did not reuse the prompt cache.

## Quant results

| Quant | Revision | Score | Generation | Agentic | Weighted generation tok/s | Peak GB | Suite wall time |
|---|---|---:|---:|---:|---:|---:|---:|
| `pipenetwork/Laguna-S-2.1-MLX-2bit` | `5a67ae47cdc38ec7d16a09f9efb7add1bb631131` | 1.000 | 1.000 | 1.000 | 63.86 | 39.31 | 87.47s |
| `unsloth/Laguna-S-2.1-GGUF` `UD-Q2_K_XL` | `8615cd7d1f90a4e83e13c0954ef6ed543b66f54a` | 0.875 | 0.750 | 1.000 | 60.37 | 37.99 RSS | 40.17s |
| `JANGQ-AI/Laguna-S-2.1-JANG_2L` | `47e1ba4eef24807751ed229ceeaff293a2bc53d2` | 0.417 | 0.417 | 0.417 | 47.79 | 45.75 | 60.53s |
| `mlx-community/Laguna-S-2.1-oQ2e` | `777afdcd509a4a2ac9007bb405ea1f97d6b60912` | 1.000 | 1.000 | 1.000 | 40.85 | 37.77 | 87.38s |
| `unsloth/Laguna-S-2.1-GGUF` `UD-IQ1_M` | `17bf31a6d627ed136f7d1f403cb692ae643debe4` | 0.792 | 0.750 | 0.833 | 57.34 | 34.22 RSS | 78.41s |
| `mlx-community/Laguna-S-2.1-oQ3e` | `b0a05345ef4ee549a2c1e7b27dbbf8aec8c1b0b3` | 0.625 | 0.417 | 0.833 | 48.58 | 50.69 | 84.79s |
| `poolside/Laguna-S-2.1-NVFP4-mlx` | `9664772ddf25ea938bbc380b26f7e7110f9f6521` | 0.875 | 0.750 | 1.000 | 7.25 | 73.47 | 301.77s |

The pipenetwork 2-bit and oQ2e runs both passed every hidden assertion: 19/19 generation checks and 19/19 agentic checks. Pipenetwork's conversion was considerably faster in both the task aggregate and fixed decode. Q2_K_XL passed all 19 agentic checks and 13/19 generation checks; its medium generation returned tuples instead of the required lists. IQ1_M passed 13/19 generation checks and 15/19 agentic checks, also returning tuples in medium generation and inverting a unit-order condition in the medium agent task. A repeated IQ1_M run produced the same task scores. oQ3e made several exact-format and implementation errors. The official, testing-only NVFP4 model failed six of eight medium-generation checks because it returned tuples where the specification required lists. Its other five tasks passed.

Pipenetwork's repository ships `laguna.py` because MLX-LM 0.31.3 does not include this architecture. The harness imports that reviewed file under `mlx_lm.models.laguna` from the pinned snapshot and records SHA-256 `89b9b3f95ed3f35b3e167bbe7af01bbd36fc3a589d451d171d87e01df682d20c`. It also enables Transformers' Mistral regex correction. No installed package files are patched.

JANG_2L's first forward pass failed in the pinned upstream runtime because its quantization predicate applied the top-level 8-bit width to every layer. The checkpoint defines 527 per-module overrides ranging from 2 to 8 bits. The harness wraps that one initialization call so MLX receives each module's config dictionary, then restores the original MLX function. With that adapter, the model runs at the expected speed, but its 0.417 suite score does not support recommending it for this workload.

During testing, the Hub advanced the oQ2e repository from the canonical run's `777afd...` revision to `830f68...`. The identities of all seven safetensor blobs are unchanged, as are the inference files. The newer revision completes the repository metadata files. The 256K result records that revision.

Both conventional community MLX 4-bit conversions failed before inference in mlx-vlm 0.6.6. Vontra's checkpoint has a gate quantization shape mismatch. Pipenetwork's checkpoint supplies 141 router parameters that the runtime does not instantiate. The export records the exact exceptions as failure rows. The official Q4_K_M GGUF loads through Poolside's llama.cpp branch.

Canonical artifacts:

- `results/20260722T010704Z-mlx-community--Laguna-S-2.1-oQ2e/`
- `results/20260722T033304Z-pipenetwork--Laguna-S-2.1-MLX-2bit/`
- `results/20260722T034647Z-unsloth--Laguna-S-2.1-GGUF:UD-Q2_K_XL/`
- `results/20260722T035828Z-JANGQ-AI--Laguna-S-2.1-JANG_2L/`
- `results/20260722T032111Z-unsloth--Laguna-S-2.1-GGUF:UD-IQ1_M/`
- `results/20260722T013118Z-mlx-community--Laguna-S-2.1-oQ3e/`
- `results/20260722T005530Z-poolside--Laguna-S-2.1-NVFP4-mlx/`
- `results/comparison.md`
- `results/laguna_s21_full_results.csv`

## Standardized quant performance profile

| Quant | 16K prefill tok/s | Fixed 256-token decode tok/s | Profile peak GB |
|---|---:|---:|---:|
| Pipenetwork mixed 2-bit | 1247.17 | 68.49 | 39.90 MLX |
| Q2_K_XL GGUF | 618.25 | 55.93 | 37.99 RSS |
| JANG_2L | 1259.03 | 49.29 | 46.62 MLX |
| IQ1_M GGUF | 754.54 | 62.68 | 34.22 RSS |
| oQ2e | 1613.07 | 55.06 | 38.46 MLX |
| oQ3e | 1275.11 | 60.83 | 51.47 MLX |
| Official NVFP4 MLX | 1646.40 | 26.16 | 74.21 MLX |

The standardized profile measures prefill and a fixed-length decode separately from task-dependent early stopping. The MLX decode cases use a 1K prompt; llama-bench reports its 256-token generation test separately from the 16K prefill test. The memory methods also differ: MLX reports its allocator peak, while the GGUF row uses the process maximum RSS reported by macOS. This explains why the older NVFP4 task aggregate looked especially slow: overhead dominated its short, variable generations.

## Context-length sweep on oQ2e

| Prompt tokens | Prefill tok/s | Decode tok/s | Peak GB | Retrieval |
|---:|---:|---:|---:|---:|
| 248 | 581.28 | 62.93 | 36.55 | pass |
| 1,016 | 1411.42 | 60.86 | 37.23 | pass |
| 4,088 | 1791.79 | 58.57 | 37.79 | pass |
| 16,376 | 1613.07 | 52.48 | 38.46 | pass |
| 65,528 | 1088.75 | 32.03 | 41.19 | pass |
| 131,064 | 771.80 | 23.48 | 45.09 | pass |
| 262,136 | 566.31 | 12.16 | 52.87 | pass |

The retrieval check places a deterministic key before the filler, then requests it at the end. This is a long-context smoke test, not a substitute for RULER or a semantic long-context evaluation. Use 64K or less when latency matters. Contexts of 128K and 256K fit in memory, but carry substantial decode penalties. We did not exercise the checkpoint's optional 1M configuration.

## Sampling search on oQ2e

| Temperature | Top-p | Overall | Generation | Agentic | Weighted generation tok/s |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 1.00 | 1.000 | 1.000 | 1.000 | 40.85 |
| 0.2 | 0.95 | 0.967 | 0.933 | 1.000 | 55.19 |
| 0.7 | 0.95 | 0.950 | 0.900 | 1.000 | 54.62 |

The throughput difference is run-to-run and workload noise, not a useful sampling speedup. The fixed 256-token microbenchmark measured 55.06, 55.79, and 55.44 tok/s respectively. Greedy decoding is the default for benchmarks and coding. Use 0.7/0.95 when you want more varied output, not better performance.

## KV cache and prefill search at 16K

| Configuration | Prefill tok/s | Decode tok/s | Peak GB | Retrieval |
|---|---:|---:|---:|---:|
| Default KV, prefill step 2048 | 1613.07 | 52.48 | 38.46 | pass |
| TurboQuant 3.5-bit KV | 1261.83 | 44.99 | 37.87 | pass |
| Fixed KV window 4096 | 1335.35 | 51.03 | 38.46 | pass |
| Prefill step 512 | 996.30 | 52.13 | 37.80 | pass |
| Prefill step 8192 | 1103.14 | 51.36 | 40.96 | pass |

Uniform 4-bit and 8-bit KV quantization both raise `NotImplementedError: RotatingKVCache Quantization NYI`. TurboQuant works, but saving 0.59 GB reduced prefill throughput by 21.8% and decode throughput by 14.3% in this test. The default 2048-token prefill step performed best.

## Engine bake-off

| Engine/path | Model or quant | Measured result | Decision |
|---|---|---:|---|
| MLX-LM 0.31.3, pinned custom loader | Pipenetwork mixed 2-bit | 63.86 weighted tok/s; 68.49 tok/s fixed decode; 38/38 assertions | Fastest tested high-quality path |
| mlx-vlm 0.6.6, in process | oQ2e | 40.85 weighted tok/s across all generation tiers; 55.06 tok/s fixed decode | Conservative stock-loader path |
| oMLX 0.5.2, OpenAI server | oQ2e | 53.9 tok/s warm medium decode; cached agent tiers took 63.87s total versus 53.48s in the original direct run | Useful for concurrent serving, but not faster from start to finish for this serial suite |
| Poolside llama.cpp `laguna` branch, Metal | Q4_K_M GGUF | 1059.05 prompt tok/s at 512 tokens; 39.68 generation tok/s at 256 tokens | Slower decode than oQ2e through MLX |
| Poolside llama.cpp DFlash recipe | Q4_K_M plus DFlash draft | 23.0 generation tok/s on the coding probe | Draft overhead/acceptance made it slower on this Mac |

The helper still defaults to direct mlx-vlm and oQ2e because that path does not execute a repository-supplied model definition. The benchmark CLI now supports the faster conversion through an explicit `mlx-lm-custom` engine. Experimental MLX prompt-cache reuse remains off. Individual cached turns were faster, but caching changed the greedy agent trajectory: the large task took 14 turns instead of 10. A controlled engine comparison needs exact output parity between the cold and cached paths.

These engine results apply to this machine and workload. oMLX remains installed in an isolated cache environment for future concurrent serving tests. The Poolside llama.cpp branch and GGUFs also remain cached for more profiling.

## Recommended local configuration

- Model: `pipenetwork/Laguna-S-2.1-MLX-2bit` for best tested speed and quality; oQ2e for a stock loader
- Runtime: in-process MLX-LM 0.31.3 with the pinned loader; mlx-vlm 0.6.6 for oQ2e
- Sampling: temperature 0, top-p 1, fixed seed for evaluation
- Prefill step: 2048
- KV cache: default, unquantized; no prompt-cache reuse for controlled agent tests
- Context: up to 64K for a better latency balance; 128K and 256K are validated when needed

## Local storage

The retained caches now also include pipenetwork's 35 GiB mixed 2-bit conversion and Unsloth's 33 GiB IQ1_M GGUF, alongside oQ2e (34 GiB), oQ3e (46 GiB), official NVFP4 MLX (67 GiB), official GGUF (72 GiB), and both broken community 4-bit uploads (62 GiB each). We keep broken checkpoints only to make compatibility failures reproducible; contributors do not need them.
