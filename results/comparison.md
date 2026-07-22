# Laguna S 2.1 local comparison

Latest complete six-task run per model.

| Model | Engine | Score | Gen | Agent | Gen tok/s | Peak GB | Memory metric | Load s |
|---|---|---:|---:|---:|---:|---:|---|---:|
| JANGQ-AI/Laguna-S-2.1-JANG_2L | jang-mlx | 0.417 | 0.417 | 0.417 | 47.79 | 45.75 | mlx_peak | 1.83 |
| JANGQ-AI/Laguna-S-2.1-JANG_4M | jang-mlx | 0.875 | 0.750 | 1.000 | 44.88 | 69.76 | mlx_peak | 4.67 |
| mlx-community/Laguna-S-2.1-oQ2e | mlx-vlm | 1.000 | 1.000 | 1.000 | 40.85 | 37.77 | mlx_peak | 1.50 |
| mlx-community/Laguna-S-2.1-oQ3e | mlx-vlm | 0.625 | 0.417 | 0.833 | 48.58 | 50.69 | mlx_peak | 2.53 |
| mlx-community/Laguna-S-2.1-oQ4e | mlx-vlm | 0.875 | 0.750 | 1.000 | 44.42 | 65.70 | mlx_peak | 5.95 |
| pipenetwork/Laguna-S-2.1-MLX-2bit | mlx-lm-custom | 1.000 | 1.000 | 1.000 | 63.86 | 39.31 | mlx_peak | 1.61 |
| pipenetwork/Laguna-S-2.1-MLX-3bit | mlx-lm-custom | 0.875 | 0.750 | 1.000 | 58.63 | 52.88 | mlx_peak | 2.95 |
| poolside/Laguna-S-2.1-NVFP4-mlx | mlx-vlm | 0.875 | 0.750 | 1.000 | 7.25 | 73.47 | mlx_peak | 9.18 |
| unsloth/Laguna-S-2.1-GGUF:UD-IQ1_M | openai-compatible | 0.792 | 0.750 | 0.833 | 57.34 | 34.22 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-IQ1_S | openai-compatible | 0.658 | 0.317 | 1.000 | 63.00 | 32.48 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-IQ2_M | openai-compatible | 0.625 | 0.417 | 0.833 | 61.15 | 35.74 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-IQ2_XXS | openai-compatible | 0.646 | 0.417 | 0.875 | 62.76 | 35.66 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-IQ3_S | openai-compatible | 0.708 | 0.417 | 1.000 | 56.99 | 46.13 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-IQ3_XXS | openai-compatible | 0.708 | 0.417 | 1.000 | 55.90 | 42.27 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-IQ4_NL | openai-compatible | 0.875 | 0.750 | 1.000 | 50.75 | 55.74 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-IQ4_XS | openai-compatible | 0.875 | 0.750 | 1.000 | 51.04 | 54.64 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-Q2_K_XL | openai-compatible | 0.875 | 0.750 | 1.000 | 60.37 | 37.99 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-Q3_K_M | openai-compatible | 0.875 | 0.750 | 1.000 | 51.27 | 51.34 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-Q3_K_XL | openai-compatible | 0.812 | 0.750 | 0.875 | 50.06 | 51.41 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-Q4_K_S | openai-compatible | 0.875 | 0.750 | 1.000 | 48.17 | 64.91 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-Q4_K_XL | openai-compatible | 0.875 | 0.750 | 1.000 | 45.95 | 69.38 | process_max_rss | 0.00 |
| unsloth/Laguna-S-2.1-GGUF:UD-Q5_K_XL | openai-compatible | 0.875 | 0.750 | 1.000 | 45.62 | 83.05 | process_max_rss | 0.00 |
| vcruz305/Laguna-S-2.1-GGUF:IQ1_S | openai-compatible | 0.042 | 0.000 | 0.083 | 75.14 | 23.16 | process_max_rss | 0.00 |
