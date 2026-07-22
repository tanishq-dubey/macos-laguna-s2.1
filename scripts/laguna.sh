#!/usr/bin/env bash
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LAGUNA_MODEL="${LAGUNA_MODEL:-mlx-community/Laguna-S-2.1-oQ2e}"
LAGUNA_FILE="${LAGUNA_FILE:-}"
LAGUNA_HOST="${LAGUNA_HOST:-127.0.0.1}"
LAGUNA_PORT="${LAGUNA_PORT:-8080}"

usage() {
  printf '%s\n' \
    "Usage: scripts/laguna.sh <command> [arguments]" \
    "" \
    "Commands:" \
    "  setup       Create .venv and install the harness/runtime" \
    "  download    Download the recommended quant into the Hugging Face cache" \
    "  prompt ...  Run one prompt, for example: prompt 'Write a Python trie'" \
    "  chat        Start an interactive terminal chat" \
    "  server      Start an OpenAI-compatible server on 127.0.0.1:8080" \
    "  bench       Run the six quality tasks and short quant performance profile" \
    "  community   Run the full context/hyperparameter sweep and export the CSV" \
    "" \
    "Override the model with LAGUNA_MODEL. Set LAGUNA_FILE for one file in a multi-quant repository." \
    "Override server binding with LAGUNA_HOST or LAGUNA_PORT."
}

setup() {
  cd "${PROJECT_DIR}"
  if ! command -v uv >/dev/null 2>&1; then
    printf '%s\n' "uv is required: https://docs.astral.sh/uv/getting-started/installation/" >&2
    exit 2
  fi
  uv sync --extra dev --python 3.13 --locked
}

ensure_setup() {
  setup
}

download() {
  ensure_setup
  cd "${PROJECT_DIR}"
  if [[ -n "${LAGUNA_FILE}" ]]; then
    uv run --frozen hf download "${LAGUNA_MODEL}" "${LAGUNA_FILE}"
  else
    uv run --frozen hf download "${LAGUNA_MODEL}"
  fi
}

command_name="${1:-}"
if [[ -z "${command_name}" ]]; then
  usage
  exit 0
fi
shift

case "${command_name}" in
  setup)
    setup
    ;;
  download)
    download
    ;;
  prompt)
    ensure_setup
    if [[ $# -eq 0 ]]; then
      printf '%s\n' "prompt requires text" >&2
      exit 2
    fi
    cd "${PROJECT_DIR}"
    uv run --frozen mlx_vlm.generate --model "${LAGUNA_MODEL}" --prompt "$*" --max-tokens 1024 --temperature 0
    ;;
  chat)
    ensure_setup
    cd "${PROJECT_DIR}"
    uv run --frozen mlx_vlm.chat --model "${LAGUNA_MODEL}"
    ;;
  server)
    ensure_setup
    cd "${PROJECT_DIR}"
    exec uv run --frozen mlx_vlm.server --model "${LAGUNA_MODEL}" --host "${LAGUNA_HOST}" --port "${LAGUNA_PORT}"
    ;;
  bench)
    ensure_setup
    cd "${PROJECT_DIR}"
    uv run --frozen laguna-bench run --model "${LAGUNA_MODEL}" --output results
    uv run --frozen laguna-bench sweep --model "${LAGUNA_MODEL}" --profile quant --output results
    uv run --frozen laguna-bench export --output results
    ;;
  community)
    ensure_setup
    cd "${PROJECT_DIR}"
    uv run --frozen laguna-bench sweep --model "${LAGUNA_MODEL}" --profile full --output results
    uv run --frozen laguna-bench export --output results
    ;;
  help|-h|--help)
    usage
    ;;
  *)
    printf 'Unknown command: %s\n\n' "${command_name}" >&2
    usage >&2
    exit 2
    ;;
esac
