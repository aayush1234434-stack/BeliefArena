#!/usr/bin/env bash
# Repair schema-invalid trials and freeze a validator-clean snapshot.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

if [[ -z "${NVIDIA_API_KEY:-}" ]]; then
  echo "Error: NVIDIA_API_KEY is required to re-collect purged Llama trials." >&2
  exit 1
fi

python3 validate_results.py --purge-invalid || test $? -eq 2
python3 main.py --model meta/llama-3.1-8b-instruct
python3 validate_results.py --write-clean
python3 -m unittest discover -s tests -v
echo "Frozen dataset repaired and validated."
