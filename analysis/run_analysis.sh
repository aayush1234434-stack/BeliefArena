#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
export MPLCONFIGDIR="${MPLCONFIGDIR:-$ROOT/.mplconfig}"
export MPLBACKEND=Agg
mkdir -p "$MPLCONFIGDIR"
python3 "$ROOT/analysis/persuasion_analysis.py"
