#!/bin/bash

set -euo pipefail

SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
cd "$SCRIPT_DIR"

usage() {
  cat <<'EOF'
Usage:
  ./run.sh collect <xpu|cuda> <name> <hardware-label> <peak-tflops> <bandwidth-gbs> [--unitrace]
  ./run.sh compare <xpu-name> <target-name> [unitrace-json] [--allow-profiler-fallback]

Environment:
  UNITRACE_BIN  Path to unitrace for XPU --unitrace collection.
EOF
}

collect() {
  if [[ $# -lt 5 || $# -gt 6 ]]; then
    usage
    exit 2
  fi

  local device=$1
  local name=$2
  local hardware_label=$3
  local peak_tflops=$4
  local bandwidth_gbs=$5
  local source_mode=${6:-}
  local output_dir="artifacts/molmoact2/$name"
  local collect_args=(
    python collect_model.py
    --device "$device"
    --output-dir "$output_dir"
    --hardware-label "$hardware_label"
    --peak-tflops "$peak_tflops"
    --memory-bandwidth-gbs "$bandwidth_gbs"
  )

  if [[ "$source_mode" == "--unitrace" ]]; then
    if [[ "$device" != "xpu" ]]; then
      echo "--unitrace is supported only for XPU" >&2
      exit 2
    fi
    local unitrace_bin=${UNITRACE_BIN:-/workspaces/pytorch/pti-gpu/tools/unitrace/build/unitrace}
    if [[ ! -x "$unitrace_bin" ]]; then
      echo "unitrace executable not found: $unitrace_bin" >&2
      exit 2
    fi
    "$unitrace_bin" \
      --chrome-kernel-logging \
      --start-paused \
      --output-dir-path "$output_dir" \
      "${collect_args[@]}" \
      --unitrace
  elif [[ -n "$source_mode" ]]; then
    echo "unsupported collection option: $source_mode" >&2
    exit 2
  else
    "${collect_args[@]}"
  fi

  python -m perf_analysis \
    "$output_dir/collection.json" \
    --output-dir "$output_dir"
}

compare() {
  if [[ $# -lt 2 || $# -gt 4 ]]; then
    usage
    exit 2
  fi

  local xpu_name=$1
  local target_name=$2
  shift 2
  local unitrace_path=""
  local allow_fallback=false
  local output_dir="artifacts/molmoact2/comparison-${xpu_name}-vs-${target_name}"
  local compare_args=(
    python -m perf_analysis.compare
    "artifacts/molmoact2/$xpu_name/collection.json"
    "artifacts/molmoact2/$target_name/analysis.json"
    --reference-name "$xpu_name"
    --target-name "$target_name"
    --output-dir "$output_dir"
  )
  while [[ $# -gt 0 ]]; do
    if [[ "$1" == "--allow-profiler-fallback" ]]; then
      allow_fallback=true
    elif [[ -z "$unitrace_path" ]]; then
      unitrace_path=$1
    else
      echo "unexpected compare argument: $1" >&2
      exit 2
    fi
    shift
  done
  if [[ -n "$unitrace_path" ]]; then
    compare_args+=(--reference-unitrace "$unitrace_path")
  fi
  if [[ "$allow_fallback" == true ]]; then
    compare_args+=(--allow-reference-profiler-fallback)
  fi
  "${compare_args[@]}"
}

if [[ $# -lt 1 ]]; then
  usage
  exit 2
fi

command_name=$1
shift
case "$command_name" in
  collect)
    collect "$@"
    ;;
  compare)
    compare "$@"
    ;;
  *)
    usage
    exit 2
    ;;
esac