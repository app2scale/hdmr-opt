#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# HDMR Optimization - 2D Benchmark Runner (Production-Ready)
#
# Runs a set of 2D benchmark functions with:
#  - Standard HDMR
#  - Adaptive HDMR
#
# Output:
#   results/benchmark_2d/<run_id>/
#     logs/<func>_standard.log
#     logs/<func>_adaptive.log
#     summary.txt
#
# Usage:
#   bash benchmark_2d.sh
#
# Notes:
# - Must be executed from repository root (where "src/" exists).
# - Uses "python -m src.main" to avoid import issues.
# - Bounds are auto-loaded from src/function_ranges.json if --min/--max not given.
# -----------------------------------------------------------------------------

set -euo pipefail
IFS=$'\n\t'

# -------------------------------
# Configuration (override via env)
# -------------------------------
PYTHON_BIN="${PYTHON_BIN:-python}"
RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"

# Benchmark list (edit as needed)
FUNCTIONS=(
  "rastrigin_2d"
  "rosenbrock_2d"
  "ackley_2d"
  "camel16_2d"
  "branin_2d"
)

# Common HDMR params
NUM_SAMPLES="${NUM_SAMPLES:-1000}"
NUM_VARS="${NUM_VARS:-2}"
BASIS_FUNCTION="${BASIS_FUNCTION:-Cosine}"   # Legendre | Cosine
DEGREE="${DEGREE:-7}"
RUNS="${RUNS:-5}"
SEED="${SEED:-42}"

# Adaptive params
ADAPTIVE_MAXITER="${ADAPTIVE_MAXITER:-25}"
ADAPTIVE_K="${ADAPTIVE_K:-100}"             # numClosestPoints
ADAPTIVE_EPSILON="${ADAPTIVE_EPSILON:-0.1}"
ADAPTIVE_CLIP="${ADAPTIVE_CLIP:-0.9}"

# Disable plots (recommended for batch runs on servers)
NO_PLOTS="${NO_PLOTS:-1}"                   # 1 => add --noPlots

# Output dir
OUT_DIR="results/benchmark_2d/${RUN_ID}"
LOG_DIR="${OUT_DIR}/logs"

# -------------------------------
# Helpers
# -------------------------------
die() { echo "ERROR: $*" >&2; exit 1; }

require_repo_root() {
  [[ -d "src" ]] || die "Run this script from repository root (missing ./src)."
  [[ -f "src/main.py" ]] || die "Missing ./src/main.py. Are you in the correct directory?"
}

check_python() {
  command -v "${PYTHON_BIN}" >/dev/null 2>&1 || die "Python not found: ${PYTHON_BIN}"
  "${PYTHON_BIN}" -c "import sys; print(sys.version)" >/dev/null
}

make_dirs() {
  mkdir -p "${LOG_DIR}"
}

common_args() {
  local args=(
    -m src.main
    --numSamples "${NUM_SAMPLES}"
    --numVariables "${NUM_VARS}"
    --basisFunction "${BASIS_FUNCTION}"
    --degree "${DEGREE}"
    --numberOfRuns "${RUNS}"
    --seed "${SEED}"
    --disp
  )
  if [[ "${NO_PLOTS}" == "1" ]]; then
    args+=(--noPlots)
  fi
  printf '%s\n' "${args[@]}"
}

run_one() {
  local func="$1"
  local mode="$2"     # standard | adaptive
  local log_file="$3"

  echo "------------------------------------------------------------"
  echo "Function : ${func}"
  echo "Mode     : ${mode}"
  echo "Log      : ${log_file}"
  echo "------------------------------------------------------------"

  local args
  mapfile -t args < <(common_args)
  args+=(--function "${func}")

  if [[ "${mode}" == "adaptive" ]]; then
    args+=(
      --adaptive
      --maxiter "${ADAPTIVE_MAXITER}"
      --numClosestPoints "${ADAPTIVE_K}"
      --epsilon "${ADAPTIVE_EPSILON}"
      --clip "${ADAPTIVE_CLIP}"
    )
  fi

  # Run and tee output to log
  # Use env to ensure we always run with repository root on sys.path.
  "${PYTHON_BIN}" "${args[@]}" 2>&1 | tee "${log_file}"
}

extract_last_result_line() {
  # best-effort: grep last "Best f(x*)" style lines if present; else no-op
  local log_file="$1"
  # Try multiple patterns; adjust if your CLI prints differently
  grep -E "Best f\\(x\\*\\)|Mean f\\(x\\*\\)|Solution:|Objective:" "${log_file}" 2>/dev/null | tail -n 5 || true
}

# -------------------------------
# Main
# -------------------------------
require_repo_root
check_python
make_dirs

echo "======================================"
echo "HDMR 2D Benchmark Tests"
echo "Run ID: ${RUN_ID}"
echo "Output: ${OUT_DIR}"
echo "Python: $(${PYTHON_BIN} -V 2>&1)"
echo "======================================"
echo ""

# Save configuration snapshot
{
  echo "HDMR 2D Benchmark Runner - Summary"
  echo "Run ID: ${RUN_ID}"
  echo "Timestamp: $(date -Is)"
  echo ""
  echo "[Config]"
  echo "NUM_SAMPLES=${NUM_SAMPLES}"
  echo "NUM_VARS=${NUM_VARS}"
  echo "BASIS_FUNCTION=${BASIS_FUNCTION}"
  echo "DEGREE=${DEGREE}"
  echo "RUNS=${RUNS}"
  echo "SEED=${SEED}"
  echo "NO_PLOTS=${NO_PLOTS}"
  echo ""
  echo "[Adaptive Config]"
  echo "ADAPTIVE_MAXITER=${ADAPTIVE_MAXITER}"
  echo "ADAPTIVE_K=${ADAPTIVE_K}"
  echo "ADAPTIVE_EPSILON=${ADAPTIVE_EPSILON}"
  echo "ADAPTIVE_CLIP=${ADAPTIVE_CLIP}"
  echo ""
  echo "[Functions]"
  printf '%s\n' "${FUNCTIONS[@]}"
  echo ""
} > "${OUT_DIR}/summary.txt"

# Execute benchmarks
for func in "${FUNCTIONS[@]}"; do
  std_log="${LOG_DIR}/${func}_standard.log"
  adp_log="${LOG_DIR}/${func}_adaptive.log"

  echo ""
  echo "======================================"
  echo "Testing: ${func}"
  echo "======================================"

  # Standard
  run_one "${func}" "standard" "${std_log}"
  {
    echo ""
    echo ">>> ${func} [standard] key lines:"
    extract_last_result_line "${std_log}"
    echo ""
  } >> "${OUT_DIR}/summary.txt"

  # Adaptive
  run_one "${func}" "adaptive" "${adp_log}"
  {
    echo ""
    echo ">>> ${func} [adaptive] key lines:"
    extract_last_result_line "${adp_log}"
    echo ""
  } >> "${OUT_DIR}/summary.txt"
done

echo ""
echo "======================================"
echo "All tests completed successfully."
echo "Logs   : ${LOG_DIR}"
echo "Summary: ${OUT_DIR}/summary.txt"
echo "======================================"

