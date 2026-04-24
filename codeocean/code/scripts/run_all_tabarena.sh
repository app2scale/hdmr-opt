#!/usr/bin/env bash
# =============================================================================
# run_all_tabarena.sh — v2
# Mevcut 3 script üzerinde çalışır:
#   tabarena_hdmr_xgb.py
#   tabarena_optuna_xgb.py
#   tabarena_random_search_xgb.py
#
# Kullanım:
#   bash run_all_tabarena.sh [method] [dataset_group]
#
# method: all | hdmr | adaptive200 | adaptive600 | rs | optuna | smoke
# dataset_group: regression (default) | all18
#
# Örnekler:
#   bash run_all_tabarena.sh smoke
#   bash run_all_tabarena.sh hdmr regression
#   bash run_all_tabarena.sh all regression
#   MAX_PARALLEL=4 bash run_all_tabarena.sh all regression
# =============================================================================

set -euo pipefail

# Resolve project root (parent of this script's directory) so the script can
# be invoked from any working directory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

METHOD=${1:-"all"}
DATASET_GROUP=${2:-"regression"}
MAX_PARALLEL=${MAX_PARALLEL:-3}
N_FOLDS=${N_FOLDS:-8}
SEED=${SEED:-42}
INNER_CV=${INNER_CV:-1}
LOG_DIR="/results/logs"
RESULTS_DIR="/results/tabarena"


mkdir -p "$LOG_DIR" "$RESULTS_DIR"

REGRESSION_DIDS=(46954 46917 46931 46904 46907 46942 46934 46961 46928 46923)
CLASSIFICATION_DIDS=(46952 46927 46938 46940 46930 46956 46963 46918)

declare -A DATASET_NAMES=(
    [46954]="QSAR_fish_toxicity"
    [46917]="concrete_compressive_strength"
    [46931]="healthcare_insurance_expenses"
    [46904]="airfoil_self_noise"
    [46907]="Fiat500_used"
    [46942]="miami_housing"
    [46934]="houses"
    [46961]="superconductivity"
    [46928]="Food_Delivery_Time"
    [46923]="diamonds"
    [46952]="qsar-biodeg"
    [46927]="Fitness_Club"
    [46938]="Is-this-a-good-customer"
    [46940]="Marketing_Campaign"
    [46930]="hazelnut-contaminant"
    [46956]="seismic-bumps"
    [46963]="website_phishing"
    [46918]="credit-g"
)

if [ "$DATASET_GROUP" = "all18" ]; then
    DIDS=("${REGRESSION_DIDS[@]}" "${CLASSIFICATION_DIDS[@]}")
else
    DIDS=("${REGRESSION_DIDS[@]}")
fi

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'

ts()  { date '+%H:%M:%S'; }
log() { echo -e "${CYAN}[$(ts)]${NC} $*"; }
ok()  { echo -e "${GREEN}[$(ts)][OK]${NC} $*"; }
err() { echo -e "${RED}[$(ts)][ERR]${NC} $*"; }
hdr() { echo -e "\n${BLUE}=== $* ===${NC}"; }

PIDS=()
JOB_NAMES=()
FAILED_JOBS=()

wait_for_slot() {
    while [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ]; do
        local new_pids=(); local new_names=()
        for i in "${!PIDS[@]}"; do
            if ! kill -0 "${PIDS[$i]}" 2>/dev/null; then
                if wait "${PIDS[$i]}"; then
                    ok "Completed: ${JOB_NAMES[$i]}"
                else
                    err "FAILED: ${JOB_NAMES[$i]}"
                    FAILED_JOBS+=("${JOB_NAMES[$i]}")
                fi
            else
                new_pids+=("${PIDS[$i]}")
                new_names+=("${JOB_NAMES[$i]}")
            fi
        done
        PIDS=("${new_pids[@]+"${new_pids[@]}"}")
        JOB_NAMES=("${new_names[@]+"${new_names[@]}"}")
        [ "${#PIDS[@]}" -ge "$MAX_PARALLEL" ] && sleep 3
    done
}

wait_all() {
    log "Waiting for ${#PIDS[@]} active jobs..."
    for i in "${!PIDS[@]}"; do
        if wait "${PIDS[$i]}"; then
            ok "Completed: ${JOB_NAMES[$i]}"
        else
            err "FAILED: ${JOB_NAMES[$i]}"
            FAILED_JOBS+=("${JOB_NAMES[$i]}")
        fi
    done
    PIDS=(); JOB_NAMES=()
}

launch() {
    local script="$1" did="$2" tag="$3"
    shift 3
    local name="${DATASET_NAMES[$did]:-unknown}"
    local logfile="${LOG_DIR}/full_${tag}_${did}.log"
    local job="${tag}_${name:0:25}"

    # Resume: tamamlanmis job'lari atla
    if [ -f "$logfile" ] && grep -q "Experiment complete" "$logfile" 2>/dev/null; then
        log "SKIP (already done): $job"
        return
    fi

    wait_for_slot
    log "Launch ${YELLOW}${job}${NC} -> $(basename $logfile)"

    env DATASET_SOURCE="openml:${did}" \
        N_FOLDS="$N_FOLDS" \
        SEED="$SEED" \
        INNER_CV_FOLDS="$INNER_CV" \
        "$@" \
        nohup python "$SCRIPT_DIR/$script" > "$logfile" 2>&1 &

    PIDS+=($!)
    JOB_NAMES+=("$job")
}

run_hdmr_200() {
    hdr "HDMR-200 (Standard, 200 evals/fold)"
    for DID in "${DIDS[@]}"; do
        launch tabarena_hdmr_xgb.py "$DID" "hdmr200" \
            HDMR_SAMPLES=200 HDMR_ADAPTIVE=0
    done
    wait_all
}

run_adaptive_200() {
    hdr "A-HDMR-200 (T=2 x 100 samples = 200 evals/fold)"
    for DID in "${DIDS[@]}"; do
        launch tabarena_hdmr_xgb.py "$DID" "ahdmr200" \
            HDMR_SAMPLES=100 HDMR_ADAPTIVE=1 HDMR_MAXITER=2
    done
    wait_all
}

run_adaptive_600() {
    hdr "A-HDMR-600 (T=3 x 200 samples = 600 evals/fold)"
    for DID in "${DIDS[@]}"; do
        launch tabarena_hdmr_xgb.py "$DID" "ahdmr600" \
            HDMR_SAMPLES=200 HDMR_ADAPTIVE=1 HDMR_MAXITER=3
    done
    wait_all
}

run_rs_200() {
    hdr "Random Search-200"
    for DID in "${DIDS[@]}"; do
        launch tabarena_random_search_xgb.py "$DID" "rs200" \
            RS_SAMPLES=200
    done
    wait_all
}

run_optuna_200() {
    hdr "Optuna TPE-200"
    for DID in "${DIDS[@]}"; do
        launch tabarena_optuna_xgb.py "$DID" "optuna200" \
            OPTUNA_TRIALS=200
    done
    wait_all
}

run_smoke() {
    hdr "SMOKE TEST (2 fold, 20 sample, 3 dataset)"
    for DID in 46917 46904 46954; do
        local name="${DATASET_NAMES[$DID]}"
        log "HDMR smoke: $name"
        DATASET_SOURCE="openml:$DID" HDMR_SAMPLES=20 HDMR_ADAPTIVE=0 \
            N_FOLDS=2 SEED=42 INNER_CV_FOLDS=1 \
            python "$SCRIPT_DIR/tabarena_hdmr_xgb.py" 2>&1 | grep -E "RMSE|DONE|ERROR|complete|Loading|Dataset" || true
        log "RS smoke: $name"
        DATASET_SOURCE="openml:$DID" RS_SAMPLES=20 \
            N_FOLDS=2 SEED=42 \
            python "$SCRIPT_DIR/tabarena_random_search_xgb.py" 2>&1 | grep -E "RMSE|DONE|ERROR|complete|Loading|Dataset" || true
        log "Optuna smoke: $name"
        DATASET_SOURCE="openml:$DID" OPTUNA_TRIALS=20 \
            N_FOLDS=2 SEED=42 \
            python "$SCRIPT_DIR/tabarena_optuna_xgb.py" 2>&1 | grep -E "RMSE|DONE|ERROR|complete|Loading|Dataset" || true
    done
    ok "Smoke tests complete."
}

# Header
echo -e "${BLUE}"
echo "==================================================="
echo "  TabArena XGBoost HPO Benchmark  -  Full Runner  "
echo "  HDMR-200 | A-HDMR-200 | RS-200 | Optuna-200    "
echo "==================================================="
echo -e "${NC}"
log "METHOD       = ${YELLOW}$METHOD${NC}"
log "DATASETS     = ${YELLOW}${#DIDS[@]} datasets${NC} ($DATASET_GROUP)"
log "MAX_PARALLEL = $MAX_PARALLEL  |  N_FOLDS = $N_FOLDS  |  SEED = $SEED"
log "INNER_CV     = $INNER_CV  (1=80/20 holdout, >1 = K-fold CV)"
echo ""

case "$METHOD" in
    smoke)       run_smoke ;;
    hdmr)        run_hdmr_200 ;;
    adaptive200) run_adaptive_200 ;;
    adaptive600) run_adaptive_600 ;;
    rs)          run_rs_200 ;;
    optuna)      run_optuna_200 ;;
    all)
        run_hdmr_200
        run_rs_200
        run_optuna_200
        run_adaptive_200
        run_adaptive_600
        ;;
    *)
        err "Unknown method: '$METHOD'"
        echo "Usage: $0 [smoke|hdmr|adaptive200|adaptive600|rs|optuna|all] [regression|all18]"
        exit 1 ;;
esac

echo ""
if [ "${#FAILED_JOBS[@]}" -eq 0 ]; then
    ok "All jobs completed successfully!"
else
    err "${#FAILED_JOBS[@]} jobs failed:"
    for j in "${FAILED_JOBS[@]}"; do echo "  x $j"; done
    exit 1
fi

echo ""
log "Logs    : $LOG_DIR/full_*.log"
log "Results : $RESULTS_DIR/"
echo "Analyze : python generate_reports.py"