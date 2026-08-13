#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-chnsw}"
DATASET="${DATASET:-nq}"

RESULTS_DIR="${ROOT_DIR}/results"
ROUND3_DIR="${RESULTS_DIR}/static_exps_03"
CSV_DIR="${ROUND3_DIR}/csv"
LOG_DIR="${ROUND3_DIR}/logs"
FIG_DIR="${ROUND3_DIR}/figures"
FIG_ROUND_DIR="${ROOT_DIR}/figures/03_static_exps_analysis"

K_CONFIG="static_exps_03_k"
SEARCH_CONFIG="static_exps_03_search"
K_CSV_NAME="${K_CONFIG}.csv"
SEARCH_CSV_NAME="${SEARCH_CONFIG}.csv"

mkdir -p "${CSV_DIR}" "${LOG_DIR}" "${FIG_DIR}" "${FIG_ROUND_DIR}"

echo "[INFO] Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"

run_experiment() {
  local config_folder="$1"
  local log_file="$2"

  echo "[RUN] Starting ${config_folder}"
  docker run \
    --rm \
    --cpus=8 \
    --memory=24g \
    --memory-swap=24g \
    --memory-swappiness=0 \
    --volume "${ROOT_DIR}/data:/sisap2026/data:ro" \
    --volume "${ROOT_DIR}/config:/sisap2026/config:ro" \
    --volume "${ROOT_DIR}/results:/sisap2026/results:rw" \
    "${IMAGE_NAME}" \
    --task task3 \
    --dataset "${DATASET}" \
    --params "${config_folder}" \
    | tee "${log_file}"
}

rm -f "${RESULTS_DIR}/${K_CSV_NAME}" "${RESULTS_DIR}/${SEARCH_CSV_NAME}"

# run_experiment "${K_CONFIG}" "${LOG_DIR}/${K_CONFIG}.log"
run_experiment "${SEARCH_CONFIG}" "${LOG_DIR}/${SEARCH_CONFIG}.log"

if [[ ! -f "${RESULTS_DIR}/${K_CSV_NAME}" ]]; then
  echo "[ERROR] Missing expected output CSV: ${RESULTS_DIR}/${K_CSV_NAME}" >&2
  exit 1
fi

if [[ ! -f "${RESULTS_DIR}/${SEARCH_CSV_NAME}" ]]; then
  echo "[ERROR] Missing expected output CSV: ${RESULTS_DIR}/${SEARCH_CSV_NAME}" >&2
  exit 1
fi

cp -f "${RESULTS_DIR}/${K_CSV_NAME}" "${CSV_DIR}/${K_CSV_NAME}"
cp -f "${RESULTS_DIR}/${SEARCH_CSV_NAME}" "${CSV_DIR}/${SEARCH_CSV_NAME}"

echo "[PLOT] Generating round-3 figures"
python3 "${ROOT_DIR}/scripts/static_exps_03_plots.py" \
  --k-csv "${CSV_DIR}/${K_CSV_NAME}" \
  --search-csv "${CSV_DIR}/${SEARCH_CSV_NAME}" \
  --outdir "${FIG_ROUND_DIR}"

cp -f "${FIG_ROUND_DIR}"/*.png "${FIG_DIR}/"

echo "[DONE] Round-3 experiments completed"
echo "[DONE] CSVs: ${CSV_DIR}"
echo "[DONE] Logs: ${LOG_DIR}"
echo "[DONE] Figures: ${FIG_ROUND_DIR}"