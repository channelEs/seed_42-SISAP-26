#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-chnsw}"
DATASET="${DATASET:-nq}"

RESULTS_DIR="${ROOT_DIR}/results"
ROUND_DIR="${RESULTS_DIR}/static_exps_04"
CSV_DIR="${ROUND_DIR}/csv"
LOG_DIR="${ROUND_DIR}/logs"
FIG_DIR="${ROUND_DIR}/figures"
FIG_OUT_DIR="${ROOT_DIR}/figures/04_static_exps_analysis"

CONFIG_FOLDER="static_exps_04_search"
RESULT_CSV_NAME="${CONFIG_FOLDER}.csv"

mkdir -p "${CSV_DIR}" "${LOG_DIR}" "${FIG_DIR}" "${FIG_OUT_DIR}"

echo "[INFO] Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"

rm -f "${RESULTS_DIR}/${RESULT_CSV_NAME}"

echo "[RUN] Starting ${CONFIG_FOLDER} on dataset=${DATASET}"
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
  --params "${CONFIG_FOLDER}" \
  | tee "${LOG_DIR}/${CONFIG_FOLDER}.log"

if [[ ! -f "${RESULTS_DIR}/${RESULT_CSV_NAME}" ]]; then
  echo "[ERROR] Missing expected output CSV: ${RESULTS_DIR}/${RESULT_CSV_NAME}" >&2
  exit 1
fi

cp -f "${RESULTS_DIR}/${RESULT_CSV_NAME}" "${CSV_DIR}/${RESULT_CSV_NAME}"

echo "[PLOT] Generating round-04 plots"
python3 "${ROOT_DIR}/scripts/static_exps_04_plots.py" \
  --csv "${CSV_DIR}/${RESULT_CSV_NAME}" \
  --outdir "${FIG_OUT_DIR}"

cp -f "${FIG_OUT_DIR}"/*.png "${FIG_DIR}/"

echo "[DONE] Round-04 search experiments completed"
echo "[DONE] CSV: ${CSV_DIR}/${RESULT_CSV_NAME}"
echo "[DONE] Logs: ${LOG_DIR}/${CONFIG_FOLDER}.log"
echo "[DONE] Figures: ${FIG_OUT_DIR}"