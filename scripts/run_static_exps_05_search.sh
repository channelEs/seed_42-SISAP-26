#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-chnsw}"
DATASET="${DATASET:-nq}"

RESULTS_DIR="${ROOT_DIR}/results"
ROUND_DIR="${RESULTS_DIR}/static_exps_05_search"
CSV_DIR="${ROUND_DIR}/csv"
LOG_DIR="${ROUND_DIR}/logs"

CONFIG_FOLDER="static_exps_05_search"
RESULT_CSV_NAME="${CONFIG_FOLDER}.csv"

mkdir -p "${CSV_DIR}" "${LOG_DIR}"

echo "[INFO] Building Docker image: ${IMAGE_NAME}"
docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"

rm -f "${RESULTS_DIR}/${RESULT_CSV_NAME}"

echo "[RUN] Starting ${CONFIG_FOLDER} on dataset=${DATASET}"
echo "[RUN] The engine processes all JSON files in ${CONFIG_FOLDER} sequentially."
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

echo "[DONE] Round-05 search sweep completed"
echo "[DONE] CSV: ${CSV_DIR}/${RESULT_CSV_NAME}"
echo "[DONE] Logs: ${LOG_DIR}/${CONFIG_FOLDER}.log"
