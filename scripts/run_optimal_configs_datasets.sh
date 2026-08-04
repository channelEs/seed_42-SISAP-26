#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IMAGE_NAME="${IMAGE_NAME:-chnsw}"
CONFIG_DIR="${ROOT_DIR}/config/optimal_configs"
DATA_DIR="${ROOT_DIR}/data"
RESULTS_DIR="${ROOT_DIR}/results"
OPT_RUNS_DIR="${RESULTS_DIR}/optimal_runs"
CSV_OUT_DIR="${OPT_RUNS_DIR}/csv"
LOG_OUT_DIR="${OPT_RUNS_DIR}/logs"
H5_OUT_DIR="${OPT_RUNS_DIR}/h5"
SUMMARY_DIR="${OPT_RUNS_DIR}/summary"
TMP_PREFIX="_tmp_optcfg"
FAILED_RUNS_CSV="${SUMMARY_DIR}/failed_runs.csv"
INCOMPAT_DATASETS_CSV="${SUMMARY_DIR}/incompatible_datasets.csv"
RESUME="${RESUME:-1}"

mkdir -p "${CSV_OUT_DIR}" "${LOG_OUT_DIR}" "${H5_OUT_DIR}" "${SUMMARY_DIR}"

echo "dataset,config,reason,log_path" > "${FAILED_RUNS_CSV}"
echo "dataset,reason" > "${INCOMPAT_DATASETS_CSV}"

if [[ ! -d "${CONFIG_DIR}" ]]; then
  echo "[ERROR] Missing config directory: ${CONFIG_DIR}" >&2
  exit 1
fi

mapfile -t DATASETS < <(find "${DATA_DIR}" -maxdepth 1 -type f -name "*.h5" | sort)
if [[ ${#DATASETS[@]} -eq 0 ]]; then
  echo "[ERROR] No .h5 datasets found in ${DATA_DIR}" >&2
  exit 1
fi

mapfile -t CONFIGS < <(find "${CONFIG_DIR}" -maxdepth 1 -type f -name "*.json" | sort)
if [[ ${#CONFIGS[@]} -eq 0 ]]; then
  echo "[ERROR] No .json config files found in ${CONFIG_DIR}" >&2
  exit 1
fi

echo "[INFO] Building Docker image: ${IMAGE_NAME}"
USE_DOCKER=1
if ! command -v docker >/dev/null 2>&1; then
  USE_DOCKER=0
  echo "[WARN] docker not found. Falling back to local binary execution."
else
  if ! docker build -t "${IMAGE_NAME}" "${ROOT_DIR}"; then
    USE_DOCKER=0
    echo "[WARN] Docker build failed. Falling back to local binary execution."
  fi
fi

echo "[INFO] Found ${#DATASETS[@]} datasets and ${#CONFIGS[@]} optimal configs"

check_dataset_compatible() {
  local dataset_file="$1"

  # If h5py is unavailable, skip precheck and let the binary decide.
  if ! python3 - <<'PY' >/dev/null 2>&1
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec('h5py') else 1)
PY
  then
    return 0
  fi

  if python3 - "$dataset_file" <<'PY' >/dev/null 2>&1
import sys
import h5py

path = sys.argv[1]
required = ["train", "otest", "otest/queries", "otest/knns"]

with h5py.File(path, "r") as f:
    for key in required:
        if key not in f:
            raise KeyError(key)
PY
  then
    return 0
  fi

  return 1
}

for dataset_path in "${DATASETS[@]}"; do
  dataset_name="$(basename "${dataset_path}" .h5)"
  echo "[DATASET] ${dataset_name}"

  if ! check_dataset_compatible "${dataset_path}"; then
    echo "  [SKIP] incompatible layout (expected train + otest/queries + otest/knns)"
    echo "${dataset_name},missing_required_groups" >> "${INCOMPAT_DATASETS_CSV}"
    continue
  fi

  for config_path in "${CONFIGS[@]}"; do
    config_name="$(basename "${config_path}" .json)"
    run_tag="${TMP_PREFIX}_${dataset_name}_${config_name}"
    run_cfg_dir="${ROOT_DIR}/config/${run_tag}"
    run_csv_local="${RESULTS_DIR}/${run_tag}.csv"
    run_log="${LOG_OUT_DIR}/${dataset_name}__${config_name}.log"
    final_csv="${CSV_OUT_DIR}/${dataset_name}__${config_name}.csv"

    if [[ "${RESUME}" == "1" && -f "${final_csv}" ]]; then
      echo "  [SKIP] config=${config_name} (already completed)"
      continue
    fi

    rm -rf "${run_cfg_dir}"
    mkdir -p "${run_cfg_dir}"
    cp "${config_path}" "${run_cfg_dir}/config.json"
    rm -f "${run_csv_local}" "${final_csv}"

    echo "  [RUN] config=${config_name}"
    run_status=0
    if [[ ${USE_DOCKER} -eq 1 ]]; then
      docker run \
        --rm \
        --cpus=8 \
        --memory=24g \
        --memory-swap=24g \
        --memory-swappiness=0 \
        --volume "${ROOT_DIR}/data:/sisap2026/data:ro" \
        --volume "${ROOT_DIR}/config:/sisap2026/config:rw" \
        --volume "${ROOT_DIR}/results:/sisap2026/results:rw" \
        "${IMAGE_NAME}" \
        --task task3 \
        --dataset "${dataset_name}" \
        --params "${run_tag}" \
        --output /sisap2026/results/optimal_runs/h5 \
        > "${run_log}" 2>&1 || run_status=$?
    else
      if [[ ! -x "${ROOT_DIR}/build/main" ]]; then
        echo "[INFO] build/main not found. Building locally..."
        (cd "${ROOT_DIR}" && ./build.sh)
      fi

      (
        cd "${ROOT_DIR}"
        OMP_NUM_THREADS=8 OMP_PLACES=cores OMP_PROC_BIND=close \
          ./build/main \
            --task task3 \
            --dataset "${dataset_name}" \
            --params "${run_tag}" \
            --output "results/optimal_runs/h5"
      ) > "${run_log}" 2>&1 || run_status=$?
    fi

    if [[ ${run_status} -ne 0 ]]; then
      echo "    [FAIL] run exited with status ${run_status}"
      echo "${dataset_name},${config_name},exit_${run_status},${run_log}" >> "${FAILED_RUNS_CSV}"

      if grep -q "Failed to open group: train" "${run_log}"; then
        echo "    [SKIP] dataset incompatible with current loader schema"
        echo "${dataset_name},missing_required_groups_runtime" >> "${INCOMPAT_DATASETS_CSV}"
        rm -rf "${run_cfg_dir}"
        break
      fi

      rm -rf "${run_cfg_dir}"
      continue
    fi

    if [[ -f "${run_csv_local}" ]]; then
      mv "${run_csv_local}" "${final_csv}"
      echo "    [OK] CSV -> ${final_csv}"
    else
      echo "    [WARN] Expected CSV not found: ${run_csv_local}" >&2
      echo "${dataset_name},${config_name},missing_csv,${run_log}" >> "${FAILED_RUNS_CSV}"
    fi

    rm -rf "${run_cfg_dir}"
  done
done

echo "[INFO] Building comparison summary"
python3 "${ROOT_DIR}/scripts/summarize_optimal_runs.py" \
  --runs-dir "${OPT_RUNS_DIR}" \
  --out-dir "${SUMMARY_DIR}"

echo "[DONE] Optimal config benchmark completed"
echo "[DONE] Combined CSV: ${SUMMARY_DIR}/combined_runs.csv"
echo "[DONE] Dataset ranking: ${SUMMARY_DIR}/ranking_per_dataset.csv"
echo "[DONE] Overall ranking: ${SUMMARY_DIR}/ranking_overall.csv"
echo "[DONE] Failed runs log: ${FAILED_RUNS_CSV}"
echo "[DONE] Incompatible datasets log: ${INCOMPAT_DATASETS_CSV}"
