#!/usr/bin/env bash
set -euo pipefail

usage() {
	cat <<'EOF'
Usage: run_model_zoo_benchmark.sh LABEL PLUGIN MODE ZOO_DIR RESULTS_DIR

Run one model-zoo performance pass and preserve its report and error logs.

Arguments:
  LABEL        Version label used in result file names, for example base or head.
  PLUGIN       Absolute path to libonnxruntime_providers_musa_plugin.so.
  MODE         single or multi.
  ZOO_DIR      Prepared onnx-model-zoo working directory.
  RESULTS_DIR  Directory that receives the report and error logs.

Modes:
  single  run_all_onnx.sh --ep musa
  multi   run_all_onnx.sh --ep musa -s 4 -q 128
EOF
}

if [[ $# -ne 5 ]]; then
	usage >&2
	exit 2
fi

LABEL="$1"
PLUGIN="$2"
MODE="$3"
ZOO_DIR="$4"
RESULTS_DIR="$5"

if [[ ! "$LABEL" =~ ^[A-Za-z0-9_.-]+$ ]]; then
	echo "Invalid label: $LABEL" >&2
	exit 2
fi
if [[ ! -f "$PLUGIN" ]]; then
	echo "MUSA plugin does not exist: $PLUGIN" >&2
	exit 1
fi
if [[ ! -d "$ZOO_DIR" ]]; then
	echo "Model zoo directory does not exist: $ZOO_DIR" >&2
	exit 1
fi

case "$MODE" in
	single)
		REPORT_NAME="musa_ep_performance.md"
		RUN_ARGS=(--ep musa)
		;;
	multi)
		REPORT_NAME="musa_ep_performance_with_multi_stream.md"
		RUN_ARGS=(--ep musa -s 4 -q 128)
		;;
	*)
		echo "Mode must be single or multi: $MODE" >&2
		exit 2
		;;
esac

mkdir -p "$RESULTS_DIR"
export MUSA_EP_LIBRARY="$PLUGIN"
export MUSA_HOME="${MUSA_HOME:-/usr/local/musa}"
export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0}"
export LD_LIBRARY_PATH="$MUSA_HOME/lib:$MUSA_HOME/lib64:${LD_LIBRARY_PATH:-}"

echo "benchmark_version=${LABEL}"
echo "benchmark_mode=${MODE}"
echo "benchmark_plugin=${MUSA_EP_LIBRARY}"
echo "MUSA_VISIBLE_DEVICES=${MUSA_VISIBLE_DEVICES}"
sha256sum "$MUSA_EP_LIBRARY"

cd "$ZOO_DIR"
set +e
bash run_all_onnx.sh "${RUN_ARGS[@]}"
run_status=$?
set -e

report_path="$ZOO_DIR/$REPORT_NAME"
result_path="$RESULTS_DIR/$LABEL-$MODE.md"
if [[ -f "$report_path" ]]; then
	cp "$report_path" "$result_path"
	echo "Saved benchmark report to $result_path"
else
	echo "Benchmark report was not generated: $report_path" >&2
	exit 1
fi

if [[ -d "$ZOO_DIR/error_logs" ]]; then
	error_results="$RESULTS_DIR/$LABEL-$MODE-error-logs"
	rm -rf "$error_results"
	cp -a "$ZOO_DIR/error_logs" "$error_results"
	echo "Saved benchmark error logs to $error_results"
fi

if [[ "$run_status" -ne 0 ]]; then
	echo "Benchmark failed with exit code $run_status" >&2
	exit "$run_status"
fi

if ! grep -Fq '| Fail | 0 |' "$result_path"; then
	echo "Benchmark reported one or more failed models: $result_path" >&2
	grep -F '| Fail |' "$result_path" >&2 || true
	exit 1
fi

if [[ "$MODE" == "single" ]]; then
	grep -Fq '| Mode | classic |' "$result_path"
else
	grep -Fq '| Querys | 128 |' "$result_path"
	grep -Fq '| Streams | 4 |' "$result_path"
fi

grep -F '| Model count |' "$result_path"
echo "Benchmark completed: version=${LABEL} mode=${MODE}"
