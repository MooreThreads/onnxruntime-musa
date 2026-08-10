#!/usr/bin/env python3
# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0.

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import threading

import numpy as np
import onnxruntime as ort
import onnxruntime_musa as musa_ep

_EP_NAME = musa_ep.get_ep_name()
_REGISTER_LOCK = threading.Lock()
_REGISTERED = False


def _register_plugin() -> None:
    global _REGISTERED
    with _REGISTER_LOCK:
        if _REGISTERED:
            return
        ort.register_execution_provider_library(
            musa_ep.get_ep_name(), musa_ep.get_library_path()
        )
        _REGISTERED = True


def _select_musa_device():
    _register_plugin()
    devices = [d for d in ort.get_ep_devices() if d.ep_name == _EP_NAME]
    if not devices:
        raise RuntimeError(f"registered {_EP_NAME} but it advertised no visible devices")
    return devices[0], devices


def _build_sessions(model_path: Path):
    device, devices = _select_musa_device()

    cpu_session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])

    session_options = ort.SessionOptions()
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    session_options.add_provider_for_devices([device], {})
    musa_session = ort.InferenceSession(str(model_path), sess_options=session_options)

    return cpu_session, musa_session, devices


def _run_single_stream(session, feeds, expected):
    (actual,) = session.run(None, feeds)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)
    return actual


def _run_multi_stream(session, feeds, expected, workers: int, repeats: int):
    def worker(worker_id: int) -> None:
        for _ in range(repeats):
            (actual,) = session.run(None, feeds)
            np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(worker, range(workers)))


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the single gemm ONNX model on CPU and the MUSA EP."
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "cpp" / "single_gemm_opset19.onnx",
        help="Path to the ONNX model.",
    )
    parser.add_argument(
        "--mode",
        choices=("single", "multi", "both"),
        default="both",
        help="Which inference path to run.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=4,
        help="Number of concurrent workers for the multi-stream path.",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=16,
        help="Number of runs per worker in the multi-stream path.",
    )
    args = parser.parse_args()

    if args.workers <= 0 or args.repeats <= 0:
        raise SystemExit("--workers and --repeats must be positive")

    feeds = {"X": np.array([[1.0, -2.0, 3.0]], dtype=np.float32)}

    cpu_session, musa_session, devices = _build_sessions(args.model)
    expected = cpu_session.run(None, feeds)[0]

    print(f"registered_ep={_EP_NAME} visible_devices={len(devices)}")
    print(f"model={args.model}")
    print(f"input_X={feeds['X'].tolist()}")

    if args.mode in {"single", "both"}:
        actual = _run_single_stream(musa_session, feeds, expected)
        print(f"single_stream_output={actual.tolist()}")

    if args.mode in {"multi", "both"}:
        _run_multi_stream(musa_session, feeds, expected, args.workers, args.repeats)
        print(
            f"multi_stream_output=verified workers={args.workers} repeats={args.repeats}"
        )

    print(f"cpu_output={expected.tolist()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
