from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
import onnxruntime_musa as musa_ep

DEFAULT_MODEL = Path("/home/workspace/onnx_musa_test/inference/metaGraph/onnx_out/meta_graph_1.onnx")
DEFAULT_FREE_DIMS = Path("/home/workspace/onnx_musa_test/inference/metaGraph/reports/meta_graph_1_free_dims.txt")


def _apply_free_dims(session_options: ort.SessionOptions, free_dims: Path, batch_size: int | None) -> None:
    if not free_dims.exists():
        raise FileNotFoundError(f"free dims file not found: {free_dims}")
    for item in free_dims.read_text().split():
        if ":" not in item:
            continue
        name, value_text = item.split(":", 1)
        value = int(value_text)
        if batch_size is not None and value == 1024:
            value = batch_size
        session_options.add_free_dimension_override_by_name(name, value)


def _make_feed(session: ort.InferenceSession, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    feeds: dict[str, np.ndarray] = {}
    for inp in session.get_inputs():
        shape = [int(dim) for dim in inp.shape]
        if inp.type == "tensor(float)":
            feeds[inp.name] = rng.uniform(0.01, 1.0, size=shape).astype(np.float32)
        elif inp.type == "tensor(int32)":
            feeds[inp.name] = rng.integers(0, 10, size=shape, dtype=np.int32)
        elif inp.type == "tensor(int64)":
            feeds[inp.name] = rng.integers(0, 10, size=shape, dtype=np.int64)
        else:
            raise TypeError(f"unsupported input type for {inp.name}: {inp.type}")
    return feeds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--free-dims", type=Path, default=DEFAULT_FREE_DIMS)
    parser.add_argument("--device", default="5")
    parser.add_argument("--batch-size", type=int, default=None, help="Override symbolic dims whose value is 1024.")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--create-session-only", action="store_true")
    args = parser.parse_args()

    os.environ["MUSA_VISIBLE_DEVICES"] = str(args.device)

    ort.register_execution_provider_library(musa_ep.get_ep_name(), musa_ep.get_library_path())
    devices = [d for d in ort.get_ep_devices() if d.ep_name == musa_ep.get_ep_name()]
    if not devices:
        raise RuntimeError(f"{musa_ep.get_ep_name()} is not visible")

    session_options = ort.SessionOptions()
    session_options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    _apply_free_dims(session_options, args.free_dims, args.batch_size)
    session_options.add_provider_for_devices(devices[:1], {})

    session = ort.InferenceSession(str(args.model), sess_options=session_options)
    print(f"[meta_graph_1] providers={session.get_providers()}")
    print(f"[meta_graph_1] inputs={len(session.get_inputs())} outputs={len(session.get_outputs())}")
    if args.create_session_only:
        print("[meta_graph_1] session=ok")
        return

    feeds = _make_feed(session, args.seed)
    feed_mb = sum(value.nbytes for value in feeds.values()) / 1024 / 1024
    print(f"[meta_graph_1] feed_mb={feed_mb:.3f}")
    start = time.perf_counter()
    outputs = session.run(None, feeds)
    latency_ms = (time.perf_counter() - start) * 1000
    print(f"[meta_graph_1] run=ok latency_ms={latency_ms:.3f}")
    for idx, output in enumerate(outputs):
        arr = np.asarray(output)
        finite = bool(np.isfinite(arr).all())
        print(f"[meta_graph_1] output[{idx}] shape={arr.shape} dtype={arr.dtype} finite={finite}")


if __name__ == "__main__":
    main()
