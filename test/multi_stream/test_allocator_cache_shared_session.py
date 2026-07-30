# Copyright (c) Moore Threads Technology Co., Ltd. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Concurrent shared-session coverage for the stream-ordered allocator cache."""

from concurrent.futures import ThreadPoolExecutor
import threading
import time

import numpy as np
import onnxruntime as ort
import pytest
from onnx import helper, numpy_helper

from op_test_utils import TensorProto, build_graph_model, musa_devices


def _make_model(feeds: dict[str, np.ndarray]) -> bytes:
    nodes = []
    initializers = []
    current = "X"
    for index in range(8):
        bias_name = f"bias_{index}"
        scale_name = f"scale_{index}"
        added = f"added_{index}"
        activated = f"activated_{index}"
        output = "Y" if index == 7 else f"scaled_{index}"
        bias = np.full(feeds["X"].shape, index * 0.125, dtype=np.float32)
        scale = np.array(0.75 + index * 0.03125, dtype=np.float32)
        initializers.extend(
            [
                numpy_helper.from_array(bias, name=bias_name),
                numpy_helper.from_array(scale, name=scale_name),
            ]
        )
        nodes.extend(
            [
                helper.make_node("Add", [current, bias_name], [added]),
                helper.make_node("Relu", [added], [activated]),
                helper.make_node("Mul", [activated, scale_name], [output]),
            ]
        )
        current = output
    return build_graph_model(
        nodes,
        inputs=feeds,
        outputs=[("Y", TensorProto.FLOAT)],
        initializers=initializers,
        name="allocator_cache_shared_session",
    )


def _create_musa_session(model: bytes):
    devices = musa_devices()
    if not devices:
        pytest.skip("No MUSA device available")

    options = ort.SessionOptions()
    options.add_provider_for_devices(devices, {})
    return ort.InferenceSession(model, sess_options=options)


def test_allocator_cache_shared_session_is_stream_ordered(monkeypatch):
    monkeypatch.setenv("ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB", "128")
    feeds = {"X": np.linspace(-2.0, 2.0, 1024, dtype=np.float32)}
    model = _make_model(feeds)

    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, feeds)

    session = _create_musa_session(model)

    def run_worker(_worker: int) -> None:
        for _ in range(20):
            (actual,) = session.run(None, feeds)
            np.testing.assert_array_equal(actual, expected)

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(run_worker, range(4)))


def test_staggered_run_end_keeps_concurrent_streams_isolated(monkeypatch):
    """Exercise overlapping Runs whose host-side RunEnd callbacks interleave."""

    monkeypatch.setenv("ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB", "128")
    feeds = {"X": np.linspace(-4.0, 4.0, 4096, dtype=np.float32)}
    model = _make_model(feeds)

    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, feeds)
    session = _create_musa_session(model)

    start = threading.Barrier(3)
    producer_finished_first_run = threading.Event()

    def producer() -> None:
        start.wait()
        for iteration in range(200):
            (actual,) = session.run(None, feeds)
            np.testing.assert_array_equal(actual, expected)
            if iteration == 0:
                producer_finished_first_run.set()
            if iteration % 7 == 0:
                time.sleep(0.0005)

    def consumer() -> None:
        start.wait()
        assert producer_finished_first_run.wait(timeout=30)
        for iteration in range(200):
            (actual,) = session.run(None, feeds)
            np.testing.assert_array_equal(actual, expected)
            if iteration % 5 == 0:
                time.sleep(0.0005)

    with ThreadPoolExecutor(max_workers=2) as executor:
        producer_future = executor.submit(producer)
        consumer_future = executor.submit(consumer)
        start.wait()
        producer_future.result()
        consumer_future.result()


def test_allocator_cache_survives_repeated_worker_stream_release(monkeypatch):
    """Repeated short-lived workers exercise stream release between cache uses."""

    monkeypatch.setenv("ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB", "128")
    feeds = {"X": np.linspace(-3.0, 3.0, 16384, dtype=np.float32)}
    model = _make_model(feeds)
    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, feeds)
    session = _create_musa_session(model)

    def run_once() -> None:
        (actual,) = session.run(None, feeds)
        np.testing.assert_array_equal(actual, expected)

    # Recreate workers rather than keeping a fixed pool: this repeatedly tears
    # down ORT's per-worker stream state while the shared session and arena
    # remain alive.
    for _ in range(128):
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: run_once(), range(4)))


@pytest.mark.parametrize("invalid_value", ["-1", "128x", "18446744073709551616"])
def test_allocator_cache_limit_rejects_invalid_values(monkeypatch, invalid_value):
    monkeypatch.setenv("ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB", invalid_value)
    feeds = {"X": np.ones(16, dtype=np.float32)}

    with pytest.raises(Exception, match="ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB"):
        _create_musa_session(_make_model(feeds))


def test_allocator_cache_zero_uses_direct_allocator(monkeypatch):
    monkeypatch.setenv("ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB", "0")
    feeds = {"X": np.linspace(-1.0, 1.0, 64, dtype=np.float32)}
    model = _make_model(feeds)
    cpu_session = ort.InferenceSession(model, providers=["CPUExecutionProvider"])
    (expected,) = cpu_session.run(None, feeds)

    (actual,) = _create_musa_session(model).run(None, feeds)
    np.testing.assert_array_equal(actual, expected)
