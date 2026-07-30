# MUSA multi-stream tests

These tests require an installed `onnxruntime` package, the current
`onnxruntime-musa` plugin wheel, and at least one visible MUSA device.

The allocator ownership index has ten device-independent CTest cases. They
cover assignment, clearing, ownership transfer, producer release after
transfer, split remainders, and rejected invalid index mutations:

```bash
cmake --build build/Release --target musa_arena_ownership_test
ctest --test-dir build/Release --output-on-failure \
  -R '^musa_arena_ownership\.'
```

Run the stream-ordered arena shared-Session test with:

```bash
PYTHONPATH=test/ops \
ORT_MUSA_ALLOCATOR_CACHE_LIMIT_MB=128 \
python -m pytest -q \
  test/multi_stream/test_allocator_cache_shared_session.py
```

The test creates one `InferenceSession` shared by four Python workers. It is
an end-to-end numerical check for concurrent Run lifetimes. For allocator
changes, also run the complete directory:

```bash
PYTHONPATH=test/ops python -m pytest -q test/multi_stream
```

Before interpreting a result, verify that the runtime plugin path and SHA256
match the newly built wheel. A skipped test does not count as MUSA validation.
