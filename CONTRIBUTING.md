# Contributing to onnxruntime-musa

Thank you for contributing to onnxruntime-musa. This repository contains an
out-of-tree ONNX Runtime Plugin Execution Provider for Moore Threads MUSA GPUs.

## Development Setup

Initialize the pinned ONNX Runtime header submodule first:

```bash
./scripts/init_onnxruntime_submodule.sh
```

Install Python dependencies in a virtual environment:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install -r requirements.txt
```

Build the plugin and wheel:

```bash
./build.sh
```

## Tests

Run the standard MUSA test suites on a machine with a visible MUSA device:

```bash
bash test/run_all.sh
```

Focused tests are preferred while developing:

```bash
python -m pytest test/ops/test_matmul.py -q
python -m pytest test/fusion/ -q
```

The tests disable CPU fallback for the MUSA run where appropriate. Unsupported
operators, dtypes, ranks, or attributes should fail loudly instead of silently
falling back to CPU.

## Pull Requests

Before opening a pull request:

- Keep the change focused and avoid unrelated formatting churn.
- Add or update end-to-end tests for new or changed operators and fusions.
- Update generated docs when operator or fusion registration changes.
- Run the narrowest relevant test first, then the broader suite when the change
  affects shared runtime, allocator, stream, fusion, or data-transfer paths.
- Do not include build artifacts, wheel files, local virtual environments, or
  profiler dumps.

Commit messages should follow conventional commit style when possible:

```text
feat(matmul): add batched fp16 path
fix(fusion): reject multi-consumer split pattern
docs: update MUSA environment variables
test(topk): add axis coverage
```

## Operator and Fusion Changes

New MUSA EP operator support must perform real MUSA device computation for the
main tensor work. Do not add hidden host-side main-compute fallbacks to make an
operator appear supported.

For fusion changes:

- Prefer extending an existing fusion when the graph shape and runtime behavior
  are naturally the same.
- Add new fusion matchers at the lowest priority unless there is a documented
  reason to run earlier.
- Prove the pattern is fused and compare outputs against a CPU reference.
- Keep matcher, runtime compute, docs, and tests in sync.

See musa/docs/fusion_development.md and musa/docs/developer_guide.md for more
project-specific guidance.
