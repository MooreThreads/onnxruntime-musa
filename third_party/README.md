# Third-party dependencies

The only third-party source checkout is the **ONNX Runtime Git submodule**:

- [`onnxruntime/`](onnxruntime/) points to upstream `microsoft/onnxruntime` tag
  **v1.26.0**, commit `8c546c37b43caaca1fa25db430dab94b901cf277`.
- Cone-mode sparse checkout materializes only
  [`onnxruntime/include/onnxruntime/`](onnxruntime/include/onnxruntime/) plus upstream
  root files such as [`VERSION_NUMBER`](onnxruntime/VERSION_NUMBER) and
  [`LICENSE`](onnxruntime/LICENSE).

These headers are required at compile time because `pip install onnxruntime` ships only
the runtime `.so` / Python bindings, not the C/C++ headers. At runtime nothing in
`third_party/` is needed; the plugin dlopens whatever `libonnxruntime.so` the user has
installed.

The submodule's upstream `VERSION_NUMBER` is also the **single source of truth** for
the wheel's `onnxruntime` dependency pin: `musa/ep/python/build_wheel.py` reads
`X.Y.Z` from `onnxruntime/VERSION_NUMBER` and substitutes it into `pyproject.toml.in`
as `onnxruntime~=X.Y.Z`.

## Initializing the submodule

After cloning the parent repository, run:

```bash
./scripts/init_onnxruntime_submodule.sh
```

The helper initializes the commit recorded by the parent repository, honors the
submodule's shallow-clone recommendation, uses partial-clone filtering when the local
Git version supports it, and configures the sparse checkout. The sparse-checkout
configuration is local worktree state, so it cannot be stored in `.gitmodules` alone.

## Updating ONNX Runtime

To move to a new ORT release `vX.Y.Z`:

```bash
git -C third_party/onnxruntime fetch --depth 1 origin tag vX.Y.Z
git -C third_party/onnxruntime checkout --detach vX.Y.Z
git -C third_party/onnxruntime sparse-checkout set include/onnxruntime
git add third_party/onnxruntime
```

Then rebuild and test against the matching `onnxruntime` package before committing the
updated gitlink. Do not configure a submodule tracking branch or use
`git submodule update --remote`: the parent repository intentionally pins an exact ORT
commit.

No other dependencies (GSL, gtest, protobuf, …) are checked out under `third_party/` —
the plugin uses only the ORT C API, the C++ wrapper, and `std::span` from C++20.
