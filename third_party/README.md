# Third-party dependencies

The only vendored source is the **ONNX Runtime public headers**:

- [`onnxruntime/include/onnxruntime/`](onnxruntime/include/onnxruntime/) — copied
  verbatim from the upstream `microsoft/onnxruntime` repository at tag **v1.26.0**
  (commit recorded in [`onnxruntime/VERSION`](onnxruntime/VERSION)).
- [`onnxruntime/LICENSE`](onnxruntime/LICENSE) — upstream MIT license for the copied
  headers.

These headers are required at compile time because `pip install onnxruntime` ships only
the runtime `.so` / Python bindings, not the C/C++ headers. At runtime nothing in
`third_party/` is needed; the plugin dlopens whatever `libonnxruntime.so` the user has
installed.

The vendored version is also the **single source of truth** for the wheel's
`onnxruntime` dependency pin: `musa/ep/python/build_wheel.py` parses
`tag: vX.Y.Z` from `onnxruntime/VERSION` and substitutes it into
`pyproject.toml.in` as `onnxruntime~=X.Y.Z`. Bumping the headers therefore bumps the
wheel's runtime constraint automatically.

## Refreshing the headers

To move to a new ORT release `vX.Y.Z`:

```bash
TMP=$(mktemp -d)
git clone --depth 1 --branch vX.Y.Z --filter=blob:none --sparse \
    https://github.com/microsoft/onnxruntime.git "$TMP/ort"
git -C "$TMP/ort" sparse-checkout set include/onnxruntime LICENSE
rm -rf third_party/onnxruntime/include
cp -r "$TMP/ort/include" third_party/onnxruntime/include
cp    "$TMP/ort/LICENSE" third_party/onnxruntime/LICENSE
{
  git -C "$TMP/ort" rev-parse HEAD
  echo "tag: vX.Y.Z"
} > third_party/onnxruntime/VERSION
```

No other dependencies (GSL, gtest, protobuf, …) are vendored — the plugin uses only the
C API, the C++ wrapper, and `std::span` from C++20.
