# Supported Ops

MUSA-backed kernel:

- `MatMul` float32 2D, opset 13-17, backed by muBLAS `mublasSgemm`.

Host-compatible kernels inside the MUSA Plugin EP:

- `Add`, `Sub`, `Mul`, `Div`, `Pow`, `Sum`
- `Gemm`, `FusedGemm`, `FusedMatMul`
- `Relu`, `LeakyRelu`, `Sqrt`, `Reciprocal`, `Neg`, `Log`, `Tanh`, `Sigmoid`, `Softmax`
- `Shape`, `Cast`, `Reshape`, `Squeeze`, `Unsqueeze`, `Concat`, `Transpose`, `Gather`, `Slice`, `Split`
- `ReduceProd`, `ReduceSum`, `ReduceMean`
