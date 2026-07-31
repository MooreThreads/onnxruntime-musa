#include "fusion/reduced_mha_flash_fusion.h"

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include "fusion/fusion_matcher_utils.h"
#include "graph/graph_utils.h"
#include "kernels/llm/attention_impl.h"
#include "kernels/llm/reduced_mha_flash_impl.h"
#include "kernels/math/matmul.h"
#include "kernels/shared_inc/blas_utils.h"
#include "kernels/shared_inc/op_kernel_common.h"
#include "plugin_ep_utils.h"

namespace {
std::string Name(Ort::ConstValueInfo v) { return v.GetName(); }
class Scratch {
 public:
  ~Scratch() {
    if (p_) FreeDeviceMemoryOnStream(p_, s_, bytes_);
  }
  void Allocate(size_t n, musaStream_t s) {
    bytes_ = n;
    s_ = s;
    if (n) {
      p_ = AllocateDeviceMemoryOnStream(n, s);
      if (!p_)
        throw std::runtime_error(MusaErrorString(musaErrorMemoryAllocation));
    }
  }
  void* data() const { return p_; }

 private:
  void* p_ = nullptr;
  size_t bytes_ = 0;
  musaStream_t s_ = nullptr;
};
std::unordered_map<std::string, size_t> Indices(Ort::ConstNode n) {
  std::unordered_map<std::string, size_t> r;
  auto in = n.GetInputs();
  for (size_t i = 0; i < in.size(); ++i) r.emplace(Name(in[i]), i);
  return r;
}
size_t Index(const std::unordered_map<std::string, size_t>& m,
             Ort::ConstValueInfo v) {
  auto i = m.find(Name(v));
  if (i == m.end())
    throw std::runtime_error("ReducedMhaFlash missing fused input");
  return i->second;
}
float FloatAttr(Ort::ConstNode n, const char* name, float d) {
  Ort::ConstOpAttr a;
  float x = d;
  return n.GetAttributeByName(name, a).IsOK() && a.GetValue(x).IsOK() ? x : d;
}

// Prefer muDNN's GEMM+bias epilogue.  It removes the standalone device bias
// launch; retain the existing device-only fallback when an algorithm is not
// available for a shape on the installed muDNN.
OrtStatus* RunGemmWithBias(float* y, const void* x, const void* w,
                           const float* bias,
                           const std::vector<int64_t>& x_shape,
                           const std::vector<int64_t>& w_shape,
                           const std::vector<int64_t>& y_shape, bool trans_b,
                           musaStream_t stream) {
  const std::vector<int64_t> bias_shape{y_shape.back()};
  if (TryMudnnGemm(y, x, w, bias, x_shape, w_shape, bias_shape, y_shape, false,
                   trans_b, 1.0f, 1.0f, true,
                   ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, stream)) {
    return nullptr;
  }
  RETURN_IF_ERROR(ComputeMusaMatMulDevice(
      x, w, y, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, x_shape, w_shape, y_shape,
      false, trans_b, false, false, 1.0f, stream));
  return LaunchStatus(LaunchMusaAttentionAddBiasKernel(
      y, bias, y_shape[0] * y_shape[1], y_shape[1], stream));
}

class ReducedMhaFlashFusionCompute final : public FusionNodeCompute {
 public:
  ReducedMhaFlashFusionCompute(size_t x, size_t qw, size_t qb, size_t mask,
                               size_t ow, size_t ob, int64_t heads, int64_t a,
                               float scale)
      : x_(x),
        qw_(qw),
        qb_(qb),
        mask_(mask),
        ow_(ow),
        ob_(ob),
        heads_(heads),
        a_(a),
        scale_(scale) {}
  OrtStatus* Compute(OrtKernelContext* c) const override {
    try {
      Ort::KernelContext ctx(c);
      musaStream_t stream = GetComputeStream(ctx);
      Ort::ConstValue x = ctx.GetInput(x_), qw = ctx.GetInput(qw_),
                      qb = ctx.GetInput(qb_), ow = ctx.GetInput(ow_),
                      ob = ctx.GetInput(ob_);
      auto xi = x.GetTensorTypeAndShapeInfo();
      auto xs = xi.GetShape();
      auto qws = qw.GetTensorTypeAndShapeInfo().GetShape();
      auto qbs = qb.GetTensorTypeAndShapeInfo().GetShape();
      auto ows = ow.GetTensorTypeAndShapeInfo().GetShape();
      auto obs = ob.GetTensorTypeAndShapeInfo().GetShape();
      if (xi.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
          xs.size() != 2 || qws.size() != 2 || qbs.size() != 1 ||
          ows.size() != 2 || obs.size() != 1 || xs[0] < 0 || xs[1] < 0)
        throw std::runtime_error(
            "ReducedMhaFlash requires static FP32 rank-2 input and rank-2 "
            "weights");
      const int64_t s = xs[0], in = xs[1], out = ows[0];
      if (a_ <= 0 || heads_ <= 0 || a_ % heads_ != 0 || qws[0] != in ||
          qws[1] != 3 * a_ || qbs[0] != 3 * a_ || ows[1] != a_ || obs[0] != out)
        throw std::runtime_error(
            "ReducedMhaFlash projection dimensions mismatch");
      Ort::ConstValue mask{nullptr};
      bool has = mask_ != SIZE_MAX;
      std::vector<int64_t> ms;
      if (has) {
        mask = ctx.GetInput(mask_);
        auto mi = mask.GetTensorTypeAndShapeInfo();
        ms = mi.GetShape();
        if (mi.GetElementType() != ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32 ||
            ms.size() != 4 || (ms[0] != 1) || (ms[1] != 1 && ms[1] != heads_) ||
            ms[2] != s || ms[3] != s)
          throw std::runtime_error(
              "ReducedMhaFlash requires int32 mask [1,1|H,S,S]");
      }
      Ort::UnownedValue y = ctx.GetOutput(0, {s, out});
      if (!IsGpuMemory(y.GetTensorMemoryInfo()))
        throw std::runtime_error("ReducedMhaFlash requires MUSA output");
      DeviceInputBuffer xb, qwb, qbb, owb, obb, mb;
      RETURN_IF_ERROR(xb.Bind(x, stream));
      RETURN_IF_ERROR(qwb.Bind(qw, stream));
      RETURN_IF_ERROR(qbb.Bind(qb, stream));
      RETURN_IF_ERROR(owb.Bind(ow, stream));
      RETURN_IF_ERROR(obb.Bind(ob, stream));
      if (has) RETURN_IF_ERROR(mb.Bind(mask, stream));
      Scratch packed, attention;
      packed.Allocate(static_cast<size_t>(s * 3 * a_) * sizeof(float), stream);
      attention.Allocate(static_cast<size_t>(s * a_) * sizeof(float), stream);
      RETURN_IF_ERROR(
          RunGemmWithBias(static_cast<float*>(packed.data()), xb.data(),
                          qwb.data(), static_cast<const float*>(qbb.data()),
                          {s, in}, qws, {s, 3 * a_}, false, stream));
      MusaReducedMhaFlashParams p{1,
                                  s,
                                  a_,
                                  heads_,
                                  a_ / heads_,
                                  has ? ms[0] : 1,
                                  has ? ms[1] : 1,
                                  scale_,
                                  0.0f,
                                  has,
                                  false,
                                  false};
      RETURN_IF_ERROR(LaunchStatus(LaunchMusaReducedMhaFlashKernel(
          static_cast<const float*>(packed.data()),
          has ? static_cast<const int32_t*>(mb.data()) : nullptr,
          static_cast<float*>(attention.data()), p, stream)));
      return RunGemmWithBias(y.GetTensorMutableData<float>(), attention.data(),
                             owb.data(), static_cast<const float*>(obb.data()),
                             {s, a_}, ows, {s, out}, true, stream);
    } catch (const std::exception& e) {
      return Ort::Status(e.what(), ORT_EP_FAIL).release();
    }
  }

 private:
  size_t x_, qw_, qb_, mask_, ow_, ob_;
  int64_t heads_, a_;
  float scale_;
};
}  // namespace

bool IsReducedMhaFlashFusionGraph(Ort::ConstGraph g) {
  int u = 0, a = 0, r = 0, m = 0;
  for (auto n : g.GetNodes()) {
    u += musa_ep::IsOnnxOp(n, "Unsqueeze");
    r += musa_ep::IsOnnxOp(n, "Reshape");
    m += musa_ep::IsOnnxOp(n, "Gemm");
    a += n.GetOperatorType() == "Attention" && n.GetDomain() == "com.microsoft";
  }
  return u == 1 && a == 1 && r == 1 && m == 1;
}
std::unique_ptr<FusionNodeCompute> CreateReducedMhaFlashFusion(
    Ort::ConstGraph g, Ort::ConstNode fused) {
  Ort::ConstNode u{nullptr}, a{nullptr}, m{nullptr};
  for (auto n : g.GetNodes()) {
    if (musa_ep::IsOnnxOp(n, "Unsqueeze"))
      u = n;
    else if (n.GetOperatorType() == "Attention")
      a = n;
    else if (musa_ep::IsOnnxOp(n, "Gemm"))
      m = n;
  }
  if (!u || !a || !m) throw std::runtime_error("invalid ReducedMhaFlash graph");
  auto ai = a.GetInputs();
  auto ui = u.GetInputs();
  auto mi = m.GetInputs();
  auto idx = Indices(fused);
  auto q = musa_ep::GetIntsAttribute(a, "qkv_hidden_sizes");
  if (!q || q->size() != 3)
    throw std::runtime_error("missing qkv_hidden_sizes");
  return std::make_unique<ReducedMhaFlashFusionCompute>(
      Index(idx, ui[0]), Index(idx, ai[1]), Index(idx, ai[2]),
      ai.size() == 4 ? Index(idx, ai[3]) : SIZE_MAX, Index(idx, mi[1]),
      Index(idx, mi[2]), musa_ep::GetIntAttribute(a, "num_heads").value_or(0),
      (*q)[0],
      FloatAttr(
          a, "scale",
          1.0f / std::sqrt(static_cast<float>(
                     (*q)[0] /
                     musa_ep::GetIntAttribute(a, "num_heads").value_or(1)))));
}
