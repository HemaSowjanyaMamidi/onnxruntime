#include "core/common/common.h"
#include "core/framework/op_kernel.h"

namespace onnxruntime {
namespace contrib {
template <typename T>
 class Swish : public OpKernel {
 public:
  explicit Swish(const OpKernelInfo& info) : OpKernel(info) {}
  
  Status Compute(OpKernelContext* ctx) const override;
};
}  // namespace contrib
}  // namespace onnxruntime
