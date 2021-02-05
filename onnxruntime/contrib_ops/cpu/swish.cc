#include "contrib_ops/cpu/swish.h"
#include "core/common/common.h"
#include "core/framework/op_kernel.h"
#include "core/util/math.h"
#include "core/util/math_cpuonly.h"
#include "core/framework/tensor.h"

namespace onnxruntime {
namespace contrib {

#define DEFINE_KERNEL(data_type)                                                                                  \
  ONNX_OPERATOR_TYPED_KERNEL_EX(Swish, kOnnxDomain, 1, data_type, kCpuExecutionProvider,                            \
                                KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<data_type>()), \
                                Swish<data_type>);
DEFINE_KERNEL(float);
DEFINE_KERNEL(double);

    
template <typename T>    
Status Swish<T>::Compute(OpKernelContext* context) const {
  auto X = context->Input<Tensor>(0);
    auto& dims = X->Shape();
    auto Y = context->Output(0, dims);
    
    auto X_Data = (X->Data<T>());
    auto Y_Data = (Y->MutableData<T>());

    for (int64_t i = 0, sz = dims.Size(); i < sz; i++,Y_Data++,X_Data++) {
      *Y_Data = *X_Data / (exp(-*X_Data) + 1);
    }
    return Status::OK();
};        
}
}
