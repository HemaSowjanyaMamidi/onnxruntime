#include "contrib_ops/cpu/swish.h"
namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Swish,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float,double>()),
    Swish);
    
template <typename T>
static Status SwisImpl(const Tensor* X, Tensor* Y) {
  const auto& X_shape = X->Shape();
  int64_t X_num_dims = static_cast<int64_t>(X_shape.NumDimensions());
  const auto* X_data = reinterpret_cast<const T*>(X->DataRaw());
  auto* Y_data = reinterpret_cast<T*>(Y->MutableDataRaw());
  for (int64_t i = 0, sz = dims.Size(); i < sz; i++,Y_Data++,X_Data++) {
      *Y_Data = *X_Data / (exp(-*X_Data) + 1);
    }

    return Status::OK();
}
    
    
Status Swish::Compute(OpKernelContext* ctx) const {
  Status status;
  const auto* X = ctx->Input<Tensor>(0);
  const auto& X_shape = X->Shape();
  auto* Y = ctx->Output(0, X_shape);
  MLDataType data_type = X->DataType();
  const auto element_size = data_type->Size();
  switch (element_size) {
    case sizeof(float):
      status = SwishImpl<float>(X, Y);
      break;
    case sizeof(double):
      status = SwishImpl<double>(X, Y);
      break;
    default:
      ORT_THROW("Unsupported input data type of ", data_type);
  }
  return status;
}        
}
}
