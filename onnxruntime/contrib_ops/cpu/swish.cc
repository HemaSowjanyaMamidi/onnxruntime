#include "contrib_ops/cpu/swish.h"
namespace onnxruntime {
namespace contrib {

ONNX_OPERATOR_KERNEL_EX(
    Swish,
    kOnnxDomain,
    1,
    kCpuExecutionProvider,
    KernelDefBuilder().TypeConstraint("T", DataTypeImpl::GetTensorType<float>()),
    Swish<float>
);

}
}
