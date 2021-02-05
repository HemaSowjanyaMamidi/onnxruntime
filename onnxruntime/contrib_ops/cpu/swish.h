namespace onnxruntime {
namespace contrib {
 class Swish final : public OpKernel {
 public:
  explicit Swish(const OpKernelInfo& info) : OpKernel(info) {}
  
  Status Compute(OpKernelContext* ctx) const override;
};
}  // namespace contrib
}  // namespace onnxruntime
