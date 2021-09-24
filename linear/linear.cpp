#include <torch/extension.h>

#include <vector>
//
torch::Tensor linear_forward(
  torch::Tensor input,
  torch::Tensor weights,
  torch::Tensor bias){

  auto out = torch::addmm(bias, input, weights.transpose(0, 1));
  return out;
}


std::vector<torch::Tensor> linear_backward(
  torch::Tensor grad_output,
  torch::Tensor input,
  torch::Tensor weights){

  auto grad_input = grad_output.mm(weights);
  auto grad_weight = grad_output.t().mm(input);
  auto grad_bias = grad_output.sum(0);

  return {grad_input, grad_weight, grad_bias};
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &linear_forward, "Linear forward");
  m.def("backward", &linear_backward, "Linear backward");
}
