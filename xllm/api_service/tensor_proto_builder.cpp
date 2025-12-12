/* Copyright 2025 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensor_proto_builder.h"

namespace xllm {

TensorProtoBuilder::TensorProtoBuilder(bool use_binary_encoding)
    : use_binary_encoding_(use_binary_encoding) {};
TensorProtoBuilder::~TensorProtoBuilder() {};

std::string TensorProtoBuilder::torch_dtype_to_string(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:
      return "fp32";
    case at::kDouble:
      return "fp64";
    case at::kHalf:
      return "fp16";
    case at::kBFloat16:
      return "bf16";
    case at::kByte:
      return "uint8";
    case at::kChar:
      return "int8";
    case at::kShort:
      return "int16";
    case at::kInt:
      return "int32";
    case at::kLong:
      return "int64";
    case at::kBool:
      return "bool";
    default:
      return "unknown";
  }
}

bool TensorProtoBuilder::must_use_binary_format(at::ScalarType dtype) {
  switch (dtype) {
    case at::kFloat:     // fp32
    case at::kDouble:    // fp64
    case at::kByte:      // uint8
    case at::kChar:      // int8
    case at::kShort:     // int16
    case at::kInt:       // int32
    case at::kLong:      // int64
    case at::kBool:      // bool
      return false;      // proto support
    case at::kHalf:      // fp16
    case at::kBFloat16:  // bf16
      return true;       // proto is not supported，must use binary format
    default:
      // unknown fall back 到 binary format
      return true;
  }
}

bool TensorProtoBuilder::build_repeated_tensor(
    const std::vector<torch::Tensor>& in_tensors,
    google::protobuf::RepeatedPtrField<xllm::proto::Tensor>& out_tensors,
    std::string& binary_payload) {
  for (const auto& in_tensor : in_tensors) {
    CHECK(in_tensor.is_contiguous())
        << "Internal Error: only support contiguous mm_embedding";

    xllm::proto::Tensor* out_tensor = out_tensors.Add();

    if (!build_tensor(in_tensor, *out_tensor, binary_payload)) {
      return false;
    }
  }
  return true;
}

bool TensorProtoBuilder::build_tensor(const torch::Tensor& in_tensor,
                                      xllm::proto::Tensor& out_tensor,
                                      std::string& binary_payload) {
  out_tensor.set_datatype(torch_dtype_to_string(in_tensor.scalar_type()));

  for (auto dim : in_tensor.sizes()) {
    out_tensor.add_shape(static_cast<int32_t>(dim));
  }

  float* data_ptr = in_tensor.data_ptr<float>();
  int64_t numel = in_tensor.numel();

  if (use_binary_encoding_ || must_use_binary_format(in_tensor.scalar_type())) {
    // build tensor in binary format
    size_t byte_len = sizeof(float) * numel;
    size_t offset = binary_payload.size();
    auto* params = out_tensor.mutable_parameters();
    (*params)["offset"].set_int64_param(offset);
    (*params)["len"].set_int64_param(byte_len);
    (*params)["is_binary"].set_bool_param(true);
    binary_payload.append(reinterpret_cast<const char*>(data_ptr),
                          sizeof(float) * numel);
  } else {
    // build tensor in json format
    auto* tensor_contents =
        new xllm::proto::TensorContents();  // protobuf take ownership of the
    tensor_contents->mutable_fp32_contents()->Add(data_ptr, data_ptr + numel);

    out_tensor.set_allocated_contents(
        tensor_contents);  // protobuf take ownership of the tensor_contents
                           // memory
  }
  return true;
}

};  // namespace xllm