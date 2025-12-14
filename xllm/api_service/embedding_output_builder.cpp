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

#include "embedding_output_builder.h"

namespace xllm {

TensorProtoBuilder::TensorProtoBuilder(bool use_binary_encoding)
    : use_binary_encoding_(use_binary_encoding) {};
TensorProtoBuilder::~TensorProtoBuilder() {};

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
  if (use_binary_encoding_) {
    // build tensor in binary format
    out_tensor.set_datatype(
        util::torch_datatype_to_proto(in_tensor.scalar_type()));

    for (auto dim : in_tensor.sizes()) {
      out_tensor.add_shape(static_cast<int32_t>(dim));
    }

    auto numel = in_tensor.numel();
    auto byte_len = numel * in_tensor.element_size();
    size_t offset = binary_payload.size();
    auto* params = out_tensor.mutable_parameters();
    (*params)["offset"].set_int64_param(offset);
    (*params)["len"].set_int64_param(byte_len);
    (*params)["is_binary"].set_bool_param(true);
    binary_payload.append(reinterpret_cast<const char*>(in_tensor.data_ptr()),
                          byte_len);
  } else {
    // build tensor in json format
    util::torch_to_proto(in_tensor, &out_tensor);
  }
  return true;
}

EmbeddingOutputBuilder::EmbeddingOutputBuilder(
    bool embedding_use_binary_encoding,
    bool metadata_use_binary_encoding)
    : embedding_use_binary_encoding_(embedding_use_binary_encoding),
      metadata_use_binary_encoding_(metadata_use_binary_encoding) {};

EmbeddingOutputBuilder::~EmbeddingOutputBuilder() {};

bool EmbeddingOutputBuilder::build_repeated_embedding_output(
    const std::vector<EmbeddingOutput>& in_embedding_outputs,
    google::protobuf::RepeatedPtrField<xllm::proto::EmbeddingData>&
        out_embedding_outputs,
    std::string& binary_payload) {
  for (const auto& in_embedding_output : in_embedding_outputs) {
    xllm::proto::EmbeddingData* out_embedding_output =
        out_embedding_outputs.Add();
    if (!build_embedding_output(
            in_embedding_output, *out_embedding_output, binary_payload)) {
      return false;
    }
  }
  return true;
}

bool EmbeddingOutputBuilder::build_embedding_output(
    const EmbeddingOutput& in_embedding_output,
    xllm::proto::EmbeddingData& out_embedding_output,
    std::string& binary_payload) {
  TensorProtoBuilder embedding_output_builder(embedding_use_binary_encoding_);
  embedding_output_builder.build_tensor(
      in_embedding_output.embedding,
      *out_embedding_output.mutable_embedding(),
      binary_payload);

  auto* meta_map = out_embedding_output.mutable_metadata();
  for (const auto& [key, value] : in_embedding_output.metadata) {
    TensorProtoBuilder meta_output_builder(metadata_use_binary_encoding_);
    xllm::proto::Tensor metadata_tensor;
    meta_output_builder.build_tensor(
        in_embedding_output.metadata.at(key), metadata_tensor, binary_payload);
    (*meta_map)[key] = std::move(metadata_tensor);
  }
  return true;
};

};  // namespace xllm