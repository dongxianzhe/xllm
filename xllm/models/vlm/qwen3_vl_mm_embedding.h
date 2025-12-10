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

#pragma once

#include "qwen3_vl.h"

namespace xllm {

class Qwen3_VLForMMEmbeddingImpl : public torch::nn::Module {
 public:
  Qwen3_VLForMMEmbeddingImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    visual_ = register_module("visual", Qwen3_VisionTransformer(context));
  }

  std::vector<int> get_images_size(torch::Tensor image_grid_thw) {
    if (!image_grid_thw.defined()) return {};

    int merge_size = model_args_.mm_image_merge_size();
    int merge_length = merge_size * merge_size;

    std::vector<int> images_size;
    int count = image_grid_thw.sizes()[0];
    images_size.reserve(count);
    for (int idx = 0; idx < count; ++idx) {
      int n_image_tokens =
          image_grid_thw[idx].prod().item<int>() / merge_length;
      images_size.emplace_back(n_image_tokens);
    }
    return images_size;
  }

  std::vector<torch::Tensor> encode(const ModelInputParams& input_params) {
    torch::NoGradGuard no_grad;
    const auto& mm_data = input_params.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();
    std::optional<Qwen3_VLImageInputs> image_inputs;

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = Qwen3_VLImageInputs{pixel_values, image_grid_thw};
    CHECK(image_inputs.has_value());

    auto [image_embeds, deep_stacks] =
        visual_(image_inputs->pixel_values.to(options_),
                image_inputs->image_grid_thw,
                input_params);
    input_params.deep_stacks = deep_stacks;

    std::vector<int> image_sizes = get_images_size(image_grid_thw);

    std::vector<torch::Tensor> mm_embeddings;
    mm_embeddings.reserve(image_sizes.size());

    int token_start_idx = 0;
    for (int image_size : image_sizes) {
      auto image_embed =
          image_embeds.slice(0, token_start_idx, token_start_idx + image_size);
      mm_embeddings.emplace_back(image_embed);
      token_start_idx += image_size;
    }
    CHECK(token_start_idx == image_embeds.size(0));

    return mm_embeddings;
  };

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      visual_->load_state_dict(
          state_dict->get_dict_with_prefix("model.visual."));
    }
#if defined(USE_NPU)
    // verify
    visual_->verify_loaded_weights("model.visual.");
    visual_->merge_loaded_weights();
#endif
  }

  torch::Device device() const { return options_.device(); }
  const torch::TensorOptions& options() const { return options_; }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;

  Qwen3_VisionTransformer visual_{nullptr};
};
TORCH_MODULE(Qwen3_VLForMMEmbedding);

template <>
class MMEmbeddingVLMImpl<xllm::Qwen3_VLForMMEmbedding> : public MMEmbeddingVLM {
 public:
  MMEmbeddingVLMImpl(xllm::Qwen3_VLForMMEmbedding model,
                     const torch::TensorOptions& options)
      : model_(std::move(model)), options_(options) {}

  std::vector<torch::Tensor> encode(
      const ModelInputParams& input_params) override {
    return model_->encode(input_params);
  };

  void load_model(std::unique_ptr<ModelLoader> loader) override {
    model_->load_model(std::move(loader));
  }

  torch::Device device() const override { return model_->device(); }

  const torch::TensorOptions& options() const override {
    return model_->options();
  }

 private:
  xllm::Qwen3_VLForMMEmbedding model_;
  torch::TensorOptions options_;
};

REGISTER_MM_EMBEDDING_VLM_MODEL_WITH_VARNAME(qwen3_vl_mm_embedding,
                                             qwen3_vl,
                                             Qwen3_VLForMMEmbedding);
}  // namespace xllm
