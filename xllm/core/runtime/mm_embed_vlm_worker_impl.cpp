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

#include "mm_embed_vlm_worker_impl.h"

#include <c10/core/DeviceGuard.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <memory>
#include <optional>
#include <utility>

#include "common/metrics.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_input_params.h"
#include "framework/state_dict/state_dict.h"
#include "models/model_registry.h"
#include "options.h"
#include "util/timer.h"

namespace xllm {

MMEmbedVLMWorkerImpl::MMEmbedVLMWorkerImpl(const ParallelArgs& parallel_args,
                                           const torch::Device& device,
                                           const runtime::Options& options)
    : WorkerImpl(parallel_args, device, options) {}

bool MMEmbedVLMWorkerImpl::init_model(ModelContext& context) {
  CHECK(model_ == nullptr) << "Model is already initialized.";

  // TODO replace with only vision model
  model_ = create_vlm_model(context);
  CHECK(model_ != nullptr) << "Failed to create model.";
  model_executor_ = std::make_unique<Executor>(
      model_.get(), context.get_model_args(), device_, options_);

  return true;
}

std::optional<ForwardOutput> MMEmbedVLMWorkerImpl::step(
    const ForwardInput& input) {
  torch::DeviceGuard device_guard(device_);
  auto ret = device_.synchronize_default_stream();

  Timer timer;

  // TODO remove language params in only vision model forward.
  // TODO to adapt multi stream parallel later, just use [0] temporarily
  // all tensors should be on the same device as model
  auto flatten_tokens = input.token_ids.to(device_);
  auto flatten_positions = input.positions.to(device_);
  auto params = input.input_params.to(device_);
  auto sampling_params = input.sampling_params.to(device_, dtype_);

  // call model executor forward to get hidden states
  auto hidden_states = model_executor_->forward(
      flatten_tokens, flatten_positions, kv_caches_, params);
  LOG(INFO) << "$$$$$$$$$$ hidden_states size: " << hidden_states.sizes();

  ret = device_.synchronize_default_stream();
  COUNTER_ADD(execution_latency_seconds_model, timer.elapsed_seconds());

  if (!driver_) {
    return std::nullopt;
  }

  // driver prepare model output
  ForwardOutput output;
  SampleOutput sample_output;

  if (input.sampling_params.is_embeddings) {
    // TODO get encoder output
    std::vector<int> n_image_tokens = {66, 1156};
    std::vector<int> cu_n_image_tokens = {0};
    for (int i = 0; i < n_image_tokens.size(); ++i) {
      cu_n_image_tokens.push_back(cu_n_image_tokens.back() + n_image_tokens[i]);
    }
    std::vector<torch::Tensor> mm_embeddings;
    for (int i = 0; i < n_image_tokens.size(); i++) {
      mm_embeddings.push_back(hidden_states.slice(
          0, cu_n_image_tokens[i], cu_n_image_tokens[i + 1]));
    }
    for (auto& emb : mm_embeddings) {
      LOG(INFO) << "$$$$$$$$$$ mm_embedding size: " << emb.sizes();
      LOG(INFO) << "$$$$$$$$$$ mm_embedding first ten value: "
                << emb.view(-1).slice(0, 0, 10);
    }

    auto embeddings =
        torch::arange(16, torch::dtype(torch::kFloat)).reshape({1, 16});
    sample_output.embeddings = embeddings;
    sample_output.mm_embeddings = mm_embeddings;
    output.sample_output = sample_output;
    // output.embedding = embeddings;
  }

  return output;
}

}  // namespace xllm
