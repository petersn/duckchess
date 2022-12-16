#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "NvInfer.h"

#define cuda_check(code) do { _cuda_check((code), __FILE__, __LINE__); } while(0)

static void _cuda_check(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA API returned error: %s at %s line %d\n", cudaGetErrorString(code), file, line);
        exit(1);
    }
}

using namespace nvinfer1;

class Logger : public ILogger           
{
    void log(Severity severity, const char* msg) noexcept override
    {
        // suppress info-level messages
        if (severity <= Severity::kWARNING)
            std::cout << msg << std::endl;
    }
} logger;

std::string slurp_file(std::string path) {
  std::ifstream t(path);
  std::stringstream buffer;
  buffer << t.rdbuf();
  return buffer.str();
}

constexpr int STREAM_COUNT = 1;

struct TensorRTWrapper {
  int max_batch_size;
  // These are created once.
  cudaStream_t streams[STREAM_COUNT];
  IRuntime* runtime;
  float* inp_features[STREAM_COUNT];
  float* out_wdl[STREAM_COUNT];
  float* out_policy[STREAM_COUNT];
  float* out_mcts_value_prediction[STREAM_COUNT];
  // We recreate these for each new model.
  ICudaEngine* engine = nullptr;
  IExecutionContext* contexts[STREAM_COUNT];

  TensorRTWrapper(int cuda_device, int max_batch_size)
    : max_batch_size(max_batch_size)
  {
    cuda_check(cudaSetDevice(cuda_device));
    for (int i = 0; i < STREAM_COUNT; i++)
      cuda_check(cudaStreamCreate(&streams[i]));
    runtime = createInferRuntime(logger);
    // Create buffers for input and output.
    for (int i = 0; i < STREAM_COUNT; i++) {
      cuda_check(cudaMallocManaged(&inp_features[i], max_batch_size * 29 * 8 * 8 * sizeof(float)));
      cuda_check(cudaMallocManaged(&out_wdl[i], max_batch_size * 3 * sizeof(float)));
      cuda_check(cudaMallocManaged(&out_policy[i], max_batch_size * 4096 * sizeof(float)));
      cuda_check(cudaMallocManaged(&out_mcts_value_prediction[i], max_batch_size * sizeof(float)));
      printf("Allocated at %p %p %p %p\n", inp_features[i], out_wdl[i], out_policy[i], out_mcts_value_prediction[i]);
      // Zero out the buffers.
      cuda_check(cudaMemsetAsync(inp_features[i], 0, max_batch_size * 29 * 8 * 8 * sizeof(float), streams[i]));
      cuda_check(cudaMemsetAsync(out_wdl[i], 0, max_batch_size * 3 * sizeof(float), streams[i]));
      cuda_check(cudaMemsetAsync(out_policy[i], 0, max_batch_size * 4096 * sizeof(float), streams[i]));
      cuda_check(cudaMemsetAsync(out_mcts_value_prediction[i], 0, max_batch_size * sizeof(float), streams[i]));
      contexts[i] = nullptr;
    }
  }

  ~TensorRTWrapper() {
    for (int i = 0; i < STREAM_COUNT; i++) {
      cuda_check(cudaFree(inp_features[i]));
      cuda_check(cudaFree(out_wdl[i]));
      cuda_check(cudaFree(out_policy[i]));
      cuda_check(cudaFree(out_mcts_value_prediction[i]));
    }
    for (int i = 0; i < STREAM_COUNT; i++)
      cuda_check(cudaStreamDestroy(streams[i]));
    unload_model();
    runtime->destroy();
  }

  void unload_model() {
    for (int i = 0; i < STREAM_COUNT; i++) {
      if (contexts[i]) {
        contexts[i]->destroy();
        contexts[i] = nullptr;
      }
    }
    if (engine) {
      engine->destroy();
      engine = nullptr;
    }
  }

  void load_model(std::string model_path) {
    unload_model();
    std::string model_data = slurp_file(model_path);
    std::cout << "model_size: " << model_data.size() << std::endl;
    engine = runtime->deserializeCudaEngine(model_data.data(), model_data.size());
    for (auto name : {
      "inp_features",
      "out_wdl",
      "out_policy",
      "out_mcts_value_prediction",
    }) {
      //std::cout << name << ": " << engine->getBindingIndex(name) << std::endl;
      auto dims = engine->getTensorShape(name);
      std::cout << dims.d[0] << " >= " << max_batch_size << std::endl;
      assert(dims.d[0] >= max_batch_size);
      //std::cout << name << ": dims=" << d.nbDims << std::endl;
      //for (int i = 0; i < d.nbDims; i++) {
      //  std::cout << name << ": dim=" << d.d[i] << std::endl;
      //}
    }
    for (int i = 0; i < STREAM_COUNT; i++) {
      contexts[i] = engine->createExecutionContext();
      contexts[i]->setTensorAddress("inp_features", inp_features[i]);
      contexts[i]->setTensorAddress("out_wdl", out_wdl[i]);
      contexts[i]->setTensorAddress("out_policy", out_policy[i]);
      contexts[i]->setTensorAddress("out_mcts_value_prediction", out_mcts_value_prediction[i]);
      printf("Set addresses at %p %p %p\n", inp_features[i], out_wdl[i], out_policy[i]);
    }
  }

  void run_inference(int stream_id) {
    if (contexts[stream_id] == nullptr) {
      std::cerr << "No model loaded!" << std::endl;
      assert(false);
    }
    contexts[stream_id]->enqueueV3(streams[stream_id]);
  }

  void wait_for_inference(int stream_id) {
    cuda_check(cudaStreamSynchronize(streams[stream_id]));
  }
};

extern "C" TensorRTWrapper* TensorRTWrapper_new(int cuda_device, int max_batch_size) {
  return new TensorRTWrapper(cuda_device, max_batch_size);
}

extern "C" void TensorRTWrapper_delete(TensorRTWrapper* wrapper) {
  delete wrapper;
}

extern "C" void TensorRTWrapper_load_model(TensorRTWrapper* wrapper, const char* model_path) {
  wrapper->load_model(model_path);
}

extern "C" void TensorRTWrapper_get_pointers(
  TensorRTWrapper* wrapper,
  int stream_id,
  float** inp_features,
  float** out_wdl,
  float** out_policy
) {
  *inp_features = wrapper->inp_features[stream_id];
  *out_wdl = wrapper->out_wdl[stream_id];
  *out_policy = wrapper->out_policy[stream_id];
  //printf("Returning addresses at %p %p %p\n", *inp_features, *out_wdl, *out_policy);
}

extern "C" void TensorRTWrapper_run_inference(TensorRTWrapper* wrapper, int stream_id) {
  wrapper->run_inference(stream_id);
}

extern "C" void TensorRTWrapper_wait_for_inference(TensorRTWrapper* wrapper, int stream_id) {
  wrapper->wait_for_inference(stream_id);
}
