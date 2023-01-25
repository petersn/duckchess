#include <stdio.h>
#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <cassert>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include <math.h>

#ifdef REQUIRE_SINGLE_THREADED_USE
#include <sys/types.h>
#include <unistd.h>
#include <sys/syscall.h>
#endif

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

using DataType = nvinfer1::DataType;

const char* dtype_to_name(DataType dtype) {
  switch (dtype) {
    case DataType::kFLOAT: return "float";
    case DataType::kHALF: return "half";
    case DataType::kINT8: return "int8";
    case DataType::kINT32: return "int32";
    case DataType::kBOOL: return "bool";
    case DataType::kUINT8: return "uint8";
    default: return "unknown";
  }
}

float* very_aligned_float_array(int num_elems) {
  return (float*) aligned_alloc(4096, num_elems * sizeof(float));
}

#ifdef REQUIRE_SINGLE_THREADED_USE
pid_t gettid() {
  return syscall(SYS_gettid);
}
#endif

constexpr int STREAM_COUNT = 2;

struct TensorRTWrapper {
  int max_batch_size;
  // These are created once.
  cudaStream_t streams[STREAM_COUNT];
  IRuntime* runtime;
  float* inp_features_device[STREAM_COUNT];
  float* out_wdl_device[STREAM_COUNT];
  float* out_policy_device[STREAM_COUNT];
  float* out_mcts_value_prediction_device[STREAM_COUNT];
  float* inp_features_host[STREAM_COUNT];
  float* out_wdl_host[STREAM_COUNT];
  float* out_policy_host[STREAM_COUNT];
  float* out_mcts_value_prediction_host[STREAM_COUNT];
  // We recreate these for each new model.
  ICudaEngine* engine = nullptr;
  IExecutionContext* contexts[STREAM_COUNT];
#ifdef REQUIRE_SINGLE_THREADED_USE
  pid_t my_thread_id = -1;
#endif
  // int state_machine = 0;

  TensorRTWrapper(int cuda_device, int max_batch_size)
    : max_batch_size(max_batch_size)
  {
    printf("new TensorRTWrapper(%d, %d)\n", cuda_device, max_batch_size);
    cuda_check(cudaSetDevice(cuda_device));
    for (int i = 0; i < STREAM_COUNT; i++)
      cuda_check(cudaStreamCreate(&streams[i]));
    runtime = createInferRuntime(logger);
    // Create buffers for input and output.
    for (int i = 0; i < STREAM_COUNT; i++) {
      cuda_check(cudaMalloc(&inp_features_device[i], max_batch_size * 37 * 8 * 8 * sizeof(float)));
      cuda_check(cudaMalloc(&out_wdl_device[i], max_batch_size * 3 * sizeof(float)));
      cuda_check(cudaMalloc(&out_policy_device[i], max_batch_size * 4096 * sizeof(float)));
      cuda_check(cudaMalloc(&out_mcts_value_prediction_device[i], max_batch_size * sizeof(float)));
      inp_features_host[i] = very_aligned_float_array(max_batch_size * 37 * 8 * 8);
      out_wdl_host[i] = very_aligned_float_array(max_batch_size * 3);
      out_policy_host[i] = very_aligned_float_array(max_batch_size * 4096);
      out_mcts_value_prediction_host[i] = very_aligned_float_array(max_batch_size);
      printf(
        "Allocated at %p %p %p %p\n",
        inp_features_device[i], out_wdl_device[i], out_policy_device[i], out_mcts_value_prediction_device[i]
      );
      // Zero out the buffers.
      cuda_check(cudaMemsetAsync(inp_features_device[i], 0, max_batch_size * 37 * 8 * 8 * sizeof(float), streams[i]));
      cuda_check(cudaMemsetAsync(out_wdl_device[i], 0, max_batch_size * 3 * sizeof(float), streams[i]));
      cuda_check(cudaMemsetAsync(out_policy_device[i], 0, max_batch_size * 4096 * sizeof(float), streams[i]));
      cuda_check(cudaMemsetAsync(out_mcts_value_prediction_device[i], 0, max_batch_size * sizeof(float), streams[i]));
      contexts[i] = nullptr;
    }
#ifdef REQUIRE_SINGLE_THREADED_USE
    my_thread_id = gettid();
#endif
  }

  ~TensorRTWrapper() {
#ifdef REQUIRE_SINGLE_THREADED_USE
    assert(my_thread_id == gettid());
#endif
    for (int i = 0; i < STREAM_COUNT; i++) {
      cuda_check(cudaFree(inp_features_device[i]));
      cuda_check(cudaFree(out_wdl_device[i]));
      cuda_check(cudaFree(out_policy_device[i]));
      cuda_check(cudaFree(out_mcts_value_prediction_device[i]));
      free(inp_features_host[i]);
      free(out_wdl_host[i]);
      free(out_policy_host[i]);
      free(out_mcts_value_prediction_host[i]);
    }
    for (int i = 0; i < STREAM_COUNT; i++)
      cuda_check(cudaStreamDestroy(streams[i]));
    unload_model();
    runtime->destroy();
  }

  void unload_model() {
#ifdef REQUIRE_SINGLE_THREADED_USE
    assert(my_thread_id == gettid());
#endif
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
#ifdef REQUIRE_SINGLE_THREADED_USE
    assert(my_thread_id == gettid());
#endif
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
      //std::cout << dims.d[0] << " >= " << max_batch_size << std::endl;
      // We now get the dtype of this tensor, and the size of the dtype.
      auto dtype = engine->getTensorDataType(name);
      //auto dtype_name = engine->getDataTypeName(dtype);
      assert(dims.d[0] >= max_batch_size);
      std::cout << name << "(idx=" << engine->getBindingIndex(name) << "): dtype=" << dtype_to_name(dtype) << " shape: ";
      for (int i = 0; i < dims.nbDims; i++) {
        std::cout << dims.d[i] << " ";
      }
      std::cout << std::endl;
    }
    for (int i = 0; i < STREAM_COUNT; i++) {
      contexts[i] = engine->createExecutionContext();
      contexts[i]->setTensorAddress("inp_features", inp_features_device[i]);
      contexts[i]->setTensorAddress("out_wdl", out_wdl_device[i]);
      contexts[i]->setTensorAddress("out_policy", out_policy_device[i]);
      contexts[i]->setTensorAddress("out_mcts_value_prediction", out_mcts_value_prediction_device[i]);
      printf("Set addresses at %p %p %p\n", inp_features_device[i], out_wdl_device[i], out_policy_device[i]);
    }
  }

  void run_inference(int stream_id) {
#ifdef REQUIRE_SINGLE_THREADED_USE
    assert(my_thread_id == gettid());
#endif
    if (contexts[stream_id] == nullptr) {
      std::cerr << "No model loaded!" << std::endl;
      assert(false);
    }
    // Copy the input data to the GPU.
    cuda_check(cudaMemcpyAsync(
      inp_features_device[stream_id],
      inp_features_host[stream_id],
      max_batch_size * 37 * 8 * 8 * sizeof(float),
      cudaMemcpyHostToDevice,
      streams[stream_id]
    ));
    contexts[stream_id]->enqueueV3(streams[stream_id]);
    // Copy the output data back to the CPU.
    cuda_check(cudaMemcpyAsync(
      out_wdl_host[stream_id],
      out_wdl_device[stream_id],
      max_batch_size * 3 * sizeof(float),
      cudaMemcpyDeviceToHost,
      streams[stream_id]
    ));
    cuda_check(cudaMemcpyAsync(
      out_policy_host[stream_id],
      out_policy_device[stream_id],
      max_batch_size * 4096 * sizeof(float),
      cudaMemcpyDeviceToHost,
      streams[stream_id]
    ));
    // assert(state_machine == 0);
    // state_machine = 1;
  }

  void wait_for_inference(int stream_id) {
#ifdef REQUIRE_SINGLE_THREADED_USE
    assert(my_thread_id == gettid());
#endif
    // assert(state_machine == 1);
    cuda_check(cudaStreamSynchronize(streams[stream_id]));
    //cuda_check(cudaDeviceSynchronize());

    // // Check normalization of the policy.
    // for (int i = 0; i < max_batch_size; i++) {
    //   float sum = 0.0f;
    //   for (int j = 0; j < 4096; j++) {
    //     sum += out_policy_host[stream_id][i * 4096 + j];
    //   }
    //   if (fabs(sum - 1.0f) > 1e-5) {
    //     std::cout << "C++: Policy sum is " << sum << " at batch index " << i << std::endl;
    //   }
    // }

    // assert(state_machine == 1);
    // state_machine = 0;
  }
};

extern "C" int get_stream_count() {
  return STREAM_COUNT;
}

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
  assert(stream_id >= 0 && stream_id < STREAM_COUNT);
  *inp_features = wrapper->inp_features_host[stream_id];
  *out_wdl = wrapper->out_wdl_host[stream_id];
  *out_policy = wrapper->out_policy_host[stream_id];
  //printf("Returning addresses at %p %p %p\n", *inp_features, *out_wdl, *out_policy);
}

extern "C" void TensorRTWrapper_run_inference(TensorRTWrapper* wrapper, int stream_id) {
  wrapper->run_inference(stream_id);
}

extern "C" void TensorRTWrapper_wait_for_inference(TensorRTWrapper* wrapper, int stream_id) {
  wrapper->wait_for_inference(stream_id);
}
