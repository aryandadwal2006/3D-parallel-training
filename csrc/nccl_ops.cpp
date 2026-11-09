#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <cstring>
#include <string>

static ncclComm_t communicator = nullptr;
static c10::cuda::CUDAStream communication_stream;
static int communicator_size = 0;
static int communicator_device = -1;

pybind11::bytes get_unique_id() {
    ncclUniqueId id;
    ncclResult_t status = ncclGetUniqueId(&id);
    TORCH_CHECK(status == ncclSuccess, ncclGetErrorString(status));
    return pybind11::bytes(std::string(reinterpret_cast<char*>(&id), sizeof(id)));
}

void init(pybind11::bytes encoded_id, int rank, int size, int device) {
    TORCH_CHECK(communicator == nullptr, "NCCL communicator already initialized");
    std::string value = encoded_id;
    TORCH_CHECK(value.size() == sizeof(ncclUniqueId), "Invalid NCCL unique ID");
    c10::cuda::CUDAGuard guard(device);
    ncclUniqueId id;
    std::memcpy(&id, value.data(), sizeof(id));
    ncclResult_t status = ncclCommInitRank(&communicator, size, id, rank);
    TORCH_CHECK(status == ncclSuccess, ncclGetErrorString(status));
    communicator_size = size;
    communicator_device = device;
    communication_stream = at::cuda::getStreamFromPool(true, device);
}

int64_t reduce_scatter_async(torch::Tensor input, torch::Tensor output) {
    TORCH_CHECK(communicator != nullptr, "NCCL communicator is not initialized");
    TORCH_CHECK(input.is_cuda() && output.is_cuda(), "Tensors must be CUDA tensors");
    TORCH_CHECK(input.scalar_type() == at::kFloat && output.scalar_type() == at::kFloat, "Only float32 is supported");
    TORCH_CHECK(input.is_contiguous() && output.is_contiguous(), "Tensors must be contiguous");
    TORCH_CHECK(input.get_device() == communicator_device && output.get_device() == communicator_device, "Unexpected CUDA device");
    TORCH_CHECK(input.numel() == output.numel() * communicator_size, "Invalid reduce-scatter tensor sizes");
    c10::cuda::CUDAGuard guard(input.device());
    auto current_stream = at::cuda::getCurrentCUDAStream(communicator_device);
    cudaEvent_t ready;
    cudaError_t event_status = cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
    TORCH_CHECK(event_status == cudaSuccess, cudaGetErrorString(event_status));
    event_status = cudaEventRecord(ready, current_stream.stream());
    TORCH_CHECK(event_status == cudaSuccess, cudaGetErrorString(event_status));
    event_status = cudaStreamWaitEvent(communication_stream.stream(), ready, 0);
    TORCH_CHECK(event_status == cudaSuccess, cudaGetErrorString(event_status));
    ncclResult_t status = ncclReduceScatter(
        input.data_ptr<float>(),
        output.data_ptr<float>(),
        output.numel(),
        ncclFloat32,
        ncclSum,
        communicator,
        communication_stream.stream()
    );
    cudaEventDestroy(ready);
    TORCH_CHECK(status == ncclSuccess, ncclGetErrorString(status));
    return reinterpret_cast<int64_t>(communication_stream.stream());
}

void close() {
    if (communicator != nullptr) {
        ncclCommDestroy(communicator);
        communicator = nullptr;
        communicator_size = 0;
        communicator_device = -1;
    }
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("get_unique_id", &get_unique_id);
    module.def("init", &init);
    module.def("reduce_scatter_async", &reduce_scatter_async);
    module.def("close", &close);
}
