#include <torch/all.h>
#include <torch/python.h>
#include <c10/cuda/CUDAGuard.h>

// adapted from https://github.com/PanQiWei/AutoGPTQ/blob/main/autogptq_extension/cuda_256/autogptq_cuda_256.cpp
void vecquant8matmul_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
);

void vecquant8matmul(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros,
  torch::Tensor g_idx
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_cuda(vec, mat, mul, scales, zeros, g_idx);
}

void vecquant8matmul_batched_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_batched(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_batched_cuda(vec, mat, mul, scales, zeros);
}

void vecquant8matmul_batched_column_compression_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_batched_column_compression(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_batched_column_compression_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4matmul_batched_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_batched(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_batched_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4matmul_batched_column_compression_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_batched_column_compression(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_batched_column_compression_cuda(vec, mat, mul, scales, zeros);
}

void vecquant8matmul_batched_old_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_batched_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_batched_old_cuda(vec, mat, mul, scales, zeros);
}


void vecquant4matmul_batched_old_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_batched_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_batched_old_cuda(vec, mat, mul, scales, zeros);
}

void vecquant8matmul_batched_column_compression_old_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_batched_column_compression_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_batched_column_compression_old_cuda(vec, mat, mul, scales, zeros);
}

void vecquant4matmul_batched_column_compression_old_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant4matmul_batched_column_compression_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant4matmul_batched_column_compression_old_cuda(vec, mat, mul, scales, zeros);
}



void vecquant8matmul_batched_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_batched_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_batched_faster_cuda(vec, mat, mul, scales, zeros);
}


void vecquant8matmul_batched_faster_old_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_batched_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_batched_faster_old_cuda(vec, mat, mul, scales, zeros);
}

void vecquant8matmul_batched_column_compression_faster_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_batched_column_compression_faster(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_batched_column_compression_faster_cuda(vec, mat, mul, scales, zeros);
}


void vecquant8matmul_batched_column_compression_faster_old_cuda(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
);

void vecquant8matmul_batched_column_compression_faster_old(
  torch::Tensor vec, torch::Tensor mat, torch::Tensor mul,
  torch::Tensor scales, torch::Tensor zeros
) {
  const at::cuda::OptionalCUDAGuard device_guard(device_of(vec));
  vecquant8matmul_batched_column_compression_faster_old_cuda(vec, mat, mul, scales, zeros);
}



PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("vecquant8matmul", &vecquant8matmul, "Vector 8-bit Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant8matmul_batched", &vecquant8matmul_batched, "Vector 8-bit Batched Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant8matmul_batched_old", &vecquant8matmul_batched_old, "Vector 8-bit old Batched Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant8matmul_batched_faster", &vecquant8matmul_batched_faster, "Vector 8-bit old Batched Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant8matmul_batched_faster_old", &vecquant8matmul_batched_faster_old, "Vector 8-bit old Batched Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant4matmul_batched_old", &vecquant4matmul_batched_old, "Vector 4-bit old Batched Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant8matmul_batched_column_compression", &vecquant8matmul_batched_column_compression, "Vector 8-bit Batched Quantized Matrix Multiplication (CUDA) with weight's column compressed (desc_act)");
  m.def("vecquant8matmul_batched_column_compression_old", &vecquant8matmul_batched_column_compression_old, "Vector old 8-bit Batched Quantized Matrix Multiplication (CUDA) with weight's column compressed (desc_act)");
  m.def("vecquant8matmul_batched_column_compression_faster", &vecquant8matmul_batched_column_compression_faster, "Vector old 8-bit Batched Quantized Matrix Multiplication (CUDA) with weight's column compressed (desc_act)");
  m.def("vecquant8matmul_batched_column_compression_faster_old", &vecquant8matmul_batched_column_compression_faster_old, "Vector old 8-bit Batched Quantized Matrix Multiplication (CUDA) with weight's column compressed (desc_act)");
  m.def("vecquant4matmul_batched_column_compression_old", &vecquant4matmul_batched_column_compression_old, "Vector old 4-bit Batched Quantized Matrix Multiplication (CUDA) with weight's column compressed (desc_act)");
  m.def("vecquant4matmul_batched", &vecquant4matmul_batched, "Vector 4-bit Batched Quantized Matrix Multiplication (CUDA) (desc_act)");
  m.def("vecquant4matmul_batched_column_compression", &vecquant4matmul_batched_column_compression, "Vector 4-bit Batched Quantized Matrix Multiplication (CUDA) with weight's column compressed (desc_act)");
}
