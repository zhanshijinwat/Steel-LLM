#define _CRT_SECURE_NO_WARNINGS
#include <torch/all.h>
#include <torch/python.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

#if (defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 700) || defined(USE_ROCM)
// adapted from https://github.com/PanQiWei/AutoGPTQ/blob/main/autogptq_extension/cuda_256/autogptq_cuda_kernel_256.cu
__device__ __forceinline__ void atomicAdd(c10::Half* address, c10::Half val) {
    unsigned int *address_as_ui = reinterpret_cast<unsigned int *>(reinterpret_cast<char *>(address) - (reinterpret_cast<size_t>(address) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        unsigned short hsum = reinterpret_cast<size_t>(address) & 2 ? (old >> 16) : (old & 0xffff);
        hsum += val;
        old = reinterpret_cast<size_t>(address) & 2
                 ? (old & 0xffff) | (hsum << 16)
                 : (old & 0xffff0000) | hsum;
        old = atomicCAS(address_as_ui, assumed, old);

    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);
}
__device__ __forceinline__ void atomicAdd(__half* address, c10::Half val) {
    unsigned int * address_as_ui = (unsigned int *) ((char *)address - ((size_t)address & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;

    do {
        assumed = old;
        __half_raw hsum;
        hsum.x = (size_t)address & 2 ? (old >> 16) : (old & 0xffff);
        half tmpres = __hadd(hsum, val);
        hsum = __half_raw(tmpres);
        old = (size_t)address & 2 ? (old & 0xffff) | (hsum.x << 16) : (old & 0xffff0000) | hsum.x;
        old = atomicCAS(address_as_ui, assumed, old);
    } while (assumed != old);
}
#endif

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width
);

template <typename scalar_t>
__global__ void VecQuant8BatchMatMulColumnCompressionKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
);

template <typename scalar_t>
__global__ void VecQuant4BatchMatMulColumnCompressionKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
);

template <typename scalar_t>
__global__ void VecQuant8BatchMatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
);

template <typename scalar_t>
__global__ void VecQuant4BatchMatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
);



template <typename scalar_t>
__global__ void VecQuant8BatchMatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
);

__global__ void VecQuant8BatchMatMulKernel_faster(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
);



__global__ void VecQuant8BatchMatMulKernel_faster_old(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width
);


template <typename scalar_t>
__global__ void VecQuant4BatchMatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
);


template <typename scalar_t>
__global__ void VecQuant8BatchMatMulColumnCompressionKernel_old(
    const  scalar_t* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
);

__global__ void VecQuant8BatchMatMulColumnCompressionKernel_faster(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
);

__global__ void VecQuant8BatchMatMulColumnCompressionKernel_faster_old(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
);


template <typename scalar_t>
__global__ void VecQuant4BatchMatMulColumnCompressionKernel_old(
    const  scalar_t* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
);


__global__ void VecQuant8BatchMatMulKernel_faster(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width
);


__global__ void VecQuant8BatchMatMulColumnCompressionKernel_faster(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
);

const int BLOCKWIDTH  = 128;
const int BLOCKHEIGHT8 =  32;
const int BLOCKHEIGHT4 =  16;
const int BLOCKHEIGHT_OLD4 =  128;
//const int BLOCKHEIGHT_OLD8 =  128;

__device__ inline unsigned int as_unsigned(int i) {
  return *reinterpret_cast<unsigned int*>(&i);
}

__device__ inline int as_int(int i) {
  return *reinterpret_cast<int*>(&i);
}

void vecquant8matmul_batched_column_compression_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int height = vec.size(3);
  int width = mat.size(3) * 4;

  dim3 blocks(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_batched_cuda", ([&] {
      VecQuant8BatchMatMulColumnCompressionKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, heads, vec_row, height, width
      );
    })
  );

}

template <typename scalar_t>
__global__ void VecQuant8BatchMatMulColumnCompressionKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
) {
  int weight_total = batch * heads * height * width / 4;
  int input_total = batch * heads * vec_row * height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  // h is index of height with step being BLOCKWIDTH
  int h = BLOCKWIDTH * blockIdx.x;
  // w is index of width with step being 1
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= height) {
    return;
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int k;
  scalar_t w_tmp;

  float weight[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h + k < height; ++k){
        int i_w = (w / 4);
        int w_bit = (w % 4) * 8;

        int w_index = (batch_shift * height + h + k) * width / 4 + i_w;
        if (w_index >= weight_total || w >= width) {
          weight[k] = 0;
        } else {
          scalar_t scale = scales[batch_shift * height + h + k];
          scalar_t zero = zeros[batch_shift * height + h + k];
          w_tmp = ((as_unsigned(mat[w_index]) >> w_bit) & 0xFF);
          weight[k] = scale * (w_tmp - zero);
        }
      }

      scalar_t res;
      for (int vr = 0; vr < vec_row; ++vr){
          res = 0;
        int vec_index = (batch_shift * vec_row + vr) * height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = vec[vec_index];
        } else {
            blockvec[tid] = 0;
        }

        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h + k < height; ++k){
          // res is the dot product of BLOCKWIDTH elements (part of width)
            res += weight[k] * blockvec[k];
        }
        // add res to the final result, final matrix shape: (batch, vec_row, width)
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], res);
        }
        __syncthreads();
      }
    }
  }
}

void vecquant8matmul_batched_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int vec_height = vec.size(3);
  int height = mat.size(2);
  int width = mat.size(3);
  int zero_width = zeros.size(2);

  dim3 blocks(
    (height + BLOCKHEIGHT8 - 1) / BLOCKHEIGHT8,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_batched_cuda", ([&] {
      VecQuant8BatchMatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, heads, vec_row, vec_height, height, width, zero_width
      );
    })
  );

}

template <typename scalar_t>
__global__ void VecQuant8BatchMatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
) {
  int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * vec_height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  // h is index of height with step being BLOCKHEIGHT8
  int h = BLOCKHEIGHT8 * blockIdx.x;
  // w is index of width with step being 1
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= vec_height) {
    return;
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  // i is index of mat of block first row
  int i = width * h + w;
  // if (i >= width * height) {
  //   return;
  // }
  int k;
  scalar_t w_tmp;

  int z_w = w / 4;
  int z_mod = (w % 4) * 8;

  float weight[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h * 4 + k < vec_height; ++k){
        int k_w = (k / 4);
        int k_bit = (k % 4) * 8;

        int w_index = batch_shift * height * width + i + (k_w * width);
        if (w_index >= weight_total || w >= width) {
          weight[k] = 0;
        } else {
          scalar_t scale = scales[batch_shift * width + w];
          scalar_t zero;
          if (zero_width == width) {
            zero = zeros[batch_shift * width + w];
          } else {
            zero = scalar_t(((as_unsigned(zeros[batch_shift * zero_width + z_w]) >> z_mod) & 0xFF) + 1);
          }
          w_tmp = ((as_unsigned(mat[w_index]) >> k_bit) & 0xFF);
          weight[k] = scale * (w_tmp - zero);
        }
      }

      scalar_t res;
      for (int vr = 0; vr < vec_row; ++vr){
          res = 0;
        int vec_index = (batch_shift * vec_row + vr) * vec_height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = vec[vec_index];
        } else {
            blockvec[tid] = 0;
        }

        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h * 4 + k < vec_height; ++k){
          // res is the dot product of BLOCKWIDTH elements (part of width)
            res += weight[k] * blockvec[k];
        }
        // add res to the final result, final matrix shape: (batch, vec_row, width)
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], res);
        }
        __syncthreads();
      }
    }
  }
}


void vecquant8matmul_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros,
  torch::Tensor g_idx
) {
  int batch = vec.size(0);
  int vec_height = vec.size(1);
  int height = mat.size(0);
  int width = mat.size(1);
  int zero_width = zeros.size(1);

  dim3 blocks(
    (height + BLOCKHEIGHT8 - 1) / BLOCKHEIGHT8,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_cuda", ([&] {
      VecQuant8MatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(), g_idx.data<int>(),
        batch, vec_height, height, width, zero_width
      );
    })
  );
}

template <typename scalar_t>
__global__ void VecQuant8MatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    const       int* __restrict__ g_idx,
    int batch,
    int vec_height,
    int height,
    int width,
    int zero_width
) {
  int h = BLOCKHEIGHT8 * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + threadIdx.x;

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int g_h = h * 4;
  int k;
  unsigned int g;
  scalar_t w_tmp;

  int z_w = w / 4;
  int z_mod = (w % 4) * 8;

  float weight[BLOCKWIDTH];

  for (k = 0; k <  BLOCKWIDTH; ++k){
    int k_w = (k / 4);
    int k_bit = (k % 4) * 8;

      g = as_int(g_idx[g_h + k]);
      scalar_t scale = scales[g * width + w];
      scalar_t zero = scalar_t(((as_unsigned(zeros[g * zero_width + z_w]) >> z_mod) & 0xFF) + 1);

      w_tmp = ((as_unsigned(mat[i + (k_w * width)]) >> k_bit) & 0xFF);

    weight[k] = scale * (w_tmp - zero);
  }


  scalar_t res;
  for (int b = 0; b < batch; ++b){
      res = 0;
    blockvec[threadIdx.x] = vec[b * vec_height + blockIdx.x * BLOCKWIDTH + threadIdx.x];
    __syncthreads();
    for (k = 0; k <  BLOCKWIDTH; ++k){
      res += weight[k] * blockvec[k];
    }
    atomicAdd(&mul[b * width + w], res);
    __syncthreads();
  }
}



void vecquant4matmul_batched_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int vec_height = vec.size(3);
  int height = mat.size(2);
  int width = mat.size(3);
  int zero_width = zeros.size(2);

  dim3 blocks(
    (height + BLOCKHEIGHT4 - 1) / BLOCKHEIGHT4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_batched_cuda", ([&] {
      VecQuant4BatchMatMulKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, heads, vec_row, vec_height, height, width, zero_width
      );
    })
  );

}

template <typename scalar_t>
__global__ void VecQuant4BatchMatMulKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
) {
  int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * vec_height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  // h is index of height with step being BLOCKHEIGHT4
  int h = BLOCKHEIGHT4 * blockIdx.x;
  // w is index of width with step being 1
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= vec_height) {
    return;
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  // i is index of mat of block first row
  int i = width * h + w;
  int k;
  scalar_t w_tmp;

  int z_w = w / 8;
  int z_mod = (w % 8) * 4;

  float weight[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h * 8 + k < vec_height; ++k){
        int k_w = (k / 8);
        int k_bit = (k % 8) * 4;

        int w_index = batch_shift * height * width + i + (k_w * width);
        if (w_index >= weight_total || w >= width) {
          weight[k] = 0;
        } else {
          scalar_t scale = scales[batch_shift * width + w];
          scalar_t zero;
          if (zero_width == width) {
            zero = zeros[batch_shift * width + w];
          } else {
            zero = scalar_t(((as_unsigned(zeros[batch_shift * zero_width + z_w]) >> z_mod) & 0xF));
          }
          w_tmp = ((as_unsigned(mat[w_index]) >> k_bit) & 0xF);
          weight[k] = scale * (w_tmp - zero);
        }
      }

      scalar_t res;
      for (int vr = 0; vr < vec_row; ++vr){
          res = 0;
        int vec_index = (batch_shift * vec_row + vr) * vec_height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = vec[vec_index];
        } else {
            blockvec[tid] = 0;
        }

        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h * 8 + k < vec_height; ++k){
          // res is the dot product of BLOCKWIDTH elements (part of width)
            res += weight[k] * blockvec[k];
        }
        // add res to the final result, final matrix shape: (batch, vec_row, width)
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], res);
        }
        __syncthreads();
      }
    }
  }
}



void vecquant4matmul_batched_column_compression_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int height = vec.size(3);
  int width = mat.size(3) * 8;

  dim3 blocks(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_batched_cuda", ([&] {
      VecQuant4BatchMatMulColumnCompressionKernel<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<int>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<int>(),
        batch, heads, vec_row, height, width
      );
    })
  );

}

template <typename scalar_t>
__global__ void VecQuant4BatchMatMulColumnCompressionKernel(
    const  scalar_t* __restrict__ vec,
    const       int* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const       int* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
) {
  int weight_total = batch * heads * height * width / 8;
  int input_total = batch * heads * vec_row * height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  // h is index of height with step being BLOCKWIDTH
  int h = BLOCKWIDTH * blockIdx.x;
  // w is index of width with step being 1
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= height) {
    return;
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int k;
  scalar_t w_tmp;

  float weight[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h + k < height; ++k){
        int i_w = (w / 8);
        int w_bit = (w % 8) * 4;

        int w_index = (batch_shift * height + h + k) * width / 8 + i_w;
        if (w_index >= weight_total || w >= width) {
          weight[k] = 0;
        } else {
          scalar_t scale = scales[batch_shift * height + h + k];
          scalar_t zero = zeros[batch_shift * height + h + k];
          w_tmp = ((as_unsigned(mat[w_index]) >> w_bit) & 0xF);
          weight[k] = scale * (w_tmp - zero);
        }
      }

      scalar_t res;
      for (int vr = 0; vr < vec_row; ++vr){
          res = 0;
        int vec_index = (batch_shift * vec_row + vr) * height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = vec[vec_index];
        } else {
            blockvec[tid] = 0;
        }

        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h + k < height; ++k){
          // res is the dot product of BLOCKWIDTH elements (part of width)
            res += weight[k] * blockvec[k];
        }
        // add res to the final result, final matrix shape: (batch, vec_row, width)
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], res);
        }
        __syncthreads();
      }
    }
  }
}


void vecquant8matmul_batched_old_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int vec_height = vec.size(3);
  int height = mat.size(2);
  int width = mat.size(3);
  int zero_width = zeros.size(2);

  dim3 blocks(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_batched_old_cuda", ([&] {
      VecQuant8BatchMatMulKernel_old<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<uint8_t>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        batch, heads, vec_row, vec_height, height, width, zero_width
      );
    })
  );
}


template <typename scalar_t>
__global__ void VecQuant8BatchMatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
) {
  int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * vec_height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  // h is index of height with step being BLOCKHEIGHT8
  int h = BLOCKWIDTH * blockIdx.x;
  // w is index of width with step being 1
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= vec_height) {
    return;
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  // i is index of mat of block first row
  int i = width * h + w;
  int k;
  scalar_t w_tmp;

  float weight[BLOCKWIDTH];
  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h + k < vec_height; ++k){
        int k_w = k;
        int w_index = batch_shift * height * width + i + (k_w * width);
        if (w_index >= weight_total || w >= width) {
          weight[k] = 0;
        } else {
          scalar_t scale = scales[batch_shift * width + w];
          scalar_t zero = zeros[batch_shift * width + w];
          w_tmp = as_unsigned(mat[w_index]);
          weight[k] = scale * (w_tmp - zero);
        }
      }

      scalar_t res;
      for (int vr = 0; vr < vec_row; ++vr){
          res = 0;
        int vec_index = (batch_shift * vec_row + vr) * vec_height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = vec[vec_index];
        } else {
            blockvec[tid] = 0;
        }

        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h + k < vec_height; ++k){
          // res is the dot product of BLOCKWIDTH elements (part of width)
            res += weight[k] * blockvec[k];
        }
        // add res to the final result, final matrix shape: (batch, vec_row, width)
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], res);
        }
        __syncthreads();
      }
    }
  }
}



void vecquant8matmul_batched_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int vec_height = vec.size(3);
  int height = mat.size(2);
  int width = mat.size(3);
  int zero_width = zeros.size(2);

  dim3 blocks(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant8BatchMatMulKernel_faster<<<blocks, threads>>>(
    (half*) vec.data_ptr(),
    (uint8_t*) mat.data_ptr(),
    (half*) mul.data_ptr(),
    (half*) scales.data_ptr(),
    (half*) zeros.data_ptr(),
    batch, heads, vec_row, vec_height, height, width, zero_width
  );
}



__global__ void VecQuant8BatchMatMulKernel_faster(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
) {
  //int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * vec_height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  int h = BLOCKWIDTH * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= height) {
    return;
  }

  __shared__ float blockvec[BLOCKWIDTH];
  int i = width * h + w;
  int k;
  float w_tmp;

  float weight[BLOCKWIDTH];
  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h + k < vec_height; ++k){
        int k_w = k;
        int w_index = batch_shift * height * width + i + (k_w * width);
        float scale = __half2float(scales[batch_shift * width + w]);
        float zero = __half2float(zeros[batch_shift * width + w]);
        w_tmp = as_unsigned(mat[w_index]);
        weight[k] = scale *(w_tmp-zero);
      }

      float res;
      for (int vr = 0; vr < vec_row; ++vr){
        res = 0;
        int vec_index = (batch_shift * vec_row + vr) * vec_height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = __half2float(vec[vec_index]);
        } else {
            blockvec[tid] = 0;
        }
        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h + k < vec_height; ++k){
            float temp_res = weight[k]*blockvec[k];
            res += temp_res;
        }
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], __float2half(res));
        }
        __syncthreads();
      }
    }
  }
}




void vecquant8matmul_batched_column_compression_faster_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int height = vec.size(3);
  int width = mat.size(3);

  dim3 blocks(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant8BatchMatMulColumnCompressionKernel_faster<<<blocks, threads>>>(
    (half*) vec.data_ptr(),
    (uint8_t*) mat.data_ptr(),
    (half*) mul.data_ptr(),
    (half*) scales.data_ptr(),
    (half*) zeros.data_ptr(),
    batch, heads, vec_row, height, width
  );

}

__global__ void VecQuant8BatchMatMulColumnCompressionKernel_faster(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
) {
  //int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  int h = BLOCKWIDTH * blockIdx.x;
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= height) {
    return;
  }

  __shared__ float blockvec[BLOCKWIDTH];
  int k;
  float w_tmp;
  float weight[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH; ++k){
        int w_index = (batch_shift * height + h + k) * width  + w;
        float scale = __half2float(scales[batch_shift * height + h + k]);
        float zero = __half2float(zeros[batch_shift * height + h + k]);
        w_tmp = mat[w_index];
        weight[k] = scale * (w_tmp-zero);
      }

      float res;
      for (int vr = 0; vr < vec_row; ++vr){
        res = 0;
        int vec_index = (batch_shift * vec_row + vr) * height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = __half2float(vec[vec_index]);
        } else {
            blockvec[tid] = 0;
        }
        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH; ++k){
            res += weight[k]*blockvec[k];
        }
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], __float2half(res));
        }
        __syncthreads();
      }
    }
  }
}



void vecquant8matmul_batched_column_compression_old_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int height = vec.size(3);
  int width = mat.size(3);

  dim3 blocks(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant8matmul_batched_column_compression_old_cuda", ([&] {
      VecQuant8BatchMatMulColumnCompressionKernel_old<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<uint8_t>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        batch, heads, vec_row, height, width
      );
    })
  );

}

template <typename scalar_t>
__global__ void VecQuant8BatchMatMulColumnCompressionKernel_old(
    const  scalar_t* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
) {
  int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  // h is index of height with step being BLOCKWIDTH
  int h = BLOCKWIDTH * blockIdx.x;
  // w is index of width with step being 1
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= height) {
    return;
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int k;
  scalar_t w_tmp;

  float weight[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h + k < height; ++k){
        int w_index = (batch_shift * height + h + k) * width  + w;
        if (w_index >= weight_total || w >= width) {
          weight[k] = 0;
        } else {
          scalar_t scale = scales[batch_shift * height + h + k];
          scalar_t zero = zeros[batch_shift * height + h + k];
          w_tmp = mat[w_index];
          weight[k] = scale * (w_tmp - zero);
        }
      }

      scalar_t res;
      for (int vr = 0; vr < vec_row; ++vr){
          res = 0;
        int vec_index = (batch_shift * vec_row + vr) * height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = vec[vec_index];
        } else {
            blockvec[tid] = 0;
        }

        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h + k < height; ++k){
          // res is the dot product of BLOCKWIDTH elements (part of width)
            res += weight[k] * blockvec[k];
        }
        // add res to the final result, final matrix shape: (batch, vec_row, width)
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], res);
        }
        __syncthreads();
      }
    }
  }
}


void vecquant4matmul_batched_old_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int vec_height = vec.size(3);
  int height = mat.size(2);
  int width = mat.size(3);
  int zero_width = zeros.size(2);

  dim3 blocks(
    (height + BLOCKHEIGHT_OLD4 - 1) / BLOCKHEIGHT_OLD4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_batched_old_cuda", ([&] {
      VecQuant4BatchMatMulKernel_old<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<uint8_t>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        batch, heads, vec_row, vec_height, height, width, zero_width
      );
    })
  );

}

template <typename scalar_t>
__global__ void VecQuant4BatchMatMulKernel_old(
    const  scalar_t* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width,
    int zero_width
) {
  int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * vec_height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  // h is index of height with step being BLOCKHEIGHT_OLD4
  int h = BLOCKHEIGHT_OLD4 * blockIdx.x;
  // w is index of width with step being 1
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= vec_height) {
    return;
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  // i is index of mat of block first row
  int i = width * h + w;
  int k;
  scalar_t w_tmp;

  float weight[BLOCKWIDTH];
  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h*2 + k < vec_height; ++k){
        int k_w = (k / 2);
        int k_bit = (k % 2) * 4;
        int w_index = batch_shift * height * width + i + (k_w * width);
        if (w_index >= weight_total || w >= width) {
          weight[k] = 0;
        } else {
          scalar_t scale = scales[batch_shift * width + w];
          scalar_t zero = zeros[batch_shift * width + w];
          w_tmp = ((as_unsigned(mat[w_index]) >> k_bit) & 0xF);
          weight[k] = scale * (w_tmp - zero);
        }
      }

      scalar_t res;
      for (int vr = 0; vr < vec_row; ++vr){
          res = 0;
        int vec_index = (batch_shift * vec_row + vr) * vec_height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = vec[vec_index];
        } else {
            blockvec[tid] = 0;
        }

        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h*2 + k < vec_height; ++k){
          // res is the dot product of BLOCKWIDTH elements (part of width)
            res += weight[k] * blockvec[k];
        }
        // add res to the final result, final matrix shape: (batch, vec_row, width)
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], res);
        }
        __syncthreads();
      }
    }
  }
}





void vecquant4matmul_batched_column_compression_old_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int height = vec.size(3);
  int width = mat.size(3);

  dim3 blocks(
    (height + BLOCKHEIGHT_OLD4 - 1) / BLOCKHEIGHT_OLD4,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  AT_DISPATCH_FLOATING_TYPES(
    vec.type(), "vecquant4matmul_batched_column_compression_old_cuda", ([&] {
      VecQuant4BatchMatMulColumnCompressionKernel_old<<<blocks, threads>>>(
        vec.data<scalar_t>(), mat.data<uint8_t>(), mul.data<scalar_t>(),
        scales.data<scalar_t>(), zeros.data<scalar_t>(),
        batch, heads, vec_row, height, width
      );
    })
  );

}

template <typename scalar_t>
__global__ void VecQuant4BatchMatMulColumnCompressionKernel_old(
    const  scalar_t* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           scalar_t* __restrict__ mul,
    const  scalar_t* __restrict__ scales,
    const  scalar_t* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int height,
    int width
) {
  int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  // h is index of height with step being BLOCKWIDTH
  int h = BLOCKHEIGHT_OLD4 * blockIdx.x;
  // w is index of width with step being 1
  int w = BLOCKWIDTH * blockIdx.y + tid;
  if (w >= width && tid >= height) {
    return;
  }

  __shared__ scalar_t blockvec[BLOCKWIDTH];
  int k;
  scalar_t w_tmp;

  float weight[BLOCKWIDTH];

  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      for (k = 0; k <  BLOCKWIDTH && h*2 + k < height; ++k){
        int k_w = (k / 2);
        int k_bit = (k % 2) * 4;
        int w_index = (batch_shift * height + h + k) * width  + k_w;
        if (w_index >= weight_total || w >= width) {
          weight[k] = 0;
        } else {
          scalar_t scale = scales[batch_shift * height + h + k];
          scalar_t zero = zeros[batch_shift * height + h + k];
          w_tmp = ((as_unsigned(mat[w_index]) >> k_bit) & 0xF);
          weight[k] = scale * (w_tmp - zero);
        }
      }

      scalar_t res;
      for (int vr = 0; vr < vec_row; ++vr){
          res = 0;
        int vec_index = (batch_shift * vec_row + vr) * height + blockIdx.x * BLOCKWIDTH + tid;
        if (vec_index < input_total) {
            blockvec[tid] = vec[vec_index];
        } else {
            blockvec[tid] = 0;
        }

        __syncthreads();
          for (k = 0; k <  BLOCKWIDTH && h*2 + k < height; ++k){
          // res is the dot product of BLOCKWIDTH elements (part of width)
            res += weight[k] * blockvec[k];
        }
        // add res to the final result, final matrix shape: (batch, vec_row, width)
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (out_index < out_total) {
            atomicAdd(&mul[out_index], res);
        }
        __syncthreads();
      }
    }
  }
}





void vecquant8matmul_batched_faster_old_cuda(
  torch::Tensor vec,
  torch::Tensor mat,
  torch::Tensor mul,
  torch::Tensor scales,
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2);
  int vec_height = vec.size(3);
  int height = mat.size(2);
  int width = mat.size(3);

  dim3 blocks(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant8BatchMatMulKernel_faster_old<<<blocks, threads>>>(
    (half*) vec.data_ptr(),
    (uint8_t*) mat.data_ptr(),
    (half*) mul.data_ptr(),
    (half*) scales.data_ptr(),
    (half*) zeros.data_ptr(),
    batch, heads, vec_row, vec_height, height, width
  );
}


__global__ void VecQuant8BatchMatMulKernel_faster_old(
    const  half* __restrict__ vec,
    const  uint8_t* __restrict__ mat,
           half* __restrict__ mul,
    const  half* __restrict__ scales,
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row,
    int vec_height,
    int height,
    int width
) {
 int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * vec_height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  const int BLOCKWIDTH_half = BLOCKWIDTH/2;

  int h = BLOCKWIDTH * blockIdx.x; //head_dim, dim=-1
  int w = BLOCKWIDTH * blockIdx.y + tid; //seq-len, +0-256 ,dim=-2
  /*
  if (w >= width && tid >= vec_height) {
    return;
  }
  */
  __shared__ half blockvec[BLOCKWIDTH]; //256
  int i = width * h + w;
  int k;

  half w_tmp1 = __float2half(0);
  half w_tmp2 = __float2half(0);

  half2 weight[BLOCKWIDTH_half];
  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      //int zero_index = batch_shift;
      for (k = 0; k <  BLOCKWIDTH_half; ++k){
        int w_index1 = batch_shift * height * width + i + (2 * k * width); // [batch,head,h+k, w]
        int w_index2 = batch_shift * height * width + i + ((2 * k + 1) * width);
        int zero_index = batch_shift * width + w; // [batch,head, w]
        if (w_index1 >= weight_total || w >= width || (2 * k + h) >= height) {
          weight[k] = __float2half2_rn(0);
        } else {
            float zero_f=__half2float(zeros[zero_index]);
            float scale_f= __half2float(scales[zero_index]);
            if (w_index2 >= weight_total){
              w_tmp1 = __float2half((as_unsigned(mat[w_index1]) -zero_f)*scale_f);
              w_tmp2 = __float2half(0);
              weight[k] = __halves2half2(w_tmp1,w_tmp2);
              //printf("zero_index is %d w is %d height is %d width is %d w_index1 is %d w_tmp1 is %f w_tmp2 is %f zero is %f scale is %f low is %f high is %f \n ",zero_index,w,height, width,w_index1,__half2float(w_tmp1),__half2float(w_tmp2),zero_f,scale_f,__low2float(weight[k]),__high2float(weight[k]));
            }else{
              w_tmp1 = __int2half_rn(as_unsigned(mat[w_index1]));
              w_tmp2 = __int2half_rn(as_unsigned(mat[w_index2]));

              //weight[k] = __hmul2(__hsub2(__halves2half2(w_tmp1,w_tmp2), __halves2half2(zero,zero)),__halves2half2(scale,scale));
              weight[k] = __hfma2(__halves2half2(w_tmp1,w_tmp2), __float2half2_rn(scale_f), __float2half2_rn(-(scale_f * zero_f)));
              //printf("zero_index1 is %d zero_index2 is %d k is %d head is %d w is %d h is %d height is %d width is %d w_index1 is %d w_index2 is %d zero is %f scale is %f low is %f high is %f \n ",zero_index1,zero_index2,k,head,w,h,height, width,w_index1,w_index2,__half2float(zero1),__half2float(scale1),__low2float(weight[k]),__high2float(weight[k]));
            }
        }
      }


      for (int vr = 0; vr < vec_row; ++vr){
        float res=0;
        int vec_index = (batch_shift * vec_row + vr) * height + blockIdx.x * BLOCKWIDTH + tid;
        int out_index = (batch_shift * vec_row + vr) * width + w;
        if (vec_index < input_total) {
            //blockvec[tid] = __half2float(vec[vec_index]);// [batch, head, vr, tid(seq_len dim+)]
            blockvec[tid] = vec[vec_index];
            //printf("width is %d height is %d h is %d w is %d vec_index is %d out_index is %d vec_row is %d vec_height is %d,vr is %d tid is %d blockvec is %f\n",width,height, h,w,vec_index,out_index,vec_row,vec_height,vr,tid,blockvec[tid]);
        } else {
            blockvec[tid] = __float2half(0);
        }
        __syncthreads();
        if (out_index < out_total) {
          for (k = 0; k <  BLOCKWIDTH_half; ++k){
            half2 res2 = __hmul2(weight[k],__halves2half2(blockvec[2*k],blockvec[2*k+1]));
            res += __low2float(res2) + __high2float(res2);
          }
          atomicAdd(&mul[out_index], __float2half(res));
        }
        __syncthreads();
      }
    }
  }
}


void vecquant8matmul_batched_column_compression_faster_old_cuda(
  torch::Tensor vec,  // [batch,heads, seq_q, seq_v]
  torch::Tensor mat, // [batch,heads, seq_v, head_dim]
  torch::Tensor mul,  // [batch,heads, seq_q,head_dim]
  torch::Tensor scales, // [batch,heads, head_dim]
  torch::Tensor zeros
) {
  int batch = vec.size(0);
  int heads = vec.size(1);
  int vec_row = vec.size(2); //ql
  int height = mat.size(2); //vl
  int width = mat.size(3); //head_dim

  dim3 blocks(
    (height + BLOCKWIDTH - 1) / BLOCKWIDTH,
    (width + BLOCKWIDTH - 1) / BLOCKWIDTH
  );
  dim3 threads(BLOCKWIDTH);

  VecQuant8BatchMatMulColumnCompressionKernel_faster_old<<<blocks, threads>>>(
    (half*) vec.data_ptr(),
    (uint8_t*) mat.data_ptr(),
    (half*) mul.data_ptr(),
    (half*) scales.data_ptr(),
    (half*) zeros.data_ptr(),
    batch, heads, vec_row, height, width
  );

}


__global__ void VecQuant8BatchMatMulColumnCompressionKernel_faster_old(
    const  half* __restrict__ vec,  // [batch,heads, seq_q, seq_v]
    const  uint8_t* __restrict__ mat, // [batch,heads, seq_v, head_dim]
           half* __restrict__ mul, // [batch,heads, seq_q,head_dim]
    const  half* __restrict__ scales, // [batch,heads, seq_v]
    const  half* __restrict__ zeros,
    int batch,
    int heads,
    int vec_row, //seq_q
    int height, //seq_v
    int width //head_dim
) {
  int weight_total = batch * heads * height * width;
  int input_total = batch * heads * vec_row * height;
  int out_total = batch * heads * vec_row * width;
  int tid = threadIdx.x;
  int h = BLOCKWIDTH * blockIdx.x; // vl
  int w = BLOCKWIDTH * blockIdx.y + tid; //head_dim + block
  if (w >= width && tid >= height) {
    return;
  }
  __shared__ half blockvec[BLOCKWIDTH];
  int k;
  half w_tmp1 = __float2half(0);
  half w_tmp2 = __float2half(0);
  int i = width * h + w;
  const int BLOCKWIDTH_half = BLOCKWIDTH/2;
  half2 weight[BLOCKWIDTH_half];

  for (int b = 0; b < batch; ++b){
    for (int head = 0; head < heads; ++head){
      int batch_shift = b * heads + head;
      //int zero_index = batch_shift;
      for (k = 0; k <  BLOCKWIDTH_half; ++k){
        int w_index1 = batch_shift * height * width + i + (2 * k) * width; // [batch,head, h+k, w]
        int w_index2 = batch_shift * height * width + i + ((2 * k + 1) * width);
        int zero_index1 = batch_shift * height + h + 2*k; // [batch,head, w]
        int zero_index2 = batch_shift * height + h + 2*k+1; // [batch,head, w]

        if (w_index1 >= weight_total || (2 * k + h)>=height) {
          weight[k]=__float2half2_rn(0);
        } else{
            //int zero_index = batch_shift + h; // [batch,head, w]
            //float scale_f1 = __half2float(scales[zero_index1]);
            //float zero_f1 =  __half2float(zeros[zero_index1]);
            if (w_index2>=weight_total){
              w_tmp1 = __float2half((as_unsigned(mat[w_index1]) - __half2float(zeros[zero_index1]))* __half2float(scales[zero_index1]));
              w_tmp2 = __float2half(0);
              weight[k] = __halves2half2(w_tmp1,w_tmp2);
              //printf("zero_index is %d k is %d w is %d head is %d height is %d width is %d w_index1 is %d w_tmp1 is %f w_tmp2 is %f zero is %f scale is %f low is %f high is %f \n ",zero_index,k,w,head,height, width,w_index1,__half2float(w_tmp1),__half2float(w_tmp2),zero_f,scale_f,__low2float(weight[k]),__high2float(weight[k]));
            }else{
              w_tmp1 = __int2half_rn(as_unsigned(mat[w_index1]));
              w_tmp2 = __int2half_rn(as_unsigned(mat[w_index2]));
              half zero1=zeros[zero_index1];
              half zero2=zeros[zero_index2];
              half scale1=scales[zero_index1];
              half scale2=scales[zero_index2];
              weight[k] = __hmul2(__hsub2(__halves2half2(w_tmp1,w_tmp2), __halves2half2(zero1,zero2)),__halves2half2(scale1,scale2));
              //weight[k] = __hfma2(__halves2half2(w_tmp1,w_tmp2), __float2half2_rn(scale_f), __float2half2_rn(-(scale_f * zero_f)));
              //printf("zero_index1 is %d zero_index2 is %d k is %d head is %d w is %d h is %d height is %d width is %d w_index1 is %d w_index2 is %d zero is %f scale is %f low is %f high is %f \n ",zero_index1,zero_index2,k,head,w,h,height, width,w_index1,w_index2,__half2float(zero1),__half2float(scale1),__low2float(weight[k]),__high2float(weight[k]));
            }
          }
       }


      for (int vr = 0; vr < vec_row; ++vr){
        float res=0;
        int vec_index = (batch_shift * vec_row + vr) * height + blockIdx.x * BLOCKWIDTH + tid;
        int out_index = (batch_shift * vec_row + vr) * width + w;

        if (vec_index < input_total) {
            //blockvec[tid] = __half2float(vec[vec_index]);
            blockvec[tid] = vec[vec_index];
            //printf("vec_index is %d out_index is %d vec_row is %d ,vr is %d tid is %d blockvec is %f\n",vec_index,out_index,vec_row,vr,tid,blockvec[tid]);
        } else {
            blockvec[tid] = __float2half(0);
            //blockvec[tid] = 0;
        }
        __syncthreads();
        if (out_index < out_total) {
            for (k = 0; k <  BLOCKWIDTH_half; ++k){
                half2 res2 = __hmul2(weight[k],__halves2half2(blockvec[2*k],blockvec[2*k+1]));
                res += __low2float(res2) + __high2float(res2);
            }
            atomicAdd(&mul[out_index], __float2half(res));
        }
        __syncthreads();
      }
    }
  }
}
