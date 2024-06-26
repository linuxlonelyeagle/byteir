//===- index_select.cu ----------------------------------------*--- C++ -*-===//
//
// Copyright 2022 ByteDance Ltd. and/or its affiliates. All rights reserved.
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
//===----------------------------------------------------------------------===//

#include "./index_select.h"
#include <algorithm>
#include <cuda_fp16.h>

namespace brt {
namespace cuda {
namespace kernel {

template <typename InputTy, typename IndexTy>
__global__ void naive_index_select_kernel(const InputTy *input, const IndexTy *index,
                                          InputTy *output, const int A, const int IB,
                                          const int OB, const int C) {
  for (int outIdx = blockIdx.x * blockDim.x + threadIdx.x; outIdx < A * OB * C;
       outIdx += gridDim.x * blockDim.x) {
    const int ind = outIdx / C % OB;
    const int inpIdx =
        outIdx / (OB * C) * (IB * C) + index[ind] * C + outIdx % C;
    output[outIdx] = input[inpIdx];
  }
}

template <typename InputTy, typename IndexTy>
void index_select(const InputTy *input, const IndexTy *index, InputTy *output, const int A,
                  const int IB, const int OB, const int C,
                  cudaStream_t stream) {
  dim3 grid = std::min(256, (A * OB * C + 63) / 64);
  dim3 block = std::min(64, A * OB * C);
  naive_index_select_kernel<<<grid, block, 0, stream>>>(input, index, output, A,
                                                        IB, OB, C);
}

template void index_select<float, uint32_t>(const float *, const uint32_t *, float *,
                                  const int, const int, const int, const int,
                                  cudaStream_t);

template void index_select<float, int64_t>(const float *, const int64_t *, float *,
                                  const int, const int, const int, const int,
                                  cudaStream_t);

template void index_select<__half, int64_t>(const __half *, const int64_t *, __half *,
                                   const int, const int, const int, const int,
                                   cudaStream_t);
} // namespace kernel
} // namespace cuda
} // namespace brt
