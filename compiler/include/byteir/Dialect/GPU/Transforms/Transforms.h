//===- Transforms.h -------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H
#define BYTEIR_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H

#include "mlir/Dialect/GPU/IR/GPUDialect.h"

namespace mlir {
namespace gpu {

// hoist shared memory alloca in gpu kernel to workgroup arg
void hoistShmAllocaToWorkgroup(gpu::GPUFuncOp func);

} // namespace gpu
} // namespace mlir

#endif // BYTEIR_DIALECT_GPU_TRANSFORMS_TRANSFORMS_H