//===- PassDetail.h -------------------------------------------*--- C++ -*-===//
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

#ifndef BYTEIR_DIALECT_MEMREF_TRANSFORMS_PASSDETAIL_H
#define BYTEIR_DIALECT_MEMREF_TRANSFORMS_PASSDETAIL_H

#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/Pass.h"

// forward dialects for conversions
namespace mlir {
class AffineDialect;
namespace arith {
class ArithDialect;
} // namespace arith

namespace func {
class FuncOp;
} // namespace func

namespace memref {
class MemRefDialect;
} // namespace memref

#define GEN_PASS_CLASSES
#include "byteir/Dialect/MemRef/Passes.h.inc"

} // namespace mlir

#endif // BYTEIR_DIALECT_MEMREF_TRANSFORMS_PASSDETAIL_H
