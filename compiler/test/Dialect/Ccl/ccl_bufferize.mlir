// RUN: byteir-opt %s  -byteir-one-shot-bufferize -split-input-file | FileCheck %s

func.func @broadcast(%arg0: tensor<2x3x8xf32>) -> tensor<2x3x8xf32> {
  %0 = ccl.broadcast %arg0 {replica_groups = [[2, 3]], synchronous = true} : (tensor<2x3x8xf32>) -> tensor<2x3x8xf32>   
  return %0 : tensor<2x3x8xf32>
}
// CHECK-LABEL:   func.func @broadcast(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<2x3x8xf32>) -> memref<2x3x8xf32> {
// CHECK:           ccl.broadcast %[[VAL_0]] {replica_groups = {{\[\[}}2, 3]], synchronous = true} : (memref<2x3x8xf32>) -> ()
// CHECK:           return %[[VAL_0]] : memref<2x3x8xf32>
// CHECK:         }

// -----

func.func @broadcast_dynamic(%arg0: tensor<2x3x8xf32>, %arg1: tensor<2x1xindex>) -> tensor<2x3x8xf32> {
  %0 = ccl.broadcast %arg0, %arg1 {synchronous = true} : (tensor<2x3x8xf32>, tensor<2x1xindex>) -> tensor<2x3x8xf32>
  return %0 : tensor<2x3x8xf32>
}
// CHECK-LABEL:   func.func @broadcast_dynamic(
// CHECK-SAME:      %[[VAL_0:.*]]: memref<2x3x8xf32>,
// CHECK-SAME:      %[[VAL_1:.*]]: memref<2x1xindex>) -> memref<2x3x8xf32> {
// CHECK:           %[[VAL_2:.*]] = memref.alloc() : memref<2x1xindex>
// CHECK:           memref.copy %[[VAL_1]], %[[VAL_2]] : memref<2x1xindex> to memref<2x1xindex>
// CHECK:           ccl.broadcast %[[VAL_0]], %[[VAL_2]] {synchronous = true} : (memref<2x3x8xf32>, memref<2x1xindex>) -> ()
// CHECK:           return %[[VAL_0]] : memref<2x3x8xf32>
// CHECK:         }

// -----

func.func @main(%arg0: tensor<3xf32>) -> (tensor<3xf32>) {
  %0 = "ccl.send"(%arg0) {sync = true, synchronous=true} : (tensor<3xf32>) -> tensor<3xf32>
  return %0 : tensor<3xf32>
}



