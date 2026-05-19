#pragma once

#include <iostream>
#include <vector>
// #include <cuda_runtime.h>

#include <string>
#include <cmath>
#include <vector>
#include <algorithm>
#include <stdexcept>
#include <memory>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <map>
#include <stdint.h>

typedef float f32;
typedef double f64;
typedef uint8_t u8;
typedef uint16_t u16;
typedef uint32_t u32;
typedef uint64_t u64;
typedef int8_t i8;
typedef int16_t i16;
typedef int32_t i32;
typedef int64_t i64;



__global__ void device_matmul(const f32* a, const f32* b, f32* c, i32 row_size_A, i32 contractor_size_K, i32 colum_size_B)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if(row < row_size_A && col < colum_size_B) {
      f32 contraction = 0.0;
        for(int k = 0 ; k < contractor_size_K; k++) {
          contraction += a[row * contractor_size_K + k] * b[k*colum_size_B + col];
        }
        
      c[row * colum_size_B + col];  
    }

}