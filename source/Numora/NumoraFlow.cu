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

#include "NumoraDeviceHandle.cu"

namespace Numora
{
    class Matrix
    {
    public:
        f32 fetch(u32 _row, u32 _col) {
            return this->data[_row * cols + _col];
        }
        void store(u32 _row, u32 _col, f32 value) {
            this->data[_row * cols + _col] = value; 
        }

        u32 rows, cols;
        f32 *data;

    public:
        Matrix() : rows(0), cols(0) {}
        Matrix(u32 _rows, u32 _cols) : rows(_rows), cols(_cols)
        {
            this->data = new f32[rows * cols]();
        }
        void flush();
        
        Matrix transpose();

        void debug_print() {
          for (int i = 0; i < rows; ++i)  {
            for (int j = 0; j < cols; j++) {
               std::cout << fetch(i, j) << " ";
            }
           std::cout << "\n";
          }
        }
        
        public: //operators
        
        Matrix matmul(Matrix other) const;
        // Matrix outer_product(Matrix _other) const;
        // Matrix element_wise(Matrix _other) const;
    };
}




Numora::Matrix Numora::Matrix::matmul(Numora::Matrix other) const
{
    
}



