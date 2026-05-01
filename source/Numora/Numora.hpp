#pragma once
#include "../reference.hpp"
#include "../NN/MatrixAlgebra.h"


namespace Numora
{
    class Matrix
    {
    public:
        f32 fetch(u32 _row, u32 _col);
        void store(u32 _row, u32 _col, f32 value);

        u32 rows, cols;
        f32 *data;

    public:
        Matrix() : rows(0), cols(0) {}
        Matrix(matrix *legacyMatrix) 
        {
            rows = legacyMatrix->rows;
            cols = legacyMatrix->cols;

            this->data = legacyMatrix->data;
        }

        Matrix(u32 _rows, u32 _cols) : rows(_rows), cols(_cols)
        {
            this->data = new f32[rows * cols]();
        }
        void flush();

        Matrix transpose();

        void save_cache(std::string path);
        void load_cache(std::string path);

        void fill_random(f32 range);
        void debug_print();
        

        public: //operators

        Matrix outer_product(Matrix _other) const;
        Matrix element_wise(Matrix _other) const;


        float operator()(int row, int col);
        Matrix operator*(const Matrix &other) const;
        Matrix operator*(const f32 &other) const;


  
        
        Matrix operator+(const Matrix &other) const;
        Matrix operator-(const Matrix &other) const;
       
    };
}
