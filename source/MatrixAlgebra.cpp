#include "MatrixAlgebra.h"


matrix matrix_create(u32 rows, u32 cols)
{
    matrix mat;
    mat.rows = rows;
    mat.cols = cols;
    mat.data = new f32[rows * cols]();
    return mat;
}

f32 matrix_fetch(matrix *mat, u32 row, u32 col)
{
    return mat->data[row * mat->cols + col];
}

f32 push_m1x1_to_f32(matrix *matrix1x1)
{
  return matrix_fetch(matrix1x1, 0, 0);
}

void matrix_store(matrix *mat, u32 row, u32 col, f32 value)
{
    mat->data[row * mat->cols + col] = value;
}

void matrix_free(matrix *mat)
{
    delete[] mat->data;
}

void matrix_add(matrix *a, matrix *b, matrix *result)
{
    if (a->rows != b->rows || a->cols != b->cols)
    {
        throw std::invalid_argument("Matrices must have the same dimensions for addition.");
    }
    for (u32 i = 0; i < a->rows; ++i)
    {
        for (u32 j = 0; j < a->cols; ++j)
        {
            f32 sum = matrix_fetch(a, i, j) + matrix_fetch(b, i, j);
            matrix_store(result, i, j, sum);
        }
    }
}

void matrix_multiply(matrix *a, matrix *b, matrix *result)
{
    if (a->cols != b->rows)
    {
        throw std::invalid_argument("Number of columns of the first matrix must equal the number of rows of the second matrix for multiplication.");
    }
    for (u32 i = 0; i < a->rows; ++i)
    {
        for (u32 j = 0; j < b->cols; ++j)
        {
            f32 sum = 0.0f;
            for (u32 k = 0; k < a->cols; ++k)
            {
                sum += matrix_fetch(a, i, k) * matrix_fetch(b, k, j);
            }
            matrix_store(result, i, j, sum);
        }
    }
}

void matrix_transpose(matrix *mat, matrix *result)
{
    for (u32 i = 0; i < mat->rows; ++i)
    {
        for (u32 j = 0; j < mat->cols; ++j)
        {
            f32 value = matrix_fetch(mat, i, j);
            matrix_store(result, j, i, value);
        }
    }
}

void matrix_print(matrix *mat) {
    for (u32 i = 0; i < mat->rows; ++i) {
        for (u32 j = 0; j < mat->cols; ++j) {
            std::cout << matrix_fetch(mat, i, j) << " ";
        }
        std::cout << std::endl;
    }
}

void matrix_fill_kronecker(matrix *mat, u32 size, f32 value)
{
    for(u32 i = 0 ; i < size ; i++) {
       matrix_store(mat,i,i,value);
    }
}

void matrix_fill_random(matrix *mat, f32 min_value, f32 max_value)
{
    for (u32 i = 0; i < mat->rows; ++i)
    {
        for (u32 j = 0; j < mat->cols; ++j)
        {
            f32 random_value = min_value + static_cast<f32>(rand()) / (static_cast<f32>(RAND_MAX / (max_value - min_value)));
            matrix_store(mat, i, j, random_value);
        }
    }
}