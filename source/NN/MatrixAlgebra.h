#pragma once
#include "../reference.hpp"

typedef struct
{
    u32 rows, cols;
    f32 *data;

} matrix;

matrix matrix_create(u32 rows, u32 cols);
f32 matrix_fetch(matrix *mat, u32 row, u32 col);
f32 push_m1x1_to_f32(matrix *matrix1x1);

void matrix_store(matrix *mat, u32 row, u32 col, f32 value);
void matrix_free(matrix *mat);
void matrix_add(matrix *a, matrix *b, matrix *result);
void matrix_sub(matrix *a, matrix *b, matrix *result);
void matrix_multiply(matrix *a, matrix *b, matrix *result);
void matrix_transpose(matrix *mat, matrix *result);
void matrix_print(matrix *mat);
void matrix_fill_kronecker(matrix *mat, u32 size, f32 value);
void matrix_fill_random(matrix *mat, f32 min_value, f32 max_value);
void matrix_outer_product(matrix* a, matrix* b, matrix* result);
void matrix_multiply_transpose(matrix *A, matrix *B, matrix *result);

void matrix_save_cache(matrix *mat,std::string name,std::string location);
void matrix_load_cache(std::string name,std::string location,matrix *mat);
void matrix_scale(matrix* A, float scalar);

#define MC matrix_create
#define MM matrix_multiply
#define MA matrix_add
#define MS matrix_sub

//Abstract Version


