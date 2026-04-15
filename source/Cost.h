#pragma once

#include "MatrixAlgebra.h"

matrix delta_targets(u32 size, u32 offset);
matrix softmax_row(matrix *Z);

f32 compute_cost(matrix *predictions, matrix *targets);
f32 activationRelu(f32 x);


void RELU(matrix *mat);
void RELU_DERIV(matrix *z, matrix *delta);

