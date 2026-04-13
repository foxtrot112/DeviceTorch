#pragma once

#include "MatrixAlgebra.h"

matrix delta_targets(u32 size, u32 offset);
f32 compute_cost(matrix *predictions, matrix *targets);
matrix softmax_row(matrix *Z);




