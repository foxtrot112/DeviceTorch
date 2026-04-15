#include "Cost.h"

matrix delta_targets(u32 size, u32 offset)
{
  matrix result = matrix_create(size, 1);
  matrix_store(&result, offset, 0, 1);

  return result;
}

f32 compute_cost(matrix *predictions, matrix *targets)
{
  f32 sum = 0.0f;
  const f32 eps = 1e-8f;

  for (int i = 0; i < predictions->cols; i++)
  {
    f32 y = matrix_fetch(targets, 0, i);
    f32 p = matrix_fetch(predictions, 0, i);

    sum += y * std::log(p + eps);
  }

  return -sum;
}

matrix softmax_row(matrix *Z)
{
  matrix result = matrix_create(1, Z->cols);

  // 1. find max for numerical stability
  float max_val = matrix_fetch(Z, 0, 0);
  for (int i = 1; i < Z->cols; i++)
  {
    float v = matrix_fetch(Z, 0, i);
    if (v > max_val)
      max_val = v;
  }

  // 2. compute exp(x - max) and sum
  float sum = 0.0f;
  for (int i = 0; i < Z->cols; i++)
  {
    float e = exp(matrix_fetch(Z, 0, i) - max_val);
    matrix_store(&result, 0, i, e);
    sum += e;
  }

  // 3. normalize
  for (int i = 0; i < Z->cols; i++)
  {
    float v = matrix_fetch(&result, 0, i);
    matrix_store(&result, 0, i, v / sum);
  }

  return result;
}

matrix softmax_col(matrix *Z)
{
  matrix result = matrix_create(Z->rows, 1);
  float sum = 0.0;
  for (int i = 0; i < Z->rows; i++)
  {
    sum += exp(matrix_fetch(Z, i, 0));
  }
  for (int i = 0; i < Z->rows; i++)
  {
    matrix_store(&result, i, 0, exp(matrix_fetch(Z, i, 0)) / sum);
  }
  return result;
}

f32 activationRelu(f32 x)
{
    return (x > 0.0f) ? x : 0.0f;
}

void RELU(matrix *mat)
{
  for (int i = 0; i < mat->rows; i++)
  {
    matrix_store(mat, i, 0, activationRelu(matrix_fetch(mat, i, 0)));
  }
}

void RELU_DERIV(matrix* z, matrix* delta)
{
    // z = pre-activation (or stored activation input to ReLU)
    // delta = gradient flowing backward (modified in-place)

    for (int i = 0; i < z->rows; i++) {
        for (int j = 0; j < z->cols; j++) {

            float z_val = matrix_fetch(z, i, j);

            if (z_val <= 0.0f) {
                matrix_store(delta, i, j, 0.0f);
            }
            // else keep delta as is
        }
    }
}