#include "Cost.h"

matrix delta_targets(u32 size, u32 offset)
{
    matrix result = matrix_create(size,1);
    matrix_store(&result,offset,0,1);
    
    return result;
} 

f32 compute_cost(matrix *predictions, matrix *targets) 
{    
   f32 sum = 0.0;
   bool is_rows = predictions->cols == 1;

   u32 size =!is_rows ? predictions->cols: predictions->rows;
   for(int i = 0; i < size; i++) {
     sum += targets->data[i] * std::log(predictions->data[i]);
   }

   return sum;
}






