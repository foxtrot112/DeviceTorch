#pragma once
#include "Numora.hpp"

namespace NumoraNN
{
    Numora::Matrix RelU(Numora::Matrix input)
    {
        Numora::Matrix result = input;

        for (int i = 0; i < result.rows; i++)
        {
            f32 value = (input.fetch(i, 0) >= 0.0) ? input.fetch(i, 0) : 0.0;
            result.store(i, 0, value);
        }
      return result;  
    }

    Numora::Matrix backRelU(Numora::Matrix input)
    {
        Numora::Matrix result = input;

        for (int i = 0; i < result.rows; i++)
        {
            f32 value = (input.fetch(i, 0) >= 0.0) ? 1.0 : 0.0;
            result.store(i, 0, value);
        }
      return result;  
    }


    Numora::Matrix Softmax(Numora::Matrix input)
    {
        f32 summation = 0.0f;
        for(int i = 0 ; i < input.rows; i++)
        {
            summation += std::exp(input.fetch(i,0));
        }
        Numora::Matrix result = input;
        
        for(int i = 0; i < input.rows; i++)
        {
            result.store(i,0,std::exp(input.fetch(i,0))/summation);
        }

        return result; 
    }

    
    
};
