#pragma once
#include "../Numora/Numora.hpp"
#include "../Numora/Numora_NN.hpp"

class AddNorm
{
public:
    Numora::Matrix gamma_v;
    Numora::Matrix beta_v;
    Numora::Matrix embedding_tilda;

    Numora::Matrix mean_v;
    Numora::Matrix var_v;

    f64 eps;

    AddNorm(int dModel)
    {
       gamma_v = Numora::Matrix(dModel,1);
       beta_v = Numora::Matrix(dModel,1);

       eps = 1e-5;
    }

    Numora::Matrix forward(Numora::Matrix X) {
      
    }

    Numora::Matrix backward(Numora::Matrix X,Numora::Matrix gradLY,f32 eta)
    {
        int nt = X.rows;
        int mt = X.cols; //same size as a dModel

        Numora::Matrix gradLZ(nt,mt);
        Numora::Matrix gradL_gamma(mt,1);
        Numora::Matrix gradL_beta(mt,1);

        for(int i = 0; i < nt; i++)
        {
            for(int j = 0; j < mt; j++)
            {
                gradL_beta.store(j,0,gradL_beta.fetch(j,0) + gradLY.fetch(i,j));
                gradL_gamma.store(j,0,gradL_gamma.fetch(j,0) + gradLY.fetch(i,j) * embedding_tilda.fetch(i,j));
            }
        }

        gamma_v = gamma_v - gradL_gamma*eta;
        beta_v = beta_v - gradL_beta*eta;
        
        
    }
};