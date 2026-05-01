#pragma once
#include "../Numora/Numora.hpp"
#include "../Numora/Numora_NN.hpp"
class PerceptronLayer
{
public:
    Numora::Matrix weights;
    Numora::Matrix biases;
    Numora::Matrix output_neurons_value;
    Numora::Matrix output_neurons_value_activated;

    PerceptronLayer(u32 InputNeurons, u32 OutputNeurons)
    {
        weights = Numora::Matrix(InputNeurons, OutputNeurons);
    }

    void forward(Numora::Matrix input)
    {
        output_neurons_value = (weights.transpose() * input) + biases;
        output_neurons_value_activated = NumoraNN::RelU(output_neurons_value);
    }

    void backward(Numora::Matrix gradient, Numora::Matrix kronecker_delta,const f32 lr)
    {
        weights = weights - (gradient * lr);
        biases = biases - (kronecker_delta * lr);
    }
};