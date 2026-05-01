#pragma once
#include "Layer.cc"

class MultiLayerPerceptron
{
public:
    std::vector<PerceptronLayer> layers;


    void forward(Numora::Matrix input)
    {
        layers[0].forward(input);

        for (int i = 1; i < layers.size(); i++)
        {
            layers[i].forward(layers[i - 1].output_neurons_value_activated);
        }
    }

    Numora::Matrix get_output()
    {
        return layers.back().output_neurons_value;
    }

    Numora::Matrix get_logits()
    {
        return NumoraNN::Softmax(get_output());
    }

};
