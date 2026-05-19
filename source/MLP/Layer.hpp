#pragma once

#include "../Numora/Numora.hpp"
#include "../Numora/Numora_NN.hpp"
class Layer
{
private:
    std::string label;

public:
    std::string cache_path = "MLP/key/";

    Numora::Matrix weights;
    Numora::Matrix biases;
    Numora::Matrix output_values;
    Numora::Matrix output_values_activated;

    Layer(std::string _label, u32 input_neurons, u32 output_neurons)
    {
        this->label = _label;
        weights = Numora::Matrix(input_neurons, output_neurons);
        biases = Numora::Matrix(output_neurons, 1);

        weights.fill_random(0.1);
        biases.fill_random(0.1);

        load_cache();

    }
    void forward(Numora::Matrix input);

    void load_cache();
    void save_cache();
};

void Layer::forward(Numora::Matrix input)
{
    output_values = (weights.transpose() * input) + biases;
    output_values_activated = NumoraNN::RelU(output_values);
}

void Layer::load_cache()
{
    std::string file_e = ".nncc";
    std::string path = cache_path + label;

    weights.load_cache(path + "_weights" + file_e);
    biases.load_cache(path + "_biases" + file_e);
}


void Layer::save_cache()
{
    std::string file_e = ".nncc";
    std::string path = cache_path + label;

    weights.save_cache(path + "_weights" + file_e);
    biases.save_cache(path + "_biases" + file_e);
}
