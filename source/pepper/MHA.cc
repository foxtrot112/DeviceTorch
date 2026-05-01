#pragma once

#include "../Numora/Numora.hpp"
#include "../Numora/Numora_NN.hpp"

class SingleAttention
{
private:
    u32 d_head;

public:
    bool post_training = true;
    std::string key_path = "pepper/keys/";
    
    u32 head_index;
    u32 layer_index;

    Numora::Matrix query_weights;
    Numora::Matrix key_weights;
    Numora::Matrix value_weights;

    Numora::Matrix out_score;
    Numora::Matrix out_attention;
    Numora::Matrix out_embedding;

    SingleAttention(u32 heads_number, u32 d_model)
    {
        d_head = (u32)(d_model / heads_number);

        query_weights = Numora::Matrix(d_model, d_head);
        key_weights = Numora::Matrix(d_model, d_head);
        value_weights = Numora::Matrix(d_model, d_head);
    }

    void init();
    void load();

    void forward(Numora::Matrix in_embedding)
    {
        Numora::Matrix query = in_embedding * query_weights;
        Numora::Matrix key = in_embedding * key_weights;
        Numora::Matrix value = in_embedding * value_weights;

        out_score = (query * key.transpose()) * (1.0 / sqrt(d_head));
        out_attention = NumoraNN::Softmax(out_score);

        out_embedding = out_attention * value;
    }
};

void SingleAttention::init()
{
    query_weights.fill_random(0.1);
    key_weights.fill_random(0.1);
    value_weights.fill_random(0.1);

    std::string filename_path_Q = key_path + std::string("query_weight_N") +  std::to_string(layer_index) + "_H" +  std::to_string(head_index);
    std::string filename_path_K = key_path + std::string("key_weight_N") +  std::to_string(layer_index) + "_H" +  std::to_string(head_index);
    std::string filename_path_V = key_path + std::string("value_weight_N") +  std::to_string(layer_index) + "_H" +  std::to_string(head_index);
}

class MultiHeadAttention
{
private:
    u32 heads_number;
    u32 d_model;
    u32 d_head;

public:
    std::vector<SingleAttention> attentions;
    std::vector<Numora::Matrix> concation_identities;

    Numora::Matrix linear_projection_matrix;
    Numora::Matrix out_concate;
    Numora::Matrix out_embedding;

    MultiHeadAttention(u32 _heads_number, u32 _d_model)
    {
     
        d_model = _d_model;
        heads_number = _heads_number;
        d_head = (u32)(d_model / heads_number);

        linear_projection_matrix = Numora::Matrix(d_model,d_model);

        for (int i = 0; i < _heads_number; i++)
        {
            attentions.push_back(SingleAttention(_heads_number, _d_model));
        }
    }

    void load() {

    }

    void concation()
    {
        for (int i = 0; i < heads_number; i++)
        {
            Numora::Matrix identity(d_head, d_model);

            for (int u = 0; u < d_head; u++)
            {
                identity.store(u, u + d_head * i, 1.0);
            }

            concation_identities.push_back(identity);
        }
    }

    void forward(Numora::Matrix in_embedding)
    {
        concation();

        out_concate = Numora::Matrix(in_embedding.rows, d_model);

        for (int i = 0; i < heads_number; i++)
        {
            attentions[i].forward(in_embedding);
            out_concate = out_concate + (attentions[i].out_embedding * concation_identities[i]);
        }

        out_embedding = out_concate * linear_projection_matrix;
    }
};