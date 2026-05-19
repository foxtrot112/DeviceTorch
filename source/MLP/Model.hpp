#pragma once

#include "Layer.hpp"

class CrossEntropyModel
{
public:
   f32 learningRate = 0.005;

   std::vector<Layer> layers;

   Numora::Matrix output_values;
   Numora::Matrix output_values_softmax;

   void forward(Numora::Matrix inputs)
   {
      layers[0].forward(inputs);
      for (int n = 1; n < layers.size(); n++)
      {
         layers[n].forward(layers[n - 1].output_values_activated);
      }

      output_values = layers.back().output_values_activated;
      output_values_softmax = NumoraNN::Softmax(layers.back().output_values);
   }

   f32 crossEntropyLoss(Numora::Matrix hot_labels)
   {
      f32 sum = 0.0;

      for (int i = 0; i < hot_labels.cols; i++)
      {
         sum += (hot_labels.fetch(i, 0) * std::log(output_values_softmax.fetch(i, 0) + 1e-4));
      }

      return -sum;
   }

   void backward(Numora::Matrix inputs, Numora::Matrix hot_labels)
   {
      /*Using the Cross Entropy Propergation*/

      Numora::Matrix kronecker_delta_H = output_values_softmax - hot_labels;
      Numora::Matrix gradient_weight_H = layers[layers.size() - 2].output_values_activated.outer_product(kronecker_delta_H);

      // layers[layers.size() - 1].biases = layers[layers.size() - 1].biases - (kronecker_delta_H * learningRate);
      // layers[layers.size() - 1].weights = layers[layers.size() - 1].weights - (gradient_weight_H * learningRate);

      layers[layers.size() - 1].save_cache();

      std::vector<Numora::Matrix> kronecker_deltas(layers.size());
      std::vector<Numora::Matrix> gradient_weights(layers.size());

      kronecker_deltas[layers.size() - 1] = kronecker_delta_H;
      gradient_weights[layers.size() - 1] = gradient_weight_H;

      for (int l = layers.size() - 2; l >= 0; l--)
      {
         Numora::Matrix error_deviation = layers[l + 1].weights * kronecker_deltas[l + 1];
         Numora::Matrix back_chain = NumoraNN::backRelU(layers[l].output_values);

         kronecker_deltas[l] = back_chain.element_wise(error_deviation);
  
         Numora::Matrix previous_activation = layers[std::max(l - 1,0)].output_values_activated;
         if (l == 0)
         {
            previous_activation = inputs;
         }

         Numora::Matrix gradient_weight_l = previous_activation.outer_product(kronecker_deltas[l]);
         gradient_weights[l] = gradient_weight_l;

         // Gradient Desent



      }


      for(int l = 0; l < layers.size(); ++l)
      {
         layers[l].biases = layers[l].biases - (kronecker_deltas[l] * learningRate);
         layers[l].weights = layers[l].weights - (gradient_weights[l] * learningRate);
         layers[l].save_cache();
      }
   }
};
