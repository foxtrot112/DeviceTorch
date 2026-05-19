#include <iostream>
#include "Numora/Numora.hpp"

#include "MLP/Model.hpp"

#include "../misc/labels.hpp"

int main()
{
  ManageLabelImage data;
  CrossEntropyModel ceModel;
  ceModel.layers = {Layer("1024to128LTMHRD", 1024, 128), Layer("128to64LTMHRD", 128, 64), Layer("64to10LTMHRD", 64, 10)};

  data.fetch_features();

  Numora::Matrix inputs = Numora::Matrix(&data.labelEmbedding[2].labelEmbedding_number[6]);

  ceModel.forward(inputs);

  Numora::Matrix hot_labels(8, 1);
  hot_labels.store(8, 0, 1.0);

  std::cout << "The Loss: " << ceModel.crossEntropyLoss(hot_labels) << "\n";

  ceModel.output_values_softmax.debug_print();

  // for (int epoch = 0; epoch < 200; epoch++)
  // {
  //   for (int d = 0; d < 10; d++)
  //   {
  //     for (int v = 0; v < 20; v++)
  //     {
  //       u32 digit = d;
  //       u32 varriant = 0;

  //       Numora::Matrix inputs = Numora::Matrix(&data.labelEmbedding[digit].labelEmbedding_number[varriant]);

  //       ceModel.forward(inputs);

  //       Numora::Matrix hot_labels(10, 1);
  //       hot_labels.store(digit, 0, 1.0);

  //       std::cout << "The Loss: " << ceModel.crossEntropyLoss(hot_labels) << "\n";

  //       ceModel.output_values_softmax.debug_print();
  //       ceModel.backward(inputs, hot_labels);
  //     }
  //   }
  //   std::cout << "---------------" << "\n";
  //   std::cout << "Epoch: " << epoch << "\n";
  // }

  return 0;
}