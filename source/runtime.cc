#include <iostream>
#include "../misc/labels.hpp"
#include "../misc/windows.hpp"
#include "Cost.h"
#include "model.hh"

int main()
{
  TorchWindows Application;
   HDSModel model;

   Application.run();
  //  windows.init();
  //  windows.loop();
  //  windows.clean();
     //model.train();
     //model.predict(4);



  //  ManageLabelImage digitsManager;
  //   digitsManager.fetch_features();
  //   std::cout << digitsManager.labelEmbedding.size() << "\n";
  //   std::cout << digitsManager.labelEmbedding[9].labelEmbedding_number.size() << "\n"; 
  //   std::cout << digitsManager.labelEmbedding[9].labelEmbedding_number[19].rows << "\n";
  //  // model.train();
    //model.predict(9);

//    matrix mat1 = matrix_create(10,10);
//    matrix_fill_kronecker(&mat1,10,1.0);

//    matrix_save_cache(&mat1,"identityTest","keys");
//    matrix mat2 = matrix_create(10,10);
//    matrix_load_cache("identityTest","keys",&mat2); 
   
//    matrix_print(&mat2);

    return 0;
}