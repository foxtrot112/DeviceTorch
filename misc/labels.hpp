#pragma once



#include "../source/NN/MatrixAlgebra.h"

#define LABELSIZE 28u

struct ImageData
{
   int extentWidth;
   int extentHeight;

   int channels;
};

typedef struct {
   std::vector<matrix> labelEmbedding_number;
}labelNumber;


class ManageLabelImage
{
   public: 
     std::vector<u32> imageExtent;
     
     u32 numberExtentBox;
     u32 numberDigits;
      
     std::vector<labelNumber> labelEmbedding;


   void fetch_features();

   private:

    std::string path = "assets/";

    void convolute();
    void resize_features();
 
};


namespace NUMBER_ONE
{
   matrix getNumberOne000();
   matrix getNumberOne001();
   matrix getNumberOne002();
   matrix getNumberOne003();
   matrix getNumberOne004();
   matrix getNumberOne005();
   matrix getNumberOne006();
   matrix getNumberOne007();
   matrix getNumberOne008();
   matrix getNumberOne009();
   matrix getNumberOne010();
};


namespace NUMBER_TWO
{
   matrix getNumberTwo000();
   matrix getNumberTwo001();
   matrix getNumberTwo002();
   matrix getNumberTwo003();
   matrix getNumberTwo004();
   matrix getNumberTwo005();
   matrix getNumberTwo006();
   matrix getNumberTwo007();
   matrix getNumberTwo008();
   matrix getNumberTwo009();
   matrix getNumberTwo010();
};

namespace NUMBER_THREE
{
   matrix getNumberThree000();
   matrix getNumberThree001();
   matrix getNumberThree002();
   matrix getNumberThree003();
   matrix getNumberThree004();
   matrix getNumberThree005();
   matrix getNumberThree006();
   matrix getNumberThree007();
   matrix getNumberThree008();
   matrix getNumberThree009();
   matrix getNumberThree010();
};

namespace NUMBER_FOUR
{
   matrix getNumberFour000();
   matrix getNumberFour001();
   matrix getNumberFour002();
   matrix getNumberFour003();
   matrix getNumberFour004();
   matrix getNumberFour005();
   matrix getNumberFour006();
   matrix getNumberFour007();
   matrix getNumberFour008();
   matrix getNumberFour009();
   matrix getNumberFour010();
};

namespace NUMBER_FIVE
{
   matrix getNumberFive000();
   matrix getNumberFive001();
   matrix getNumberFive002();
   matrix getNumberFive003();
   matrix getNumberFive004();
   matrix getNumberFive005();
   matrix getNumberFive006();
   matrix getNumberFive007();
   matrix getNumberFive008();
   matrix getNumberFive009();
   matrix getNumberFive010();
};
