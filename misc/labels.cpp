#include "labels.hpp"

#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

std::vector<f32> acess_data(unsigned char *data, u32 x, u32 y, ImageData information)
{
    u32 index = (y * information.extentWidth + x) * information.channels;
    std::vector<f32> result = {(float)(((int)data[index + 0]) / 255.0), (float)(((int)data[index + 1]) / 255.0), (float)(((int)data[index + 2]) / 255.0)};

    return result;
}

void ManageLabelImage::fetch_features()
{
    const std::string name = "digits.png";
    const std::string filename = path + name;

    ImageData imageInformation{};

    unsigned char *data = stbi_load(filename.c_str(), &imageInformation.extentWidth, &imageInformation.extentHeight, &imageInformation.channels, 0);

    //std::cout << acess_data(data, 3, 3, imageInformation)[0];
    u32 number_digits = 10;
    u32 box_size = 32;
 
    u32 number_refrence = 20;

    for (u32 v = 0; v < number_digits; v++)
    { 
       labelNumber label;

        for (u32 u = 0; u < number_refrence; u++)
        {
            u32 h_offset = u * box_size;
            u32 v_offset = v * box_size;

            matrix embedding = matrix_create(box_size * box_size, 1);

            for (int i = 0; i < box_size; i++)
            {
                for (int j = 0; j < box_size; j++)
                {
                    auto pixel_data = acess_data(data,j + h_offset , i + v_offset, imageInformation);
                    
                    //loat rounded = round(pixel_data[0]);
          
                    matrix_store(&embedding,i*box_size + j,0,pixel_data[0]);
                }
            }

            label.labelEmbedding_number.push_back(embedding);
        }

      labelEmbedding.push_back(label);
    }

    //matrix_print(&labelEmbedding[0].labelEmbedding_number[0]);
}

// matrix NUMBER_ONE::getNumberOne000()
// {
//     matrix result = matrix_create(28u * 28u, 1);
//     std::vector<std::vector<float>> img(28u, std::vector<float>(28u, 0.0f));

//     // Draw a "1" (vertical line)
//     for (int i = 4; i < 24; i++)
//     {
//         img[i][14] = 1.0f; // main vertical stroke
//         img[i][13] = 1.0f; // slight thickness
//     }

//     // Optional: small base
//     for (int j = 12; j <= 16; j++)
//     {
//         img[23][j] = 1.0f;
//     }

//     for (int i = 0; i < 28; i++)
//     {
//         for (int j = 0; j < 28; j++)
//         {
//             matrix_store(&result, j + i * 28u, 0, img[i][j]);
//         }
//     }

//     return result;
// }
