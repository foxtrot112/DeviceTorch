#include "model.hh"

void HDSModel::init_random()
{
    weight_a = matrix_create(128, 32 * 32);
    bais_a = matrix_create(128, 1);

    matrix_fill_random(&weight_a, inf_var, sup_var);
    matrix_fill_random(&bais_a, inf_var, sup_var);

    weight_b = matrix_create(64, 128);
    bais_b = matrix_create(64, 1);

    matrix_fill_random(&weight_b, inf_var, sup_var);
    matrix_fill_random(&bais_b, inf_var, sup_var);

    weight_c = matrix_create(10, 64);
    bais_c = matrix_create(10, 1);

    matrix_fill_random(&weight_c, inf_var, sup_var);
}
void HDSModel::predict(u32 digit)
{
    loadCache();
    digitsManager.fetch_features();

    matrix emedding_feature = digitsManager.labelEmbedding[digit].labelEmbedding_number[2];

    matrix embedding_tilda = matrix_create(128, 1);

    matrix_multiply(&weight_a, &emedding_feature, &embedding_tilda);
    matrix_add(&embedding_tilda, &bais_a, &embedding_tilda);
    RELU(&embedding_tilda);

    matrix embedding_tilda2 = matrix_create(64, 1);

    matrix_multiply(&weight_b, &embedding_tilda, &embedding_tilda2);
    matrix_add(&embedding_tilda2, &bais_b, &embedding_tilda2);
    RELU(&embedding_tilda2);

    matrix embedding_tilda3 = matrix_create(10, 1);

    matrix_multiply(&weight_c, &embedding_tilda2, &embedding_tilda3);
    matrix_add(&embedding_tilda3, &bais_c, &embedding_tilda3);

    matrix logits = matrix_create(1, 10);
    matrix_transpose(&embedding_tilda3, &logits);

    //logits = softmax_row(&logits);

    logits = softmax_row(&logits);

    std::vector<f32> output;

    for (int i = 0; i < logits.cols; i++)
    {
        output.push_back(matrix_fetch(&logits, 0, i));

        std::cout << matrix_fetch(&logits, 0, i) << "\n";
    }

    size_t predicted_digit = std::max_element(output.begin(), output.end()) - output.begin();

    std::cout << "Predicted digit: " << predicted_digit << "\n";
}

void HDSModel::train()
{
    // set up dataset
    init_random();
    // loadCache();

    digitsManager.fetch_features();

    for (int epoch = 0; epoch < 1000; epoch++)
    {
        for (int digit = 0; digit < 10; digit++)
        {
            for (int varriant = 0; varriant < 20; varriant++)
            {

                matrix emedding_feature = digitsManager.labelEmbedding[digit].labelEmbedding_number[varriant];

                matrix embedding_tilda = matrix_create(128, 1);

                matrix_multiply(&weight_a, &emedding_feature, &embedding_tilda);
                matrix_add(&embedding_tilda, &bais_a, &embedding_tilda);
                RELU(&embedding_tilda);

                matrix embedding_tilda2 = matrix_create(64, 1);

                matrix_multiply(&weight_b, &embedding_tilda, &embedding_tilda2);
                matrix_add(&embedding_tilda2, &bais_b, &embedding_tilda2);
                RELU(&embedding_tilda2);

                matrix embedding_tilda3 = matrix_create(10, 1);

                matrix_multiply(&weight_c, &embedding_tilda2, &embedding_tilda3);
                matrix_add(&embedding_tilda3, &bais_c, &embedding_tilda3);

                matrix logits = matrix_create(1, 10);
                matrix_transpose(&embedding_tilda3, &logits);

                logits = softmax_row(&logits);

                std::vector<f32> output;

                for (int i = 0; i < logits.cols; i++)
                {
                    output.push_back(matrix_fetch(&logits, 0, i));
                }

                size_t predicted_digit = std::max_element(output.begin(), output.end()) - output.begin();

                matrix hot_target = matrix_create(1, 10);
                matrix_store(&hot_target, 0, digit, 1.0);

                f32 entropy_loss = compute_cost(&logits, &hot_target);

                std::cout << "Digit: " << digit << "Predicted: " << predicted_digit << "Loss: " << entropy_loss << "\n";

                matrix delta3 = matrix_create(1, 10);

                matrix_sub(&logits, &hot_target, &delta3);

                matrix dW_c = matrix_create(10, 64);
                matrix dW_b = matrix_create(64, 128);
                matrix dW_a = matrix_create(128, 32 * 32);

                matrix_outer_product(&delta3, &embedding_tilda2, &dW_c);

                matrix delta2 = matrix_create(64, 1);
                matrix_multiply_transpose(&weight_c, &delta3, &delta2);
                RELU_DERIV(&embedding_tilda2, &delta2);

                matrix_outer_product(&delta2, &embedding_tilda, &dW_b);

                matrix delta1 = matrix_create(128, 1);
                matrix_multiply_transpose(&weight_b, &delta2, &delta1);
                RELU_DERIV(&embedding_tilda, &delta1);

                matrix_outer_product(&delta1, &emedding_feature, &dW_a);

                f32 lr = 0.01f;
                matrix_scale(&dW_c, lr);
                matrix_scale(&dW_b, lr);
                matrix_scale(&dW_a, lr);

                matrix delta3_rows = matrix_create(10, 1);
                matrix_transpose(&delta3, &delta3_rows);

                matrix_scale(&delta3_rows, lr);
                matrix_scale(&delta2, lr);
                matrix_scale(&delta1, lr);
                //
                matrix_sub(&weight_a, &dW_a, &weight_a);
                matrix_sub(&weight_b, &dW_b, &weight_b);
                matrix_sub(&weight_c, &dW_c, &weight_c);
                //

                matrix_sub(&bais_c, &delta3_rows, &bais_c);
                matrix_sub(&bais_b, &delta2, &bais_b);
                matrix_sub(&bais_a, &delta1, &bais_a);

                matrix_free(&dW_c);
                matrix_free(&dW_b);
                matrix_free(&dW_a);

                matrix_free(&delta2);
                matrix_free(&delta1);

                matrix_free(&embedding_tilda);
                matrix_free(&embedding_tilda2);
                matrix_free(&logits);
            }
        }
        std::cout << "EpochLevel: " << epoch << "\n";

        saveCache();
    }
}

void HDSModel::loadCache()
{
    std::string path = "keys";

    matrix_load_cache("A_weight", path, &weight_a);
    matrix_load_cache("B_weight", path, &weight_b);
    matrix_load_cache("C_weight", path, &weight_c);

    matrix_load_cache("A_Bais", path, &bais_a);
    matrix_load_cache("B_Bais", path, &bais_b);
    matrix_load_cache("C_Bais", path, &bais_c);
}

void HDSModel::saveCache()
{
    std::string path = "keys";

    matrix_save_cache(&weight_a, "A_weight", path);
    matrix_save_cache(&weight_b, "B_weight", path);
    matrix_save_cache(&weight_c, "C_weight", path);

    matrix_save_cache(&bais_a, "A_Bais", path);
    matrix_save_cache(&bais_b, "B_Bais", path);
    matrix_save_cache(&bais_c, "C_Bais", path);
}