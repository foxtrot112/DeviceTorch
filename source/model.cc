#include "model.hh"

void HDSModel::init_random()
{
    weight_a = matrix_create(1024, 128);
    bais_a = matrix_create(128, 1);

    matrix_fill_random(&weight_a, inf_var, sup_var);
    matrix_fill_random(&bais_a, inf_var, sup_var);

    weight_b = matrix_create(128, 64);
    bais_b = matrix_create(64, 1);

    matrix_fill_random(&weight_b, inf_var, sup_var);
    matrix_fill_random(&bais_b, inf_var, sup_var);

    weight_c = matrix_create(64, 10);
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

    // logits = softmax_row(&logits);

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

                matrix weight_c_t = matrix_create(10, 64);
                matrix weight_b_t = matrix_create(64, 128);
                matrix weight_a_t = matrix_create(128, 1024);
                matrix_transpose(&weight_c, &weight_c_t);
                matrix_transpose(&weight_b, &weight_b_t);
                matrix_transpose(&weight_a, &weight_a_t);

                matrix embedding_tilda = matrix_create(128, 1);
                matrix embedding_tilda2 = matrix_create(64, 1);
                matrix embedding_tilda3 = matrix_create(10, 1);

                matrix embedding_tilda_h = matrix_create(128, 1);
                matrix embedding_tilda2_h = matrix_create(64, 1);
                // matrix embedding_tilda3_h = matrix_create(10, 1);

                matrix_multiply(&weight_a_t, &emedding_feature, &embedding_tilda);
                matrix_add(&embedding_tilda, &bais_a, &embedding_tilda);
                embedding_tilda_h = embedding_tilda;
                RELU(&embedding_tilda_h);

                matrix_multiply(&weight_b_t, &embedding_tilda_h, &embedding_tilda2);
                matrix_add(&embedding_tilda2, &bais_b, &embedding_tilda2);
                embedding_tilda2_h = embedding_tilda2;
                RELU(&embedding_tilda2_h);

                matrix_multiply(&weight_c_t, &embedding_tilda2_h, &embedding_tilda3);
                matrix_add(&embedding_tilda3, &bais_c, &embedding_tilda3);
                matrix logits = softmax_col(&embedding_tilda3);
                // RELU(&embedding_tilda3_h);
                std::vector<f32> output;

                for (int i = 0; i < logits.cols; i++)
                {
                    output.push_back(matrix_fetch(&logits, i, 0));
                }
                size_t predicted_digit = std::max_element(output.begin(), output.end()) - output.begin();

                matrix hot_target = matrix_create(10, 1);
                matrix_store(&hot_target,digit,0, 1.0);

                f32 entropy_loss = compute_cost(&logits, &hot_target);

                std::cout << "Digit: " << digit << "Predicted: " << predicted_digit << "Loss: " << entropy_loss << "\n";

                matrix kronicker_delta_03 = matrix_create(10, 1);
                matrix_sub(&logits, &hot_target, &kronicker_delta_03);

                matrix kronicker_delta_02 = matrix_create(64, 1);

                matrix_multiply(&weight_c, &kronicker_delta_03, &kronicker_delta_02);
                RELU_DERIV(&embedding_tilda2, &kronicker_delta_02);

                matrix kronicker_delta_01 = matrix_create(128, 1);

                matrix_multiply(&weight_b, &kronicker_delta_02, &kronicker_delta_01);
                RELU_DERIV(&embedding_tilda, &kronicker_delta_01);

                matrix dJ_dWa = matrix_create(1024, 128);
                matrix dJ_dWb = matrix_create(128, 64);
                matrix dJ_dWc = matrix_create(64, 10);

                matrix_transpose(&kronicker_delta_01, &kronicker_delta_01);
                matrix_outer_product(&emedding_feature, &kronicker_delta_01, &dJ_dWa);

                matrix_transpose(&kronicker_delta_02, &kronicker_delta_02);
                matrix_outer_product(&embedding_tilda, &kronicker_delta_02, &dJ_dWb);

                matrix_transpose(&kronicker_delta_03, &kronicker_delta_03);
                matrix_outer_product(&embedding_tilda2, &kronicker_delta_03, &dJ_dWc);

                float learning_rate = 0.001f;

                matrix_scale(&dJ_dWa, learning_rate);
                matrix_scale(&dJ_dWb, learning_rate);
                matrix_scale(&dJ_dWc, learning_rate);

                matrix_scale(&kronicker_delta_03, learning_rate);
                matrix_scale(&kronicker_delta_02, learning_rate);
                matrix_scale(&kronicker_delta_01, learning_rate);

                matrix_sub(&weight_c,&dJ_dWc,&weight_c);
                matrix_sub(&weight_b,&dJ_dWb,&weight_b);
                matrix_sub(&weight_a,&dJ_dWa,&weight_a);

                matrix_sub(&bais_c,&kronicker_delta_03,&bais_c);
                matrix_sub(&bais_b,&kronicker_delta_02,&bais_b);
                matrix_sub(&bais_a,&kronicker_delta_01,&bais_a);
               
                // matrix dJ_dWa = matrix_create(1024, 128);
                // matrix_outer_product(&emedding_feature, &kronicker_delta_01, &dJ_dWa);

                // matrix dJ_dWa_T = matrix_create(128, 1024);

                // matrix_transpose(&dJ_dWa, &dJ_dWa_T);

                // matrix dJ_dWb = matrix_create(128, 64);

                // matrix_outer_product(&embedding_tilda, &kronicker_delta_02, &dJ_dWb);
                // matrix dJ_dWb_T = matrix_create(64, 128);

                // matrix_transpose(&dJ_dWb, &dJ_dWb_T);

                // matrix dJ_dWc = matrix_create(64, 10);

                // matrix_outer_product(&embedding_tilda2, &kronicker_delta_03, &dJ_dWc);
                // matrix dJ_dWc_T = matrix_create(10, 64);

                // matrix_transpose(&dJ_dWc, &dJ_dWc_T);

                // matrix dJ_dBa = matrix_create(128,1);
                // matrix dJ_dBb = matrix_create(64,1);
                // matrix dJ_dBc = matrix_create(10,1);

                // matrix_transpose(&kronicker_delta_03,&dJ_dBc);
                // matrix_transpose(&kronicker_delta_02,&dJ_dBb);
                // matrix_transpose(&kronicker_delta_01,&dJ_dBa);

                //

                // matrix_scale(&dJ_dBc,learning_rate);
                // matrix_scale(&dJ_dBb,learning_rate);
                // matrix_scale(&dJ_dBa,learning_rate);
                //
                //
                //
                //

                // matrix_sub(&weight_c,&dJ_dWc_T,&weight_c);
                // matrix_sub(&weight_b,&dJ_dWb_T,&weight_b);
                // matrix_sub(&weight_a,&dJ_dWa_T,&weight_a);

                // matrix_sub(&bais_c,&dJ_dBc,&bais_c);
                // matrix_sub(&bais_b,&dJ_dBb,&bais_b);
                // matrix_sub(&bais_a,&dJ_dBa,&bais_a);
                // //
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