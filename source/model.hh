#pragma once

#include "../misc/labels.hpp"
#include "Cost.h"

class HDSModel
{
public:
    void train();

    void predict(u32 digit);

private:
    ManageLabelImage digitsManager;

    const f32 sup_var = 0.08;
    const f32 inf_var = -0.08;

    matrix weight_a;
    matrix weight_b;
    matrix weight_c;

    matrix bais_a;
    matrix bais_b;
    matrix bais_c;

    void init_random();
    void loadCache();
    void saveCache();
};