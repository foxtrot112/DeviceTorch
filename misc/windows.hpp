#pragma once

#include "../source/reference.hpp"
#include <GLFW/glfw3.h>

#include "engine.hpp"

class TorchWindows
{
private:
    GLFWwindow *window;

    void init();
    void loop();
    void clean();

public:
    Extent resolution;

    void run();
};