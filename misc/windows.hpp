#pragma once

#include "../source/reference.hpp"
// #include <GLFW/glfw3.h>
#include <SDL3/SDL.h>
#include "engine.hpp"

class TorchWindows
{
private:
    // GLFWwindow *window;
    SDL_Window* window = NULL;
    SDL_Renderer* renderer = NULL;

    SDL_Event event;

    void init();
    void loop();
    void clean();

public:
    Extent resolution;

    void run();
};