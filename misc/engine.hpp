#pragma once

#include "../source/reference.hpp"
#include <OpenGL/gl.h>
#include <GLFW/glfw3.h>

struct Extent
{
    u32 width = 640, height = 480;
};

class Engine
{
private:
    GLFWwindow *window;
    Renderer renderer;

public:
    void init_opengl();
    void delete_opengl();
};

class ShaderModule
{
public:
    u32 vertex_shader;
    u32 fragment_shader;

    u32 shader_program;

    void compile_shader();
};

class Renderer
{
public:
};
