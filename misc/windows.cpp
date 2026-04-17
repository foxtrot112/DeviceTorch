#include "windows.hpp"

void TorchWindows::init()
{
    if (!glfwInit())
        return;
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 1);

    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    /* Create a windowed mode window and its OpenGL context */
    window = glfwCreateWindow(resolution.width, resolution.height, "DeviceTorch", NULL, NULL);
    if (!window)
    {
        glfwTerminate();
        return;
    }

    /* Make the window's context current */
    glfwMakeContextCurrent(window);
}

void TorchWindows::loop()
{
    while (!glfwWindowShouldClose(window))
    {

        glClearColor(0.1f,0.1,0.1, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);
        /* Swap front and back buffers */
        glfwSwapBuffers(window);
        glfwPollEvents();
    }
}

void TorchWindows::clean()
{
    glfwTerminate();
}

void TorchWindows::run()
{
    init();
    loop();
    clean();
}