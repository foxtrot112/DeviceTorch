#include "windows.hpp"

void TorchWindows::init()
{
    if (SDL_Init(SDL_INIT_VIDEO) < 0)
    {
        
    }

    window = SDL_CreateWindow("Device",resolution.width,resolution.height,SDL_WINDOW_RESIZABLE);
    renderer = SDL_CreateRenderer(window,nullptr);

}

void TorchWindows::loop()
{
    bool running = true;
    while (running)
    {
        // SDL_Event event;
        // 3. Handle Events
        while (SDL_PollEvent(&event))
        {
            if (event.type == SDL_EVENT_QUIT)
            {
                running = false;
            }
        }
   
    }
}

void TorchWindows::clean()
{
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void TorchWindows::run()
{
    init();
    loop();
    clean();
}