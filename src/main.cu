#include "renderer.h"
#include "physics.cuh"
#include <cstdio>

int main()
{
    Renderer renderer;
    if (!renderer.initialize(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT, "CUDA Particle Physics"))
    {
        printf("Failed to initialize renderer\n");
        return -1;
    }

    PhysicsSimulator physics;

    int width, height;
    renderer.getWindowSize(&width, &height);

    physics.initializeParticles(width, height);

    physics.registerVBO(renderer.getVBO());

    // Main loop
    double lastTime = glfwGetTime();

    while (!renderer.shouldClose())
    {
        double currentTime = glfwGetTime();
        float dt = (currentTime - lastTime);
        lastTime = currentTime;

        // Cap delta time for stability
        if (dt > 0.033f)
            dt = 0.033f;

        renderer.getWindowSize(&width, &height);

        physics.update(dt, width, height);

        renderer.render();
    }

    return 0;
}