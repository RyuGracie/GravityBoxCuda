#include "renderer.h"
#include "physics.cuh"
#include <cstdio>

int main()
{
    // Create renderer
    Renderer renderer;
    if (!renderer.initialize(INITIAL_WINDOW_WIDTH, INITIAL_WINDOW_HEIGHT, "CUDA Particle Physics"))
    {
        printf("Failed to initialize renderer\n");
        return -1;
    }

    // Create physics simulator
    PhysicsSimulator physics;

    // Get initial window size
    int width, height;
    renderer.getWindowSize(&width, &height);

    // Initialize particles
    physics.initializeParticles(width, height);

    // Register VBO with CUDA
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

        // Get current window size
        renderer.getWindowSize(&width, &height);

        // Update physics with current window dimensions
        physics.update(dt, width, height);

        // Render
        renderer.render();
    }

    return 0;
}