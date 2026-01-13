#include "renderer.h"
#include "physics.cuh"
#include <cstdio>

int WinWidth;
int WinHeight;


int main() {
    // Create renderer
    WinHeight = WINDOW_HEIGHT;
    WinWidth = WINDOW_WIDTH;
    Renderer renderer;
    if (!renderer.initialize(WINDOW_WIDTH, WINDOW_HEIGHT, "CUDA Particle Physics")) {
        printf("Failed to initialize renderer\n");
        return -1;
    }
    
    // Create physics simulator
    PhysicsSimulator physics;
    
    // Initialize particles
    glBindBuffer(GL_ARRAY_BUFFER, renderer.getVBO());
    physics.initializeParticles();
    // Register VBO with CUDA
    physics.registerVBO(renderer.getVBO());
    // Main loop
    double lastTime = glfwGetTime();
    
    while (!renderer.shouldClose()) {
        double currentTime = glfwGetTime();
        float dt = (currentTime - lastTime);
        lastTime = currentTime;
        
        // Cap delta time for stability
        if (dt > 0.033f) dt = 0.033f;
        WinHeight = renderer.winHeight;
        WinWidth = renderer.winWidth;
        
        // Update physics
        physics.update(dt, WinWidth, WinHeight);
        
        // Render
        renderer.render();
    }
    
    return 0;
}
