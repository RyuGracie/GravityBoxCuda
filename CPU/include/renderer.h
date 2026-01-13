#ifndef RENDERER_H
#define RENDERER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "particle.h"

class Renderer {
public:
    Renderer();
    ~Renderer();
    
    bool initialize(int width, int height, const char* title);
    void shutdown();
    
    GLFWwindow* getWindow() { return window; }
    void getWindowSize(int* width, int* height);
    
    void render(const std::vector<Particle>& particles);
    bool shouldClose();
    
private:
    GLFWwindow* window;
    unsigned int vao;
    unsigned int vbo;
    unsigned int shaderProgram;
    int windowWidth;
    int windowHeight;
    
    bool createShaders();
    void setupBuffers();
    void updateProjectionMatrix(int width, int height);
    
    static void framebufferSizeCallback(GLFWwindow* window, int width, int height);
};

#endif