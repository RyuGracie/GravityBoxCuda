#ifndef RENDERER_H
#define RENDERER_H

#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include "particle.h"

class Renderer
{
public:
    Renderer();
    ~Renderer();

    bool initialize(int width, int height, const char *title);
    void shutdown();

    GLFWwindow *getWindow() { return window; }
    unsigned int getVBO() { return vbo; }

    void getWindowSize(int &height, int &width);

    void render();
    bool shouldClose();

    void updateProjection(int width, int height);
    
    int winWidth = 0;
    int winHeight = 0;

private:

    double lastFpsTime = 0.0;
    int frameCount = 0;


    GLFWwindow *window;
    unsigned int vao;
    unsigned int vbo;
    unsigned int shaderProgram;

    bool createShaders();
    void setupBuffers();
};

#endif