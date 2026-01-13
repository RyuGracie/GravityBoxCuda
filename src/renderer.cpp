#include "renderer.h"
#include <cstdio>
#include <GL/glew.h>

static void framebuffer_size_callback(GLFWwindow *window, int width, int height)
{
    glViewport(0, 0, width, height);

    Renderer *renderer =
        static_cast<Renderer *>(glfwGetWindowUserPointer(window));

    if (renderer)
        renderer->updateProjection(width, height);
}

// Vertex shader
const char *vertexShaderSource = R"(
#version 330 core
layout (location = 0) in vec2 aPos;
layout (location = 1) in vec3 aColor;
layout (location = 2) in float aRadius;

out vec3 color;
uniform mat4 projection;

void main() {
    gl_Position = projection * vec4(aPos, 0.0, 1.0);
    gl_PointSize = aRadius * 2.0;
    color = aColor;
}
)";

// Fragment shader
const char *fragmentShaderSource = R"(
#version 330 core
in vec3 color;
out vec4 FragColor;

void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    if (length(coord) > 0.5)
        discard;
    FragColor = vec4(color, 1.0);
}
)";

static unsigned int compileShader(unsigned int type, const char *source)
{
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);

    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success)
    {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        printf("Shader compilation failed: %s\n", log);
    }
    return shader;
}

Renderer::Renderer()
    : window(nullptr), vao(0), vbo(0), shaderProgram(0)
{
}

Renderer::~Renderer()
{
    shutdown();
}

bool Renderer::initialize(int width, int height, const char *title)
{
    // Initialize GLFW
    if (!glfwInit())
    {
        printf("Failed to initialize GLFW\n");
        return false;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window)
    {
        printf("Failed to create window\n");
        glfwTerminate();
        return false;
    }
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1);

    glewExperimental = GL_TRUE;
    glewInit();
    while (glGetError() != GL_NO_ERROR)
        ;

    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    if (!createShaders())
        return false;

    setupBuffers();
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);

    int fbWidth, fbHeight;
    glfwGetFramebufferSize(window, &fbWidth, &fbHeight);

    glViewport(0, 0, fbWidth, fbHeight);
    updateProjection(fbWidth, fbHeight);


    return true;
}

bool Renderer::createShaders()
{
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);

    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);

    int success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success)
    {
        char log[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, log);
        printf("Shader linking failed: %s\n", log);
        return false;
    }

    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    int widthPx, heightPx;
    glfwGetFramebufferSize(window, &widthPx, &heightPx);
    updateProjection(widthPx, heightPx);

    return true;
}

void Renderer::setupBuffers()
{
    // Create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * sizeof(Particle), NULL, GL_DYNAMIC_DRAW);

    // Create VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(Particle), (void *)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(Particle), (void *)(4 * sizeof(float)));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(Particle), (void *)(7 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
}

void Renderer::render()
{
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);

    glUseProgram(shaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, NUM_PARTICLES);

    // 🔹 FPS COUNTER
    double currentTime = glfwGetTime();
    frameCount++;

    if (currentTime - lastFpsTime >= 1.0)
    {
        double fps = frameCount / (currentTime - lastFpsTime);

        char title[256];
        snprintf(title, sizeof(title),
                 "CUDA Particle Physics | FPS: %.1f", fps);
        glfwSetWindowTitle(window, title);

        frameCount = 0;
        lastFpsTime = currentTime;
    }

    glfwSwapBuffers(window);
    glfwPollEvents();
}


bool Renderer::shouldClose()
{
    return glfwWindowShouldClose(window);
}

void Renderer::shutdown()
{
    if (vao)
        glDeleteVertexArrays(1, &vao);
    if (vbo)
        glDeleteBuffers(1, &vbo);
    if (shaderProgram)
        glDeleteProgram(shaderProgram);
    if (window)
    {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

void Renderer::updateProjection(int fbWidth, int fbHeight)
{
    winHeight = fbHeight;
    winWidth = fbWidth;
    float projection[16] = {
        2.0f / fbWidth, 0.0f, 0.0f, 0.0f,
        0.0f, -2.0f / fbHeight, 0.0f, 0.0f,
        0.0f, 0.0f, 1.0f, 0.0f,
        -1.0f, 1.0f, 0.0f, 1.0f};

    glUseProgram(shaderProgram);
    GLint projLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
}
