#include "renderer.h"
#include <cstdio>

// Vertex shader
const char* vertexShaderSource = R"(
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
const char* fragmentShaderSource = R"(
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

static unsigned int compileShader(unsigned int type, const char* source) {
    unsigned int shader = glCreateShader(type);
    glShaderSource(shader, 1, &source, NULL);
    glCompileShader(shader);
    
    int success;
    glGetShaderiv(shader, GL_COMPILE_STATUS, &success);
    if (!success) {
        char log[512];
        glGetShaderInfoLog(shader, 512, NULL, log);
        printf("Shader compilation failed: %s\n", log);
    }
    return shader;
}

Renderer::Renderer() 
    : window(nullptr), vao(0), vbo(0), shaderProgram(0),
      windowWidth(INITIAL_WINDOW_WIDTH), windowHeight(INITIAL_WINDOW_HEIGHT) {
}

Renderer::~Renderer() {
    shutdown();
}

void Renderer::framebufferSizeCallback(GLFWwindow* window, int width, int height) {
    // Get renderer instance from window user pointer
    Renderer* renderer = (Renderer*)glfwGetWindowUserPointer(window);
    if (renderer) {
        renderer->windowWidth = width;
        renderer->windowHeight = height;
        renderer->updateProjectionMatrix(width, height);
    }
}

void Renderer::updateProjectionMatrix(int width, int height) {
    glViewport(0, 0, width, height);
    
    // Update projection matrix
    float projection[16] = {
        2.0f/width, 0, 0, 0,
        0, -2.0f/height, 0, 0,
        0, 0, 1, 0,
        -1, 1, 0, 1
    };
    
    glUseProgram(shaderProgram);
    int projLoc = glGetUniformLocation(shaderProgram, "projection");
    glUniformMatrix4fv(projLoc, 1, GL_FALSE, projection);
}

bool Renderer::initialize(int width, int height, const char* title) {
    windowWidth = width;
    windowHeight = height;
    
    // Initialize GLFW
    if (!glfwInit()) {
        printf("Failed to initialize GLFW\n");
        return false;
    }
    
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
    glfwWindowHint(GLFW_RESIZABLE, GL_TRUE);
    
    window = glfwCreateWindow(width, height, title, NULL, NULL);
    if (!window) {
        printf("Failed to create window\n");
        glfwTerminate();
        return false;
    }
    
    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    
    // Initialize GLEW (must be after context creation)
    glewExperimental = GL_TRUE; // Needed for core profile
    GLenum err = glewInit();
    if (err != GLEW_OK) {
        printf("Failed to initialize GLEW: %s\n", glewGetErrorString(err));
        printf("Trying to continue anyway...\n");
    }
    
    // Clear any GLEW initialization errors (common with core profile)
    while (glGetError() != GL_NO_ERROR);
    
    // Verify OpenGL context
    printf("OpenGL Version: %s\n", glGetString(GL_VERSION));
    printf("GLSL Version: %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));
    
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    
    if (!createShaders()) {
        return false;
    }
    
    setupBuffers();
    
    // Set up resize callback
    glfwSetWindowUserPointer(window, this);
    glfwSetFramebufferSizeCallback(window, framebufferSizeCallback);
    
    return true;
}

bool Renderer::createShaders() {
    unsigned int vertexShader = compileShader(GL_VERTEX_SHADER, vertexShaderSource);
    unsigned int fragmentShader = compileShader(GL_FRAGMENT_SHADER, fragmentShaderSource);
    
    shaderProgram = glCreateProgram();
    glAttachShader(shaderProgram, vertexShader);
    glAttachShader(shaderProgram, fragmentShader);
    glLinkProgram(shaderProgram);
    
    int success;
    glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
    if (!success) {
        char log[512];
        glGetProgramInfoLog(shaderProgram, 512, NULL, log);
        printf("Shader linking failed: %s\n", log);
        return false;
    }
    
    glDeleteShader(vertexShader);
    glDeleteShader(fragmentShader);
    
    // Setup initial projection matrix
    updateProjectionMatrix(windowWidth, windowHeight);
    
    return true;
}

void Renderer::setupBuffers() {
    // Create VBO
    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, NUM_PARTICLES * sizeof(ParticleVertex), NULL, GL_DYNAMIC_DRAW);
    
    // Create VAO
    glGenVertexArrays(1, &vao);
    glBindVertexArray(vao);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)0);
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)(2 * sizeof(float)));
    glVertexAttribPointer(2, 1, GL_FLOAT, GL_FALSE, sizeof(ParticleVertex), (void*)(5 * sizeof(float)));
    glEnableVertexAttribArray(0);
    glEnableVertexAttribArray(1);
    glEnableVertexAttribArray(2);
}

void Renderer::getWindowSize(int* width, int* height) {
    *width = windowWidth;
    *height = windowHeight;
}

void Renderer::render(const std::vector<Particle>& particles) {
    // Copy particle data to VBO
    std::vector<ParticleVertex> vertices(particles.size());
    for (size_t i = 0; i < particles.size(); i++) {
        vertices[i].x = particles[i].x;
        vertices[i].y = particles[i].y;
        vertices[i].r = particles[i].r;
        vertices[i].g = particles[i].g;
        vertices[i].b = particles[i].b;
        vertices[i].radius = particles[i].radius;
    }
    
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferSubData(GL_ARRAY_BUFFER, 0, vertices.size() * sizeof(ParticleVertex), vertices.data());
    
    // Render
    glClearColor(0.1f, 0.1f, 0.15f, 1.0f);
    glClear(GL_COLOR_BUFFER_BIT);
    
    glUseProgram(shaderProgram);
    glBindVertexArray(vao);
    glDrawArrays(GL_POINTS, 0, particles.size());
    
    glfwSwapBuffers(window);
    glfwPollEvents();
}

bool Renderer::shouldClose() {
    return glfwWindowShouldClose(window);
}

void Renderer::shutdown() {
    if (vao) glDeleteVertexArrays(1, &vao);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (shaderProgram) glDeleteProgram(shaderProgram);
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}