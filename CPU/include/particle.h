#ifndef PARTICLE_H
#define PARTICLE_H

#include <vector>

// Single particle structure
struct Particle {
    float x, y;        // Position
    float vx, vy;      // Velocity
    float r, g, b;     // Color
    float radius;      // Radius
    float mass;        // Mass
};

// Vertex data for rendering (interleaved for OpenGL)
struct ParticleVertex {
    float x, y;        // Position
    float r, g, b;     // Color
    float radius;      // Radius
};

// Constants
const int NUM_PARTICLES = 100000;
const int INITIAL_WINDOW_WIDTH = 1200;
const int INITIAL_WINDOW_HEIGHT = 800;
const float GRAVITY = 500.0f;
const float DAMPING = 0.98f;
const float MIN_RADIUS = 3.0f;
const float MAX_RADIUS = 8.0f;
const float CELL_SIZE = MAX_RADIUS * 2.5f; // Grid cell size

#endif