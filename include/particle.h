#ifndef PARTICLE_H
#define PARTICLE_H

// Structure of Arrays (SoA) for better GPU coalescing
struct ParticlesSoA {
    float* x;       // Position X
    float* y;       // Position Y
    float* vx;      // Velocity X
    float* vy;      // Velocity Y
    float* r;       // Color R
    float* g;       // Color G
    float* b;       // Color B
    float* radius;  // Radius
    float* mass;    // Mass
};

// Vertex data for rendering (interleaved for OpenGL)
struct ParticleVertex {
    float x, y;        // Position
    float r, g, b;     // Color
    float radius;      // Radius
};

// Spatial grid for collision detection
struct SpatialGrid {
    int* cellStart;    // Start index of particles in each cell
    int* cellEnd;      // End index of particles in each cell
    int* particleHash; // Grid cell hash for each particle
    int* particleIndex;// Original particle index after sorting
};

// Constants
const int NUM_PARTICLES = 100000;
const int INITIAL_WINDOW_WIDTH = 2100;
const int INITIAL_WINDOW_HEIGHT = 1400;
const float GRAVITY = 980.0f;
const float DAMPING = 0.90f;
const float MIN_RADIUS = 2.0f;
const float MAX_RADIUS = 6.0f;
const float CELL_SIZE = MAX_RADIUS * 2.5f; // Grid cell size

#endif