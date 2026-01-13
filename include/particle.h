#ifndef PARTICLE_H
#define PARTICLE_H

struct Particle {
    float x, y;        // Position
    float vx, vy;      // Velocity
    float r, g, b;     // Color
    float radius;      // Radius
    float mass;        // Mass
};

// Constants
const int NUM_PARTICLES = 1000;
const int WINDOW_WIDTH = 1200;
const int WINDOW_HEIGHT = 800;
const float GRAVITY = 500.0f;
const float DAMPING = 0.98f;
const float MIN_RADIUS = 3.0f;
const float MAX_RADIUS = 8.0f;

#endif