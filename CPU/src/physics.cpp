#include "physics.h"
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <algorithm>

PhysicsSimulator::PhysicsSimulator() 
    : currentWidth(INITIAL_WINDOW_WIDTH), currentHeight(INITIAL_WINDOW_HEIGHT) {
    updateGridDimensions(currentWidth, currentHeight);
}

PhysicsSimulator::~PhysicsSimulator() {
}

void PhysicsSimulator::updateGridDimensions(int width, int height) {
    gridWidth = (int)ceilf(width / CELL_SIZE);
    gridHeight = (int)ceilf(height / CELL_SIZE);
    int totalCells = gridWidth * gridHeight;
    
    printf("Grid dimensions: %dx%d = %d cells (window: %dx%d)\n", 
           gridWidth, gridHeight, totalCells, width, height);
}

int PhysicsSimulator::computeGridHash(float x, float y) {
    int cellX = (int)(x / CELL_SIZE);
    int cellY = (int)(y / CELL_SIZE);
    
    // Clamp to grid bounds
    cellX = std::max(0, std::min(cellX, gridWidth - 1));
    cellY = std::max(0, std::min(cellY, gridHeight - 1));
    
    return cellY * gridWidth + cellX;
}

void PhysicsSimulator::buildSpatialGrid() {
    spatialGrid.clear();
    
    for (size_t i = 0; i < particles.size(); i++) {
        int hash = computeGridHash(particles[i].x, particles[i].y);
        spatialGrid[hash].push_back(i);
    }
}

void PhysicsSimulator::updatePhysics(float dt) {
    for (auto& p : particles) {
        // Apply gravity
        p.vy += GRAVITY * dt;
        
        // Update position
        p.x += p.vx * dt;
        p.y += p.vy * dt;
        
        // Wall collisions with damping
        if (p.x - p.radius < 0) {
            p.x = p.radius;
            p.vx = -p.vx * DAMPING;
        }
        if (p.x + p.radius > currentWidth) {
            p.x = currentWidth - p.radius;
            p.vx = -p.vx * DAMPING;
        }
        if (p.y - p.radius < 0) {
            p.y = p.radius;
            p.vy = -p.vy * DAMPING;
        }
        if (p.y + p.radius > currentHeight) {
            p.y = currentHeight - p.radius;
            p.vy = -p.vy * DAMPING;
        }
    }
}

void PhysicsSimulator::handleCollisions() {
    // Build spatial grid
    buildSpatialGrid();
    
    // Check collisions using spatial grid
    for (size_t i = 0; i < particles.size(); i++) {
        Particle& p1 = particles[i];
        
        // Get particle's grid cell
        int cellX = (int)(p1.x / CELL_SIZE);
        int cellY = (int)(p1.y / CELL_SIZE);
        
        // Check 3x3 neighboring cells
        for (int dy = -1; dy <= 1; dy++) {
            for (int dx = -1; dx <= 1; dx++) {
                int neighborX = cellX + dx;
                int neighborY = cellY + dy;
                
                // Check bounds
                if (neighborX < 0 || neighborX >= gridWidth || 
                    neighborY < 0 || neighborY >= gridHeight) continue;
                
                int neighborCell = neighborY * gridWidth + neighborX;
                
                // Check if cell has particles
                auto it = spatialGrid.find(neighborCell);
                if (it == spatialGrid.end()) continue;
                
                // Check all particles in this cell
                for (int j : it->second) {
                    if ((int)i >= j) continue; // Avoid duplicate checks
                    
                    Particle& p2 = particles[j];
                    
                    float dx_dist = p2.x - p1.x;
                    float dy_dist = p2.y - p1.y;
                    float dist = sqrtf(dx_dist * dx_dist + dy_dist * dy_dist);
                    float minDist = p1.radius + p2.radius;
                    
                    if (dist < minDist && dist > 0.1f) {
                        // Normalize collision vector
                        float nx = dx_dist / dist;
                        float ny = dy_dist / dist;
                        
                        // Relative velocity
                        float dvx = p1.vx - p2.vx;
                        float dvy = p1.vy - p2.vy;
                        float dvn = dvx * nx + dvy * ny;
                        
                        // Don't process if particles are separating
                        if (dvn < 0) continue;
                        
                        // Elastic collision response
                        float m1 = p1.mass;
                        float m2 = p2.mass;
                        float impulse = 2.0f * dvn / (m1 + m2);
                        
                        p1.vx -= impulse * m2 * nx;
                        p1.vy -= impulse * m2 * ny;
                        p2.vx += impulse * m1 * nx;
                        p2.vy += impulse * m1 * ny;
                        
                        // Separate particles to avoid overlap
                        float overlap = minDist - dist;
                        float separation = overlap / 2.0f;
                        p1.x -= separation * nx;
                        p1.y -= separation * ny;
                        p2.x += separation * nx;
                        p2.y += separation * ny;
                    }
                }
            }
        }
    }
}

void PhysicsSimulator::updateWorldSize(int width, int height) {
    if (width == currentWidth && height == currentHeight) return;
    
    printf("Physics world size updated: %dx%d -> %dx%d\n", 
           currentWidth, currentHeight, width, height);
    
    currentWidth = width;
    currentHeight = height;
    updateGridDimensions(width, height);
}

void PhysicsSimulator::update(float dt, int windowWidth, int windowHeight) {
    // Check if world size changed
    if (windowWidth != currentWidth || windowHeight != currentHeight) {
        updateWorldSize(windowWidth, windowHeight);
    }
    
    // Update physics
    updatePhysics(dt);
    
    // Handle collisions using spatial grid
    handleCollisions();
}

void PhysicsSimulator::initializeParticles(int windowWidth, int windowHeight) {
    srand(time(NULL));
    
    currentWidth = windowWidth;
    currentHeight = windowHeight;
    updateGridDimensions(windowWidth, windowHeight);
    
    particles.clear();
    particles.reserve(NUM_PARTICLES);
    
    for (int i = 0; i < NUM_PARTICLES; i++) {
        Particle p;
        p.x = rand() % windowWidth;
        p.y = rand() % (windowHeight / 2);
        p.vx = (rand() % 200 - 100);
        p.vy = (rand() % 100);
        p.radius = MIN_RADIUS + (rand() % 100) / 100.0f * (MAX_RADIUS - MIN_RADIUS);
        p.mass = p.radius * p.radius;
        p.r = (rand() % 100) / 100.0f;
        p.g = (rand() % 100) / 100.0f;
        p.b = (rand() % 100) / 100.0f;
        
        particles.push_back(p);
    }
}
