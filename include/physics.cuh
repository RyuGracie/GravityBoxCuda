#ifndef PHYSICS_H
#define PHYSICS_H

#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <random>
#include <cuda_gl_interop.h>
#include "particle.h"

class PhysicsSimulator {
public:
    PhysicsSimulator();
    ~PhysicsSimulator();
    
    void registerVBO(unsigned int vbo);
    void unregisterVBO();
    void update(float dt, int windowWidth, int windowHeight);
    void initializeParticles(int windowWidth, int windowHeight);
    void updateWorldSize(int width, int height);
    
private:
    cudaGraphicsResource* cudaVboResource;
    ParticlesSoA d_particles;  // Device particle data (SoA)
    SpatialGrid d_grid;        // Spatial grid for collision detection
    int threadsPerBlock;
    int blocks;
    int currentWidth;
    int currentHeight;
    int gridWidth;
    int gridHeight;
    int totalCells;
    
    void allocateDeviceMemory();
    void freeDeviceMemory();
    void updateGridDimensions(int width, int height);
};

#endif