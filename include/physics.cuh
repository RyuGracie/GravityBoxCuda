#ifndef PHYSICS_H
#define PHYSICS_H

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include "particle.h"

class PhysicsSimulator {
public:
    PhysicsSimulator();
    ~PhysicsSimulator();
    
    void registerVBO(unsigned int vbo);
    void unregisterVBO();
    void update(float dt, int width, int height);
    void initializeParticles();
    
private:
    cudaGraphicsResource* cudaVboResource;
    int threadsPerBlock;
    int blocks;
};

#endif