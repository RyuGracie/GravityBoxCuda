#ifndef PHYSICS_H
#define PHYSICS_H

#include "particle.h"
#include <vector>
#include <unordered_map>

class PhysicsSimulator {
public:
    PhysicsSimulator();
    ~PhysicsSimulator();
    
    void update(float dt, int windowWidth, int windowHeight);
    void initializeParticles(int windowWidth, int windowHeight);
    void updateWorldSize(int width, int height);
    const std::vector<Particle>& getParticles() const { return particles; }
    
private:
    std::vector<Particle> particles;
    int currentWidth;
    int currentHeight;
    int gridWidth;
    int gridHeight;
    
    // Spatial grid for collision detection
    std::unordered_map<int, std::vector<int>> spatialGrid;
    
    void updatePhysics(float dt);
    void handleCollisions();
    void updateGridDimensions(int width, int height);
    int computeGridHash(float x, float y);
    void buildSpatialGrid();
};

#endif
