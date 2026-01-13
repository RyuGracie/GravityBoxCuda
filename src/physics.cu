#include "physics.cuh"
#include <cstdlib>
#include <ctime>
#include <cstdio>
#include <GLES2/gl2.h>

// CUDA kernel for physics update
__global__ void updatePhysicsKernel(Particle *particles, int n, float dt, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    Particle &p = particles[idx];

    // Apply gravity
    p.vy += GRAVITY * dt;

    // Update position
    p.x += p.vx * dt;
    p.y += p.vy * dt;

    // Wall collisions with damping
    if (p.x - p.radius < 0)
    {
        p.x = p.radius;
        p.vx = -p.vx * DAMPING;
    }
    if (p.x + p.radius > width)
    {
        p.x = width - p.radius;
        p.vx = -p.vx * DAMPING;
    }
    if (p.y - p.radius < 0)
    {
        p.y = p.radius;
        p.vy = -p.vy * DAMPING;
    }
    if (p.y + p.radius > height)
    {
        p.y = height - p.radius;
        p.vy = -p.vy * DAMPING;
    }
}

// CUDA kernel for particle-particle collisions
__global__ void handleCollisionsKernel(Particle *particles, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    Particle &p1 = particles[idx];

    for (int i = 0; i < n; i++)
    {
        if (i == idx)
            continue;

        Particle &p2 = particles[i];

        float dx = p2.x - p1.x;
        float dy = p2.y - p1.y;
        float dist = sqrtf(dx * dx + dy * dy);
        float minDist = p1.radius + p2.radius;

        if (dist < minDist && dist > 0.1f)
        {
            // Normalize collision vector
            float nx = dx / dist;
            float ny = dy / dist;

            // Relative velocity
            float dvx = p1.vx - p2.vx;
            float dvy = p1.vy - p2.vy;
            float dvn = dvx * nx + dvy * ny;

            // Don't process if particles are separating
            if (dvn < 0)
                continue;

            // Elastic collision response
            float m1 = p1.mass;
            float m2 = p2.mass;
            float impulse = DAMPING * 2.0f * dvn / (m1 + m2);

            p1.vx -= impulse * m2 * nx;
            p1.vy -= impulse * m2 * ny;

            // Separate particles to avoid overlap
            float overlap = minDist - dist;
            float separation = overlap / 2.0f;
            p1.x -= separation * nx;
            p1.y -= separation * ny;
        }
    }
}

PhysicsSimulator::PhysicsSimulator()
    : cudaVboResource(nullptr), threadsPerBlock(256)
{
    blocks = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;
}

PhysicsSimulator::~PhysicsSimulator()
{
    if (cudaVboResource)
    {
        unregisterVBO();
    }
}

void PhysicsSimulator::registerVBO(unsigned int vbo)
{
    cudaGraphicsGLRegisterBuffer(&cudaVboResource, vbo, cudaGraphicsMapFlagsWriteDiscard);
}

void PhysicsSimulator::unregisterVBO()
{
    if (cudaVboResource)
    {
        cudaGraphicsUnregisterResource(cudaVboResource);
        cudaVboResource = nullptr;
    }
}

void PhysicsSimulator::update(float dt, int width, int height)
{
    if (!cudaVboResource)
        return;

    // Map VBO to CUDA
    cudaGraphicsMapResources(1, &cudaVboResource, 0);
    Particle *d_particles;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&d_particles, &num_bytes, cudaVboResource);

    // Run physics kernels
    updatePhysicsKernel<<<blocks, threadsPerBlock>>>(
        d_particles, NUM_PARTICLES, dt, width, height);
    handleCollisionsKernel<<<blocks, threadsPerBlock>>>(d_particles, NUM_PARTICLES);

    cudaDeviceSynchronize();

    // Unmap VBO
    cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
}

void PhysicsSimulator::initializeParticles() {
    srand(time(NULL));
    Particle* h_particles = new Particle[NUM_PARTICLES];

    float cx = WINDOW_WIDTH * 0.5f;
    float cy = WINDOW_HEIGHT * 0.5f;
    float radius = WINDOW_HEIGHT * 0.3f;

    for (int i = 0; i < NUM_PARTICLES; i++) {
        float t = static_cast<float>(i) / NUM_PARTICLES;
        float angle = t * 2.0f * 3.1415926f;

        h_particles[i].x = cx + radius * cosf(angle);
        h_particles[i].y = cy + radius * sinf(angle);

        h_particles[i].vx = (rand() % 200 - 100);
        h_particles[i].vy = (rand() % 100);

        h_particles[i].radius =
            MIN_RADIUS + (rand() % 100) / 100.0f * (MAX_RADIUS - MIN_RADIUS);
        h_particles[i].mass = h_particles[i].radius * h_particles[i].radius;

        h_particles[i].r = (rand() % 100) / 100.0f;
        h_particles[i].g = (rand() % 100) / 100.0f;
        h_particles[i].b = (rand() % 100) / 100.0f;
    }

    glBufferSubData(GL_ARRAY_BUFFER, 0,
                    NUM_PARTICLES * sizeof(Particle), h_particles);
    delete[] h_particles;
}

