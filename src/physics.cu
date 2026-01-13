#include "physics.cuh"
#include <algorithm>

// Compute grid hash for a particle
__device__ int computeGridHash(float x, float y, int gridWidth, int gridHeight)
{
    int cellX = (int)(x / CELL_SIZE);
    int cellY = (int)(y / CELL_SIZE);

    cellX = max(0, min(cellX, gridWidth - 1));
    cellY = max(0, min(cellY, gridHeight - 1));

    return cellY * gridWidth + cellX;
}

// CUDA kernel to compute particle hashes
__global__ void computeHashKernel(
    int *particleHash, int *particleIndex,
    float *x, float *y, int n, int gridWidth, int gridHeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    particleHash[idx] = computeGridHash(x[idx], y[idx], gridWidth, gridHeight);
    particleIndex[idx] = idx;
}

// CUDA kernel to find cell boundaries after sorting
__global__ void findCellBoundariesKernel(
    int *cellStart,
    int *cellEnd,
    const int *particleHash,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int hash = particleHash[idx];

    if (idx == 0)
    {
        cellStart[hash] = 0;
    }
    else
    {
        int prevHash = particleHash[idx - 1];
        if (prevHash != hash)
        {
            cellStart[hash] = idx;
            cellEnd[prevHash] = idx;
        }
    }

    if (idx == n - 1)
    {
        cellEnd[hash] = n;
    }
}

// CUDA kernel for physics update using Structure of Arrays
__global__ void updatePhysicsKernel(
    float *x, float *y, float *vx, float *vy,
    float *radius, int n, float dt, int width, int height)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    // Apply gravity
    vy[idx] += GRAVITY * dt;

    // Update position
    x[idx] += vx[idx] * dt;
    y[idx] += vy[idx] * dt;

    // Wall collisions with damping
    float r = radius[idx];

    if (x[idx] - r < 0)
    {
        x[idx] = r;
        vx[idx] = -vx[idx] * DAMPING;
    }
    if (x[idx] + r > width)
    {
        x[idx] = width - r;
        vx[idx] = -vx[idx] * DAMPING;
    }
    if (y[idx] - r < 0)
    {
        y[idx] = r;
        vy[idx] = -vy[idx] * DAMPING;
    }
    if (y[idx] + r > height)
    {
        y[idx] = height - r;
        vy[idx] = -vy[idx] * DAMPING;
    }
}

// CUDA kernel for particle-particle collisions using spatial grid
__global__ void handleCollisionsGridKernel(
    float *x, float *y, float *vx, float *vy,
    float *radius, float *mass,
    int *cellStart, int *cellEnd, int *particleHash, int *particleIndex,
    int n, int gridWidth, int gridHeight)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    int originalIdx = particleIndex[idx];

    float x1 = x[originalIdx];
    float y1 = y[originalIdx];
    float vx1 = vx[originalIdx];
    float vy1 = vy[originalIdx];
    float r1 = radius[originalIdx];
    float m1 = mass[originalIdx];

    float xdisp = 0.0f;
    float ydisp = 0.0f;
    float xspd = 0.0f;
    float yspd = 0.0f;
    int colcount = 0;

    int cellX = (int)(x1 / CELL_SIZE);
    int cellY = (int)(y1 / CELL_SIZE);

    // Check 3x3 neighboring cells
    for (int dy = -1; dy <= 1; dy++)
    {
        for (int dx = -1; dx <= 1; dx++)
        {
            int neighborX = cellX + dx;
            int neighborY = cellY + dy;

            if (neighborX < 0 || neighborX >= gridWidth ||
                neighborY < 0 || neighborY >= gridHeight)
                continue;

            int neighborCell = neighborY * gridWidth + neighborX;
            int start = cellStart[neighborCell];
            int end = cellEnd[neighborCell];

            // Check all particles in this cell
            for (int j = start; j < end; j++)
            {
                int otherIdx = particleIndex[j];
                if (otherIdx == originalIdx)
                    continue;

                float dx_dist = x[otherIdx] - x1;
                float dy_dist = y[otherIdx] - y1;
                float dist = sqrtf(dx_dist * dx_dist + dy_dist * dy_dist);
                float minDist = r1 + radius[otherIdx];

                if (dist < minDist && dist > 1e-6f)
                {
                    // Normalize collision vector
                    float nx = dx_dist / dist;
                    float ny = dy_dist / dist;

                    // Relative velocity
                    float dvx_rel = vx1 - vx[otherIdx];
                    float dvy_rel = vy1 - vy[otherIdx];
                    float dvn = dvx_rel * nx + dvy_rel * ny;

                    // Don't process if particles are separating
                    if (dvn <= 0)
                        continue;

                    float m2 = mass[otherIdx];
                    float impulse = DAMPING * 2.0f * dvn / (m1 + m2);

                    xspd += impulse * m2 * nx;
                    yspd += impulse * m2 * ny;

                    float overlap = minDist - dist;
                    float separation = overlap;
                    xdisp += separation * nx;
                    ydisp += separation * ny;

                    colcount++;
                }
            }
        }
    }
    float avg = colcount > 1 ? 2.0f / colcount : 1.0f;
    x[originalIdx] -= xdisp * avg;
    y[originalIdx] -= ydisp * avg;
    vx[originalIdx] -= xspd * avg;
    vy[originalIdx] -= yspd * avg;
}

// CUDA kernel to copy SoA data to interleaved VBO format
__global__ void copyToVBOKernel(
    ParticleVertex *vbo,
    float *x, float *y, float *r, float *g, float *b, float *radius,
    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n)
        return;

    vbo[idx].x = x[idx];
    vbo[idx].y = y[idx];
    vbo[idx].r = r[idx];
    vbo[idx].g = g[idx];
    vbo[idx].b = b[idx];
    vbo[idx].radius = radius[idx];
}

PhysicsSimulator::PhysicsSimulator()
    : cudaVboResource(nullptr), threadsPerBlock(256),
      currentWidth(INITIAL_WINDOW_WIDTH), currentHeight(INITIAL_WINDOW_HEIGHT)
{
    blocks = (NUM_PARTICLES + threadsPerBlock - 1) / threadsPerBlock;
    d_particles = {nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr, nullptr};
    d_grid = {nullptr, nullptr, nullptr, nullptr};

    updateGridDimensions(currentWidth, currentHeight);
    allocateDeviceMemory();
}

PhysicsSimulator::~PhysicsSimulator()
{
    if (cudaVboResource)
    {
        unregisterVBO();
    }
    freeDeviceMemory();
}

void PhysicsSimulator::updateGridDimensions(int width, int height)
{
    gridWidth = (int)ceilf(width / CELL_SIZE);
    gridHeight = (int)ceilf(height / CELL_SIZE);
    totalCells = gridWidth * gridHeight;

    printf("Grid dimensions: %dx%d = %d cells (window: %dx%d)\n",
           gridWidth, gridHeight, totalCells, width, height);
}

void PhysicsSimulator::allocateDeviceMemory()
{
    size_t size = NUM_PARTICLES * sizeof(float);
    cudaMalloc(&d_particles.x, size);
    cudaMalloc(&d_particles.y, size);
    cudaMalloc(&d_particles.vx, size);
    cudaMalloc(&d_particles.vy, size);
    cudaMalloc(&d_particles.r, size);
    cudaMalloc(&d_particles.g, size);
    cudaMalloc(&d_particles.b, size);
    cudaMalloc(&d_particles.radius, size);
    cudaMalloc(&d_particles.mass, size);

    cudaMalloc(&d_grid.cellStart, totalCells * sizeof(int));
    cudaMalloc(&d_grid.cellEnd, totalCells * sizeof(int));
    cudaMalloc(&d_grid.particleHash, NUM_PARTICLES * sizeof(int));
    cudaMalloc(&d_grid.particleIndex, NUM_PARTICLES * sizeof(int));
}

void PhysicsSimulator::freeDeviceMemory()
{
    cudaFree(d_particles.x);
    cudaFree(d_particles.y);
    cudaFree(d_particles.vx);
    cudaFree(d_particles.vy);
    cudaFree(d_particles.r);
    cudaFree(d_particles.g);
    cudaFree(d_particles.b);
    cudaFree(d_particles.radius);
    cudaFree(d_particles.mass);

    cudaFree(d_grid.cellStart);
    cudaFree(d_grid.cellEnd);
    cudaFree(d_grid.particleHash);
    cudaFree(d_grid.particleIndex);
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

void PhysicsSimulator::updateWorldSize(int width, int height)
{
    if (width == currentWidth && height == currentHeight)
        return;

    printf("Physics world size updated: %dx%d -> %dx%d\n",
           currentWidth, currentHeight, width, height);

    currentWidth = width;
    currentHeight = height;

    // Update grid dimensions and reallocate if needed
    int oldTotalCells = totalCells;
    updateGridDimensions(width, height);

    if (totalCells != oldTotalCells)
    {
        cudaFree(d_grid.cellStart);
        cudaFree(d_grid.cellEnd);
        cudaMalloc(&d_grid.cellStart, totalCells * sizeof(int));
        cudaMalloc(&d_grid.cellEnd, totalCells * sizeof(int));
    }
}

void PhysicsSimulator::update(float dt, int windowWidth, int windowHeight)
{
    if (!cudaVboResource)
        return;

    // Check if world size changed
    if (windowWidth != currentWidth || windowHeight != currentHeight)
    {
        updateWorldSize(windowWidth, windowHeight);
    }

    // Run physics on SoA data
    updatePhysicsKernel<<<blocks, threadsPerBlock>>>(
        d_particles.x, d_particles.y, d_particles.vx, d_particles.vy,
        d_particles.radius, NUM_PARTICLES, dt, currentWidth, currentHeight);
    cudaDeviceSynchronize();

    // Spatial grid collision detection
    // 1. Compute hashes
    computeHashKernel<<<blocks, threadsPerBlock>>>(
        d_grid.particleHash, d_grid.particleIndex,
        d_particles.x, d_particles.y, NUM_PARTICLES, gridWidth, gridHeight);

    // 2. Sort particles by hash using Thrust
    thrust::device_ptr<int> hash_ptr(d_grid.particleHash);
    thrust::device_ptr<int> index_ptr(d_grid.particleIndex);
    thrust::sort_by_key(hash_ptr, hash_ptr + NUM_PARTICLES, index_ptr);

    // 3. Initialize cell boundaries to empty
    cudaMemset(d_grid.cellStart, 0xFF, totalCells * sizeof(int));
    cudaMemset(d_grid.cellEnd, 0, totalCells * sizeof(int));

    // 4. Find cell boundaries
    findCellBoundariesKernel<<<blocks, threadsPerBlock>>>(
        d_grid.cellStart, d_grid.cellEnd, d_grid.particleHash,
        NUM_PARTICLES);

    // 5. Handle collisions using grid
    handleCollisionsGridKernel<<<blocks, threadsPerBlock>>>(
        d_particles.x, d_particles.y, d_particles.vx, d_particles.vy,
        d_particles.radius, d_particles.mass,
        d_grid.cellStart, d_grid.cellEnd, d_grid.particleHash, d_grid.particleIndex,
        NUM_PARTICLES, gridWidth, gridHeight);

    // Map VBO and copy data to it
    cudaGraphicsMapResources(1, &cudaVboResource, 0);
    ParticleVertex *d_vbo;
    size_t num_bytes;
    cudaGraphicsResourceGetMappedPointer((void **)&d_vbo, &num_bytes, cudaVboResource);

    // Copy SoA to interleaved VBO format
    copyToVBOKernel<<<blocks, threadsPerBlock>>>(
        d_vbo, d_particles.x, d_particles.y,
        d_particles.r, d_particles.g, d_particles.b, d_particles.radius,
        NUM_PARTICLES);

    cudaDeviceSynchronize();
    cudaGraphicsUnmapResources(1, &cudaVboResource, 0);
}

void PhysicsSimulator::initializeParticles(int windowWidth, int windowHeight)
{
    currentWidth = windowWidth;
    currentHeight = windowHeight;
    updateGridDimensions(windowWidth, windowHeight);

    // --- Random engine ---
    std::mt19937 rng(static_cast<unsigned int>(time(nullptr)));
    std::uniform_real_distribution<float> velXDist(-100.0f, 100.0f);
    std::uniform_real_distribution<float> velYDist(0.0f, 100.0f);
    std::uniform_real_distribution<float> colorDist(0.0f, 1.0f);
    std::uniform_real_distribution<float> radiusDist(MIN_RADIUS, MAX_RADIUS);

    // --- Host buffers ---
    std::vector<float> h_x(NUM_PARTICLES);
    std::vector<float> h_y(NUM_PARTICLES);
    std::vector<float> h_vx(NUM_PARTICLES);
    std::vector<float> h_vy(NUM_PARTICLES);
    std::vector<float> h_r(NUM_PARTICLES);
    std::vector<float> h_g(NUM_PARTICLES);
    std::vector<float> h_b(NUM_PARTICLES);
    std::vector<float> h_radius(NUM_PARTICLES);
    std::vector<float> h_mass(NUM_PARTICLES);

    // --- Circle geometry ---
    const float centerX = windowWidth * 0.4f;
    const float centerY = windowHeight * 0.4f;

    // Radius chosen to fit entirely in window
    const float circleRadius = 0.45f * std::min(windowWidth, windowHeight);

    // --- Particle placement ---
    std::uniform_real_distribution<float> angleDist(0.0f, 2.0f * static_cast<float>(M_PI));
    std::uniform_real_distribution<float> radiusNormDist(0.0f, 1.0f);

    for (int index = 0; index < NUM_PARTICLES; ++index)
    {
        // Uniform distribution inside circle
        float theta = angleDist(rng);
        float r = circleRadius * std::sqrt(radiusNormDist(rng));

        h_x[index] = centerX + r * std::cos(theta);
        h_y[index] = centerY + r * std::sin(theta);

        h_vx[index] = velXDist(rng);
        h_vy[index] = velYDist(rng);

        h_radius[index] = radiusDist(rng);
        h_mass[index] = h_radius[index] * h_radius[index];

        h_r[index] = colorDist(rng);
        h_g[index] = colorDist(rng);
        h_b[index] = colorDist(rng);
    }

    // --- Upload to device ---
    const size_t size = NUM_PARTICLES * sizeof(float);
    cudaMemcpy(d_particles.x, h_x.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.y, h_y.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.vx, h_vx.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.vy, h_vy.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.r, h_r.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.g, h_g.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.b, h_b.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.radius, h_radius.data(), size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_particles.mass, h_mass.data(), size, cudaMemcpyHostToDevice);
}
