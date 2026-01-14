# Particle Physics Simulation

A real-time 2D particle physics simulator with elastic collisions, gravity, and spatial grid optimization. Available in both **CPU** and **CUDA GPU** implementations for performance comparison.

## Features

- ✨ **1000+ particles** with varying mass and radius
- 🌍 **Gravity simulation** with realistic falling and bouncing
- ⚡ **Spatial grid optimization** for O(n) collision detection
- 🖼️ **Dynamic window resizing** - physics world automatically adapts
- 🎨 **OpenGL visualization** with point sprites

## Project Structure

```
particle-physics-sim/
├── CPU/                    # CPU-only implementation
│   ├── src/
│   │   ├── main.cpp
│   │   ├── physics.cpp
│   │   └── renderer.cpp
│   ├── include/
│   │   ├── particle.h
│   │   ├── physics.h
│   │   └── renderer.h
│   └── Makefile
│
├── src/                    # CUDA GPU implementation
│   ├── main.cu
│   ├── physics.cu
│   └── renderer.cpp
├── include/
│   ├── particle.h
│   ├── physics.cuh
│   └── renderer.h
└── CMakeLists.txt
```

## Prerequisites

### CPU Version
- C++17 compiler (GCC 7+)
- OpenGL 4.6+, GLFW 3.x, GLEW

### GPU Version (Additional)
- NVIDIA GPU with CUDA support
- CUDA Toolkit 11.0+

## Installation

**Install Dependencies:**
```bash
# Ubuntu/Debian
sudo apt-get install build-essential libglfw3-dev libglew-dev libgl-dev

# Fedora
sudo dnf install gcc-c++ glfw-devel glew-devel mesa-libGL-devel

# Arch Linux
sudo pacman -S base-devel glfw-x11 glew
```

**For GPU version, install [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads)**

## Building & Running

### CPU Version

```bash
cd CPU
make
./particles
```

### GPU Version

```bash
cmake .
cmake --build .
./particles
```

**Note**: Adjust `CUDA_ARCHITECTURES` in `CMakeLists.txt` for your GPU (RTX 3000: `86`, RTX 4000: `89`)

## Configuration

Edit `NUM_PARTICLES` and other constants in `include/particle.h`:

```cpp
const int NUM_PARTICLES = 1000;           // Number of particles
const float GRAVITY = 500.0f;             // Gravity strength
const float DAMPING = 0.98f;              // Energy loss on collision
const float MIN_RADIUS = 3.0f;            // Minimum particle size
const float MAX_RADIUS = 8.0f;            // Maximum particle size
```

## Performance Comparison

### Benchmark Setup
- **CPU**: AMD Ryzen 7
- **GPU**: NVIDIA GeForce RTX 4060
- **Resolution**: 2884x1681
- **Measurement**: Average FPS over 60 seconds

### CPU Implementation Results

| Particle Count | FPS | Notes |
|----------------|-----|-------|
| 500            | ~120 | Rather stable visualization |
| 1,000          | ~120 | First drops, to around 100 |
| 5,000          | 120~100 | Drops are more significant and appear reguraly |
| 50,000          | ~30 | FPS are not consistent, very much depends on clusters of particles |
| 100,000         | ~14 | CPU is struggling |

### GPU Implementation Results (CUDA)

| Particle Count | FPS | Notes |
|----------------|-----|-------|
| 500            | ~120 | Stable |
| 1,000          | ~120 | Stable |
| 5,000          | ~120 | Stable |
| 50,000         | ~120 | Stable, probably the best one |
| 100,000        | 120~90 | Unstability due to lacking physic model, particles start clustering and forces are just escalating, when freely moving models has more fps. |
| 150,000          | 110~60 | Similar as above, but more clear |


The most important aspect which affects the FPS is physic model, as it leads to cluster of particles and more chaotic behaviour.

### Key Optimizations

- Spatial hashing with uniform grid
- 3×3 neighbor cell checking only
- Hash map for O(1) grid lookup
- Structure of Arrays (SoA) for memory coalescing
- Thrust library for fast sorting
- CUDA-OpenGL interop (zero-copy rendering)
- Warp-level parallelism

### Spatial Grid Impact
- **Without grid**: O(n²) collision checks
- **With grid**: O(n) collision checks
- **Typical speedup**: 10-100x for 1000+ particles

## Implementation Details

**Elastic Collision Formula:**
```
impulse = 2 × (v₁ - v₂) · n / (m₁ + m₂)
v₁' = v₁ - impulse × m₂ × n
v₂' = v₂ + impulse × m₁ × n
```

**Spatial Grid:**
- Window divided into uniform cells (size = `MAX_RADIUS × 2.5`)
- Each particle checks only 3×3 neighboring cells
- Reduces collision complexity from O(n²) to O(n)

**GPU Memory Layout (SoA):**
```
Array of Structures:        Structure of Arrays:
[x,y,vx,vy,r,g,b,...]  →   [x,x,x,...][y,y,y,...][vx,vx,vx,...]
Scattered access           Coalesced access (faster)
```

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Acknowledgments

- Spatial hashing: [Optimized Spatial Hashing for Collision Detection](https://matthias-research.github.io/pages/publications/tetraederCollision.pdf)
- CUDA interop: [NVIDIA CUDA Samples](https://github.com/NVIDIA/cuda-samples)

---

**Fill in benchmark tables after testing on your hardware**