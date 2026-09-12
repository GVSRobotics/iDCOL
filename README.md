<p align="center">
  <img src="idcol.png" width="200">
</p>

# iDCOL

iDCOL is a differentiable contact kinematic framework for strictly convex contact geometry, designed for gradient-based simulation, planning, and optimization in contact-rich robotic systems.

**Paper:** [Collision Detection With Analytical Derivatives of Contact Kinematics](https://doi.org/10.1109/LRA.2026.3732913), IEEE Robotics and Automation Letters

**Website:** [gvsrobotics.github.io/iDCOL](https://gvsrobotics.github.io/iDCOL)

**Julia port:** a Julia implementation is available at [iDCOL.jl](https://github.com/GVSRobotics/iDCOL.jl)

It provides:

* robust collision detection for strictly convex implicit shapes,
* analytical derivatives of contact kinematics,
* efficient warm-started contact tracking,
* a lightweight C++ API suitable for physics engines and optimizers.

At its core, iDCOL reduces contact computation to a fixed-size nonlinear solve, making it fast, differentiable, and easy to integrate. 

---
## Quickstart (C++)

The intended usage is deliberately simple:

1. Create shapes
2. Create a contact pair
3. Solve contact at a relative pose

```cpp
#include <Eigen/Dense>
#include <iostream>

#include "core/idcol_implicitfamily.hpp"
#include "core/idcol_contactpair.hpp"

int main() {
    using namespace idcol;

    // --- 1) Create shapes ---
    Eigen::MatrixXd A(8,3);
    A <<  1,  1,  1,
          1, -1, -1,
         -1,  1, -1,
         -1, -1,  1,
         -1, -1, -1,
         -1,  1,  1,
          1, -1,  1,
          1,  1, -1;

    Eigen::VectorXd b(8);
    b << 1,1,1,1, 5.0/3,5.0/3,5.0/3,5.0/3;

    auto poly  = make_poly(20.0, A, b);
    auto ellip = make_se(1.0, 0.5, 1.0, 1.5);

    // --- 2) Create a contact pair ---
    ContactPair pair(poly, ellip);

    // --- 3) Solve contact ---
    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
    g.topRightCorner<3,1>() << 0.2, 0.1, 0.3;

    SolveResult out = pair.solve(g);

    if (!out.newton.converged) {
        std::cout << "Contact solve failed: "
                  << out.newton.message << std::endl;
        return 0;
    }

    std::cout << "Contact solved in "
              << out.newton.iters_used << " iterations\n";
    std::cout << "Contact point x = "
              << out.newton.x.transpose() << "\n";
    std::cout << "alpha = " << out.newton.alpha << "\n";
}
```

That is the entire workflow.

---
## Supported shapes

Shapes are constructed using helper functions in the `idcol` namespace.
The current implementation supports a sphere and four families of smooth convex implicit shapes.  
Additional shape families will be added in future releases.

### Shape constructors

```cpp
idcol::make_sphere(R)
```
Sphere of radius `R`.

```cpp
idcol::make_poly(beta, A, b)
```
Smoothed convex polytope defined by the half-space constraints
A x ≤ b

- `A` is an \(m x 3\) matrix of outward normals  
- `b` is an \(m x 1\) vector of offsets  
- `beta` controls the smooth-max approximation (larger = sharper)

```cpp
idcol::make_tc(beta, rb, rt, a, b)
```
Smooth truncated cone.

- `rb` : base radius  
- `rt` : top radius  
- `a, b` : axial profile parameters  
- `beta` : smoothing parameter

```cpp
idcol::make_se(n, a, b, c)
```
Superellipsoid with semi-axes `(a, b, c)` and exponent `n`.

Setting `n = 1` yields a standard ellipsoid.

```cpp
idcol::make_sec(n, r, h)
```
Superelliptic cylinder.

- `r` : radius  
- `h` : half-height  
- `n` : shape exponent
---
## Requirements

* CMake (>= 3.16)
* A C++17-compatible compiler
* Eigen3, discoverable via CMake's `find_package(Eigen3)` (e.g. installed through [vcpkg](https://vcpkg.io/), your system package manager, or any other install that provides `Eigen3Config.cmake`)
* (Optional) MATLAB with C++17 MEX support

---

## Building the C++ code

### Linux / macOS

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

### Windows (Visual Studio)

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 -DCMAKE_PREFIX_PATH="<path to your Eigen3 install>"
cmake --build build --config Release
```

If CMake can't find Eigen3 automatically, pass its install prefix via `-DCMAKE_PREFIX_PATH` (for a vcpkg install this is typically `<vcpkg root>/installed/x64-windows`).

Executables are generated in:

* build/ on Linux/macOS
* build/Release/ on Windows

---

## Running the C++ example

Windows:

```
build\Release\idcol_ergodic.exe
```

Linux/macOS:

```
./build/idcol_ergodic
```

This example demonstrates warm-started contact tracking over a sequence of configurations.

---

## MATLAB MEX interface (optional)

iDCOL provides MATLAB MEX bindings.

To build:

```
build_mex
```

Compiled MEX files are placed in mex/.

MATLAB usage:

```
out = idcol_solve_mex(S, guess, opt, sopt);
```

The output structure contains the contact solution, residuals, and analytical Jacobians.

---

## Design philosophy

iDCOL is built around three principles:

* Strictly convex implicit primitives
* Scaling based convex optimization
* Fixed-size Newton solver
* Analytical derivatives first

---

## Status

This is an active research codebase accompanying ongoing work on
implicit differentiable collision detection. The implementation is
usable but evolving; APIs and interfaces may change without notice.

The code is provided as-is for research and experimentation. If you use this code in your research, please cite:

```bibtex
@article{mathew2026collisiondetectionanalyticalderivatives,
      title={Collision Detection With Analytical Derivatives of Contact Kinematics},
      author={Anup Teejo Mathew and Anees Peringal and Daniele Caradonna and Frederic Boyer and Federico Renda},
      journal={IEEE Robotics and Automation Letters},
      year={2026},
      doi={10.1109/LRA.2026.3732913},
}

