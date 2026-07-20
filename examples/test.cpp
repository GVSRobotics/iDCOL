#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <chrono>

#include "core/idcol_implicitfamily.hpp"   // shape builders
#include "core/idcol_contactpair.hpp"      // ContactPair

int main()
{
    using namespace idcol;

    // -------------------------------------------------
    // 1) Define all supported shapes
    // -------------------------------------------------

    // ---- Sphere (shape_id = 1)
    ShapeSpec sphere = make_sphere(0.75);

    // ---- Polytope: Ax <= b (shape_id = 2)
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
    b << 1.0, 1.0, 1.0, 1.0,
         5.0/3.0, 5.0/3.0, 5.0/3.0, 5.0/3.0;

    double beta = 20.0;
    ShapeSpec poly = make_poly(beta, A, b);

    // ---- Superellipsoid (shape_id = 3)
    int n = 8;
    ShapeSpec se = make_se(n, 0.5, 1.0, 1.5);

    // ---- Superelliptic cylinder (shape_id = 4)
    ShapeSpec sec = make_sec(n, 1.0, 2.0);   // radius, half-height

    // ---- Truncated cone (shape_id = 5)
    ShapeSpec tc = make_tc(beta, 1.0, 1.5, 1.5, 1.5);

    // -------------------------------------------------
    // 2) Pick any contact pair
    // -------------------------------------------------
    ContactPair pair(poly, se);   // poly–superellipsoid contact

    // -------------------------------------------------
    // 3) Relative pose between the two bodies
    // -------------------------------------------------
    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
    g.topRightCorner<3,1>() << -1.8, -2.7, -0.3;

    // -------------------------------------------------
    // 4) Solve contact (Cold Start Measurement)
    // -------------------------------------------------
    auto start_cold = std::chrono::high_resolution_clock::now();
    
    SolveResult out = pair.solve(g);
    
    auto end_cold = std::chrono::high_resolution_clock::now();
    auto time_cold = std::chrono::duration_cast<std::chrono::microseconds>(end_cold - start_cold).count();

    if (!out.newton.converged) {
        std::cout << "[iDCOL] did not converge\n";
        return 0;
    }

    // -------------------------------------------------
    // 5) Inspect solution
    // -------------------------------------------------
    std::cout << "[iDCOL] converged\n"
              << "  contact point x = " << out.newton.x.transpose() << "\n"
              << "  alpha           = " << out.newton.alpha << "\n"
              << "  lambda1         = " << out.newton.lambda1 << "\n"
              << "  lambda2         = " << out.newton.lambda2 << "\n"
              << "  iterations      = " << out.newton.iters_used << "\n";

    // -------------------------------------------------
    // 6) Warm start Benchmark (Simulating Continuous Motion)
    // -------------------------------------------------
    int num_runs = 10000;
    long long total_time_warm = 0;

    for(int i = 0; i < num_runs; i++) {
        // Add a tiny translation to simulate a moving trajectory 
        // This forces the solver to do actual work instead of instantly returning
        g(0, 3) += 0.0001; 
        g(1, 3) += 0.0001;

        auto start = std::chrono::high_resolution_clock::now();
        
        SolveResult out_warm = pair.solve(g);
        
        auto end = std::chrono::high_resolution_clock::now();
        // Using nanoseconds for precision, then converting to microseconds later
        total_time_warm += std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        
        // Prevent compiler from stripping the loop by checking convergence
        if (!out_warm.newton.converged) {
            std::cout << "Warm solve failed at iteration " << i << "\n";
            break;
        }
    }

    double avg_warm_time_us = (total_time_warm / 1000.0) / num_runs;

    std::cout << "\n--- Performance Benchmarks ---\n";
    std::cout << "Cold start time: " << time_cold << " microseconds\n";
    std::cout << "Avg warm start time (continuous motion): " << avg_warm_time_us << " microseconds\n";

    return 0;
}