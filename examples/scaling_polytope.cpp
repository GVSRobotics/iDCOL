// examples/scaling_polytope.cpp
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>
#include <numeric>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <stdexcept>
#include <string>
#include <tuple>

#include "core/idcol_implicitfamily.hpp"
#include "core/idcol_contactpair.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------------------------------------------------------------
// Ergodic pose generator
// ---------------------------------------------------------------------
static Eigen::Matrix4d getSystematicPose(double t, double r_min, double r_max) {
    const double f1 = std::sqrt(2.0),  f2 = std::sqrt(3.0),  f3 = std::sqrt(5.0);
    const double f4 = std::sqrt(7.0),  f5 = std::sqrt(11.0), f6 = std::sqrt(13.0);
    const double f7 = std::sqrt(17.0);
    const double TWO_PI = 2.0 * M_PI;

    const double u = 0.5 * (1.0 + std::sin(TWO_PI * f1 * t));
    const double r_val = std::cbrt(r_min*r_min*r_min +
                                   (r_max*r_max*r_max - r_min*r_min*r_min) * u);
    const double theta = (M_PI/2.0) * std::sin(TWO_PI * f2 * t);
    const double phi   = TWO_PI * f3 * t;

    Eigen::Vector3d pos;
    pos << r_val*std::cos(theta)*std::cos(phi),
           r_val*std::cos(theta)*std::sin(phi),
           r_val*std::sin(theta);

    Eigen::Quaterniond q(std::sin(TWO_PI*f4*t), std::cos(TWO_PI*f5*t),
                         std::sin(TWO_PI*f6*t), std::cos(TWO_PI*f7*t));
    q.normalize();

    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
    g.topLeftCorner<3,3>()  = q.toRotationMatrix();
    g.topRightCorner<3,1>() = pos;
    return g;
}

// ---------------------------------------------------------------------
// Halton sequence (1D) and Halton sphere (S^2 directions)
// Deterministic, low-discrepancy, no rotational symmetry.
// ---------------------------------------------------------------------
static double halton(int i, int base) {
    double h = 0.0;
    double f = 1.0;
    while (i > 0) {
        f /= double(base);
        h += f * double(i % base);
        i /= base;
    }
    return h;
}

static Eigen::Vector3d halton_sphere_dir(int i) {
    // i must be >= 1
    const double u = halton(i, 2);
    const double v = halton(i, 3);
    const double z = 2.0 * u - 1.0;
    const double r = std::sqrt(std::max(0.0, 1.0 - z*z));
    const double phi = 2.0 * M_PI * v;
    Eigen::Vector3d n(r * std::cos(phi), r * std::sin(phi), z);
    n.normalize();
    return n;
}

// ---------------------------------------------------------------------
// Chunky polytope generator (matches the MATLAB make_chunky).
//   Base: asymmetric axis-aligned box with half-extents (1.6, 1.2, 1.0).
//   Extras: m-6 Halton-sphere chip facets, each cutting at 85% of the
//           ray-to-box-boundary distance in its normal direction.
//   index_offset shifts the Halton sequence so two paired bodies use
//   different chip sets.
// ---------------------------------------------------------------------
static void make_chunky_polytope(int m, int index_offset,
                                 Eigen::MatrixXd& A, Eigen::VectorXd& b)
{
    if (m < 6) {
        throw std::invalid_argument("m must be >= 6");
    }

    const double hx = 1.6, hy = 1.2, hz = 1.0;

    A.resize(m, 3);
    b.resize(m);

    // Asymmetric box
    A.row(0) <<  1, 0, 0; b(0) = hx;
    A.row(1) << -1, 0, 0; b(1) = hx;
    A.row(2) <<  0, 1, 0; b(2) = hy;
    A.row(3) <<  0,-1, 0; b(3) = hy;
    A.row(4) <<  0, 0, 1; b(4) = hz;
    A.row(5) <<  0, 0,-1; b(5) = hz;

    // Halton-sphere chips
    const double eps = 1e-12;
    for (int i = 6; i < m; ++i) {
        const Eigen::Vector3d n = halton_sphere_dir(i - 5 + index_offset);
        // Distance from origin to box boundary along direction n
        const double t_box = std::min({
            hx / (std::abs(n(0)) + eps),
            hy / (std::abs(n(1)) + eps),
            hz / (std::abs(n(2)) + eps)
        });
        A.row(i) = n.transpose();
        b(i) = 0.85 * t_box;
    }
}

// ---------------------------------------------------------------------
// Benchmark runner for one m
// ---------------------------------------------------------------------
struct ScalingResult {
    int    m            = 0;
    double avg_us_cold  = 0.0;
    double avg_us_warm  = 0.0;
    double med_us_cold  = 0.0;
    double med_us_warm  = 0.0;
    double avg_iters    = 0.0;
    double success_pct  = 0.0;
};

static ScalingResult run_scaling_case(int m, double beta,
                                      double t_max, double dt)
{
    using namespace idcol;
    using Clock = std::chrono::steady_clock;
    using us_t  = std::chrono::duration<double, std::micro>;

    Eigen::MatrixXd A1, A2;
    Eigen::VectorXd b1, b2;
    
    // Body 1 and body 2 are identical (same chip set)
    make_chunky_polytope(m, 0, A1, b1);
    make_chunky_polytope(m, 0, A2, b2);

    RadialBoundsOptions optr;
    optr.num_starts = 1000;

    auto P1 = idcol::make_poly(beta, A1, b1, optr);
    auto P2 = idcol::make_poly(beta, A2, b2, optr);

    NewtonOptions opt;
    opt.L = 1; opt.max_iters = 30; opt.tol = 1e-10; opt.verbose = false;

    SurrogateOptions sopt;
    sopt.fS_values = {1, 3, 5, 7, 9};
    sopt.enable_scaling = false;

    ContactPair pair(P1, P2, opt, sopt);

    const double r_min = 0.05;
    const double r_max = 4.0;   // matches the larger asymmetric body
    const int    N     = static_cast<int>(std::round(t_max / dt)) + 1;

    auto bench = [&](bool warm) {
        std::vector<double> dur;
        std::vector<int>    iters;
        dur.reserve(N); iters.reserve(N);

        for (int i = 0; i < N; ++i) {
            const double t = i * dt;
            const Eigen::Matrix4d g = getSystematicPose(t, r_min, r_max);
            if (!warm) pair.reset_guess();

            const auto t0 = Clock::now();
            SolveResult out = pair.solve(g);
            const auto t1 = Clock::now();

            if (!out.newton.converged) continue;

            const double us = std::chrono::duration_cast<us_t>(t1 - t0).count();
            dur.push_back(us);
            iters.push_back(out.newton.iters_used);
        }

        std::sort(dur.begin(), dur.end());
        const double avg = dur.empty() ? 0.0
            : std::accumulate(dur.begin(), dur.end(), 0.0) / double(dur.size());
        const double med = dur.empty() ? 0.0 : dur[dur.size()/2];
        const double its = iters.empty() ? 0.0
            : std::accumulate(iters.begin(), iters.end(), 0.0) / double(iters.size());
        const double succ_pct = 100.0 * double(dur.size()) / double(N);
        return std::tuple<double,double,double,double>(avg, med, its, succ_pct);
    };

    auto [avg_c, med_c, its_c, succ_c] = bench(false);
    auto [avg_w, med_w, its_w, succ_w] = bench(true);

    ScalingResult R;
    R.m            = m;
    R.avg_us_cold  = avg_c; R.med_us_cold = med_c;
    R.avg_us_warm  = avg_w; R.med_us_warm = med_w;
    R.avg_iters    = its_c;
    R.success_pct  = std::min(succ_c, succ_w);

    std::cout << "m=" << std::setw(4) << m
              << " | cold avg=" << std::setw(8) << std::fixed << std::setprecision(2) << avg_c
              << " med=" << std::setw(8) << med_c
              << " iters=" << std::setprecision(2) << its_c
              << " | warm avg=" << std::setw(8) << avg_w
              << " med=" << std::setw(8) << med_w
              << " iters=" << std::setprecision(2) << its_w
              << " | succ=" << std::setprecision(1) << R.success_pct << "%\n";
    return R;
}

// ---------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------
int main() {
    Eigen::setNbThreads(1);

    const std::vector<int> m_values = {8, 16, 32, 64, 128};
    const double beta  = 20.0;
    const double t_max = 100.0;
    const double dt    = 1e-4;

    std::ofstream csv("scaling_polytope.csv");
    csv << "m,avg_us_cold,med_us_cold,avg_us_warm,med_us_warm,avg_iters,success_pct\n";

    std::cout << "=== iDCOL scaling: chunky polytope vs. m ===\n";
    std::cout << "beta=" << beta << ", t_max=" << t_max << ", dt=" << dt << "\n\n";

    for (int m : m_values) {
        ScalingResult R = run_scaling_case(m, beta, t_max, dt);
        csv << R.m << ","
            << R.avg_us_cold << "," << R.med_us_cold << ","
            << R.avg_us_warm << "," << R.med_us_warm << ","
            << R.avg_iters   << "," << R.success_pct << "\n";
    }

    csv.close();
    std::cout << "\nWrote scaling_polytope.csv\n";
    return 0;
}