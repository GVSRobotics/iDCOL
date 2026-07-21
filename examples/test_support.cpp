// Convergence + speed check for shape_support_point_local (core/shape_core.cpp).

#include <Eigen/Dense>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "core/idcol_implicitfamily.hpp"
#include "core/shape_core.hpp"

using Eigen::Vector3d;
using Eigen::VectorXd;

namespace {

Vector3d random_unit_vector(std::mt19937& rng) {
    std::normal_distribution<double> nd(0.0, 1.0);
    Vector3d v;
    do {
        v = Vector3d(nd(rng), nd(rng), nd(rng));
    } while (v.squaredNorm() < 1e-12);
    return v.normalized();
}

void test_shape(const char* name, int shape_id, const VectorXd& params,
                 int n_dirs, int n_bench, std::mt19937& rng)
{
    int n_converged = 0;
    Vector3d y;
    double k;
    for (int i = 0; i < n_dirs; ++i)
        if (shape_support_point_local(random_unit_vector(rng), shape_id, params, y, k)) ++n_converged;

    std::vector<Vector3d> dirs(n_bench);
    for (auto& d : dirs) d = random_unit_vector(rng);

    for (int i = 0; i < 100 && i < n_bench; ++i)
        shape_support_point_local(dirs[i], shape_id, params, y, k);

    auto t0 = std::chrono::steady_clock::now();
    for (auto& d : dirs) shape_support_point_local(d, shape_id, params, y, k);
    auto t1 = std::chrono::steady_clock::now();
    const double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / n_bench;

    std::printf("%-8s %5d/%-5d converged   %8.3f us/call\n", name, n_converged, n_dirs, us);
}

} // namespace

int main()
{
    using namespace idcol;
    std::mt19937 rng(42);

    ShapeSpec sphere = make_sphere(0.75);

    Eigen::MatrixXd A(8, 3);
    A <<  1,  1,  1,
          1, -1, -1,
         -1,  1, -1,
         -1, -1,  1,
         -1, -1, -1,
         -1,  1,  1,
          1, -1,  1,
          1,  1, -1;
    Eigen::VectorXd b(8);
    b << 1.0, 1.0, 1.0, 1.0, 5.0/3.0, 5.0/3.0, 5.0/3.0, 5.0/3.0;
    ShapeSpec poly = make_poly(20.0, A, b);

    ShapeSpec tc  = make_tc(20.0, 1.0, 1.5, 1.5, 1.5);
    ShapeSpec se  = make_se(8, 0.5, 1.0, 1.5);
    ShapeSpec sec = make_sec(8, 1.0, 2.0);

    struct Entry { const char* name; int shape_id; const Eigen::VectorXd* params; };
    std::vector<Entry> shapes = {
        {"sphere", sphere.shape_id, &sphere.params},
        {"poly",   poly.shape_id,   &poly.params},
        {"cone",   tc.shape_id,     &tc.params},
        {"se",     se.shape_id,     &se.params},
        {"sec",    sec.shape_id,    &sec.params},
    };

    for (auto& e : shapes) test_shape(e.name, e.shape_id, *e.params, 1000, 200000, rng);

    return 0;
}
