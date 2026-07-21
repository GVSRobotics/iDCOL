// Correctness + speed check for the ray-intersection additions in
// core/shape_core.cpp: shape_eval_local_phi, ray_surface_intersection_local,
// ray_surface_intersection_global.

#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <chrono>
#include <cstdio>
#include <random>
#include <vector>

#include "core/idcol_implicitfamily.hpp"
#include "core/shape_core.hpp"

using Eigen::Matrix4d;
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

// Result must lie on the ray (y0 = t*d, t>0) and on the boundary (|phi| small).
void check_ray_local(const char* name, int shape_id, const VectorXd& params, int n, std::mt19937& rng)
{
    int n_ok = 0;
    double max_abs_phi = 0.0, max_on_ray_err = 0.0;

    for (int i = 0; i < n; ++i) {
        Vector3d d = random_unit_vector(rng);
        Vector3d y0;
        if (!ray_surface_intersection_local(d, shape_id, params, y0)) continue;
        ++n_ok;

        double phi;
        shape_eval_local_phi(y0, shape_id, params, phi);
        max_abs_phi = std::max(max_abs_phi, std::abs(phi));

        const double t = y0.dot(d);
        max_on_ray_err = std::max(max_on_ray_err, (y0 - t * d).norm());
    }
    std::printf("%-8s %5d/%-5d ok, max|phi|=%.3e, max on-ray err=%.3e\n",
                name, n_ok, n, max_abs_phi, max_on_ray_err);
}

// ray_surface_intersection_global must equal R*alpha*y0_local + r exactly.
void check_ray_global(const char* name, int shape_id, const VectorXd& params, int n, std::mt19937& rng)
{
    Matrix4d g = Matrix4d::Identity();
    g.topLeftCorner<3,3>() = Eigen::AngleAxisd(1.1, Vector3d(1, 2, 3).normalized()).toRotationMatrix();
    g.topRightCorner<3,1>() << 1.3, -0.7, 2.1;
    const double alpha = 1.4;
    const Eigen::Matrix3d R = g.topLeftCorner<3,3>();
    const Vector3d r = g.topRightCorner<3,1>();

    int n_ok = 0;
    double max_err = 0.0;

    for (int i = 0; i < n; ++i) {
        Vector3d d_world = random_unit_vector(rng);
        Vector3d d_local = R.transpose() * d_world;

        Vector3d y0_local, x0_global;
        const bool ok_l = ray_surface_intersection_local(d_local, shape_id, params, y0_local);
        const bool ok_g = ray_surface_intersection_global(g, alpha, d_world, shape_id, params, x0_global);
        if (!ok_l || !ok_g) continue;
        ++n_ok;

        const Vector3d expected = R * (alpha * y0_local) + r;
        max_err = std::max(max_err, (x0_global - expected).norm());
    }
    std::printf("%-8s %5d/%-5d ok, max local/global mismatch = %.3e\n", name, n_ok, n, max_err);
}

double bench_us_per_call(int shape_id, const VectorXd& params, int n_calls, std::mt19937& rng)
{
    std::vector<Vector3d> dirs(n_calls);
    for (auto& d : dirs) d = random_unit_vector(rng);

    Vector3d y0;
    for (int i = 0; i < 100 && i < n_calls; ++i)
        ray_surface_intersection_local(dirs[i], shape_id, params, y0);

    auto t0 = std::chrono::steady_clock::now();
    for (auto& d : dirs) ray_surface_intersection_local(d, shape_id, params, y0);
    auto t1 = std::chrono::steady_clock::now();

    return std::chrono::duration<double, std::micro>(t1 - t0).count() / n_calls;
}

} // namespace

int main()
{
    using namespace idcol;
    std::mt19937 rng(99);

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

    ShapeSpec tc     = make_tc(20.0, 1.0, 1.5, 1.5, 1.5);
    ShapeSpec se     = make_se(8, 0.5, 1.0, 1.5);
    ShapeSpec sec    = make_sec(8, 1.0, 2.0);
    ShapeSpec custom = make_custom(Eigen::VectorXd(0));

    struct Entry { const char* name; int shape_id; const Eigen::VectorXd* params; };
    std::vector<Entry> shapes = {
        {"sphere", sphere.shape_id, &sphere.params},
        {"poly",   poly.shape_id,   &poly.params},
        {"cone",   tc.shape_id,     &tc.params},
        {"se",     se.shape_id,     &se.params},
        {"sec",    sec.shape_id,    &sec.params},
        {"custom", custom.shape_id, &custom.params},
    };

    const int n = 1000;

    std::printf("=== ray_surface_intersection_local ===\n");
    for (auto& e : shapes) check_ray_local(e.name, e.shape_id, *e.params, n, rng);

    std::printf("\n=== ray_surface_intersection_global vs local ===\n");
    for (auto& e : shapes) check_ray_global(e.name, e.shape_id, *e.params, n, rng);

    std::printf("\n=== ray_surface_intersection_local speed ===\n");
    for (auto& e : shapes) {
        double us = bench_us_per_call(e.shape_id, *e.params, 200000, rng);
        std::printf("%-8s %8.4f us/call\n", e.name, us);
    }

    return 0;
}
