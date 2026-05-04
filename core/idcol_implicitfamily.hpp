#pragma once

#include <Eigen/Dense>
#include <cmath>
#include <optional>
#include <stdexcept>
#include <string>
#include "radial_bounds.hpp"

namespace idcol {

// ============================================================
// ShapeSpec (user-facing)
// ============================================================

struct ShapeSpec {
    int shape_id = -1;
    Eigen::VectorXd params;
    ::RadialBounds bounds;   // explicitly global
    std::string name;
};

// ============================================================
// internal helpers (not part of API)
// ============================================================

namespace detail {

inline void require(bool cond, const char* msg) {
    if (!cond) throw std::runtime_error(msg);
}

inline void require_pos(double x, const char* msg) {
    if (!(std::isfinite(x) && x > 0.0))
        throw std::runtime_error(msg);
}

// MATLAB-style A(:) column-major packing
inline void pack_A_colmajor(Eigen::VectorXd& out, int offset,
                            const Eigen::Ref<const Eigen::MatrixXd>& A)
{
    const int m = static_cast<int>(A.rows());
    for (int j = 0; j < 3; ++j)
        for (int i = 0; i < m; ++i)
            out(offset + j*m + i) = A(i,j);
}

inline double default_Lscale(const Eigen::VectorXd& b)
{
    double v = b.cwiseAbs().maxCoeff();
    return (v == 0.0) ? 1.0 : v;
}

} // namespace detail

// ============================================================
// USER API — normal-form constructors
// ============================================================
//
// Engine shape IDs:
// 1 sphere
// 2 polytope (smooth-max)
// 3 truncated cone
// 4 superellipsoid
// 5 superelliptic cylinder
// 6 custom shape (implemented in core/custom_shape.hpp)
//

// -------------------- Sphere --------------------
inline ShapeSpec make_sphere(double R,
                             const ::RadialBoundsOptions& opt = {})
{
    detail::require_pos(R, "Sphere: R must be > 0");

    ShapeSpec s;
    s.shape_id = 1;
    s.name = "sphere";
    s.params.resize(1);
    s.params(0) = R;

    s.bounds = ::compute_radial_bounds_local(s.shape_id, s.params, opt);
    return s;
}

// -------------------- Polytope --------------------
inline ShapeSpec make_poly(double beta,
                           const Eigen::Ref<const Eigen::MatrixXd>& A_in,
                           const Eigen::Ref<const Eigen::VectorXd>& b_in,
                           const ::RadialBoundsOptions& opt = {},
                           std::optional<double> Lscale = std::nullopt)
{
    detail::require_pos(beta, "Poly: beta must be > 0");
    detail::require(A_in.cols() == 3, "Poly: A must be m x 3");
    detail::require(A_in.rows() == b_in.size(), "Poly: size mismatch");
    detail::require(A_in.allFinite(), "Poly: A has NaN/Inf");
    detail::require(b_in.allFinite(), "Poly: b has NaN/Inf");

    const int m = static_cast<int>(A_in.rows());
    detail::require(m >= 1, "Poly: m must be >= 1");

    Eigen::MatrixXd A = A_in;
    Eigen::VectorXd b = b_in;

    // normalize rows (MATLAB-equivalent)
    for (int i = 0; i < m; ++i) {
        double n = A.row(i).norm();
        detail::require(n > 0.0 && std::isfinite(n),
                        "Poly: zero/invalid row normal");
        A.row(i) /= n;
        b(i)     /= n;
    }

    double Ls = (Lscale && *Lscale > 0.0)
                  ? *Lscale
                  : detail::default_Lscale(b);

    ShapeSpec s;
    s.shape_id = 2;
    s.name = "poly";
    s.params.resize(3 + 3*m + m);

    s.params(0) = beta;
    s.params(1) = static_cast<double>(m);
    s.params(2) = Ls;

    detail::pack_A_colmajor(s.params, 3, A);
    s.params.segment(3 + 3*m, m) = b;

    s.bounds = ::compute_radial_bounds_local(s.shape_id, s.params, opt);
    return s;
}

// -------------------- Truncated cone --------------------
inline ShapeSpec make_tc(double beta, double Rb, double Rt,
                         double a, double b,
                         const ::RadialBoundsOptions& opt = {})
{
    detail::require_pos(beta, "TC: beta");
    detail::require_pos(Rb,   "TC: Rb");
    detail::require_pos(Rt,   "TC: Rt");
    detail::require_pos(a,    "TC: a");
    detail::require_pos(b,    "TC: b");

    ShapeSpec s;
    s.shape_id = 3;
    s.name = "tc";
    s.params.resize(5);
    s.params << beta, Rb, Rt, a, b;

    s.bounds = ::compute_radial_bounds_local(s.shape_id, s.params, opt);
    return s;
}

// -------------------- Superellipsoid --------------------
inline ShapeSpec make_se(double n, double a, double b, double c,
                         const ::RadialBoundsOptions& opt = {})
{
    detail::require_pos(n, "SE: n");
    detail::require_pos(a, "SE: a");
    detail::require_pos(b, "SE: b");
    detail::require_pos(c, "SE: c");

    ShapeSpec s;
    s.shape_id = 4;
    s.name = "se";
    s.params.resize(4);
    s.params << n, a, b, c;

    s.bounds = ::compute_radial_bounds_local(s.shape_id, s.params, opt);
    return s;
}

// -------------------- Superelliptic cylinder --------------------
inline ShapeSpec make_sec(double n, double r, double h,
                          const ::RadialBoundsOptions& opt = {})
{
    detail::require_pos(n, "SEC: n");
    detail::require_pos(r, "SEC: r");
    detail::require_pos(h, "SEC: h");

    ShapeSpec s;
    s.shape_id = 5;
    s.name = "sec";
    s.params.resize(3);
    s.params << n, r, h;

    s.bounds = ::compute_radial_bounds_local(s.shape_id, s.params, opt);
    return s;
}

// -------------------- Custom shape --------------------
inline ShapeSpec make_custom(const Eigen::Ref<const Eigen::VectorXd>& params,
                             const ::RadialBoundsOptions& opt = {},
                             const std::string& name = "custom")
{
    ShapeSpec s;
    s.shape_id = 6;
    s.name = name;
    s.params = params;

    s.bounds = ::compute_radial_bounds_local(s.shape_id, s.params, opt);
    return s;
}

} // namespace idcol
