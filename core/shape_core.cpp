#include "shape_core.hpp"
#include "custom_shape.hpp"
#include <cmath>
#include <Eigen/Dense>
#include <stdexcept>
#include <limits>
#include <algorithm>

using Eigen::Vector3d;
using Eigen::Vector4d;
using Eigen::VectorXd;
using Eigen::Matrix3d;
using Eigen::Matrix4d;

inline double pow_int_pos(double z, int n) {
    // z > 0, n >= 0
    double out = 1.0;
    for (int k = 0; k < n; ++k) out *= z;
    return out;
}

inline void phi_grad_from_S(
    double S,
    double inv_e,
    double& phi,
    double& q,       // output q = S_safe^(1/e)
    double& qprime   // output qprime = (1/e) * S_safe^(1/e - 1)
) {
    const double S_eps  = 1e-16;
    const double S_safe = S + S_eps;

    q = std::pow(S_safe, inv_e);
    phi = q - 1.0;

    // q'(S) = inv_e * q / S_safe
    qprime = inv_e * q / S_safe;
}


inline double phi_from_S(double S, double inv_e) {
    const double S_eps = 1e-16;
    return std::pow(S + S_eps, inv_e) - 1.0;
}

inline double eps_u_from_n(int n) {
    const double eps_min = 1e-12;   // enough to prevent 0/0, tiny geometry perturbation
    const double eps_max = 2e-3;
    // Geometry budget: max boundary shrink ~ eps/2 in normalized coords.
    // For max 0.1% change at n = 8 => eps_max = 0.002

    if (n <= 1) return eps_min;

    const double t = std::min(1.0, std::max(0.0, (double(n) - 1.0) / 7.0));
    return std::max(eps_min, eps_max * t * t);
}


// phi only, no gradient/Hessian 
void shape_eval_local_phi(
    const Vector3d& y,
    int shape_id,
    const VectorXd& params,
    double& phi)
{
    const double* p = params.data();
    const std::size_t nParams = static_cast<std::size_t>(params.size());

    phi = 0.0;

    auto fail = [&](const char* msg){ throw std::runtime_error(msg); };

    // ------------------------
    // shape_id = 1 : Sphere
    // ------------------------
    if (shape_id == 1)
    {
        if (nParams < 1) fail("Sphere needs params(1) = radius.");
        const double R = p[0];
        phi = y.squaredNorm() / (R * R) - 1.0;
        return;
    }

    // -----------------------------------------
    // shape_id = 2 : Smooth polytope, smooth-max
    // -----------------------------------------
    else if (shape_id == 2)
    {
        if (nParams < 3) fail("Smooth polytope params too short.");

        const double beta   = p[0];
        const int    m      = static_cast<int>(p[1]);
        const double Lscale = p[2];

        if (m <= 0)          fail("Smooth polytope: m must be positive.");
        if (!(beta  > 0.0))  fail("Smooth polytope: beta must be > 0.");
        if (!(Lscale > 0.0)) fail("Smooth polytope: Lscale must be > 0.");

        const std::size_t expected = 3 + 4 * static_cast<std::size_t>(m);
        if (nParams != expected) fail("Smooth polytope params size must be 3 + 4*m.");

        const double* A_data = p + 3;
        const double* b_data = p + 3 + 3*m;

        using Matm3Col = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor>;
        Eigen::Map<const Matm3Col> A(A_data, m, 3);
        Eigen::Map<const Eigen::VectorXd> b(b_data, m);

        const double invL = 1.0 / Lscale;

        double zmax = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i) {
            const double zi_hat = (A.row(i).dot(y) - b(i)) * invL;
            if (zi_hat > zmax) zmax = zi_hat;
        }

        double sum_ez = 0.0;
        for (int i = 0; i < m; ++i) {
            const double zi_hat = (A.row(i).dot(y) - b(i)) * invL;
            sum_ez += std::exp(beta * (zi_hat - zmax));
        }

        phi = zmax + (1.0 / beta) * std::log(sum_ez);
        return;
    }

    // ---------------------------------
    // shape_id = 3 : Smooth truncated cone
    // ---------------------------------
    else if (shape_id == 3)
    {
        if (nParams < 5)
            fail("Smooth truncated cone needs params = [beta; Rb; Rt; a; b].");

        const double beta = p[0];
        const double Rb   = p[1];
        const double Rt   = p[2];
        const double a    = p[3];
        const double b    = p[4];

        if (Rb <= 0.0 || Rt <= 0.0 || a <= 0.0 || b <= 0.0 || beta <= 0.0)
            fail("Smooth truncated cone params must all be > 0.");

        const double x1 = y(0);
        const double x2 = y(1);
        const double x3 = y(2);

        const double r2 = x2 * x2 + x3 * x3;
        const double h  = a + b;

        const double t  = (x1 + a) / h;
        const double Rx = Rb + (Rt - Rb) * t;

        double phi_side;
        if (Rx > 0.0 && r2 > 0.0) phi_side = r2 / (Rx * Rx) - 1.0;
        else                      phi_side = -1.0;

        const double phi_bot = -x1 / a - 1.0;
        const double phi_top =  x1 / b - 1.0;

        const double mphi = std::max(phi_side, std::max(phi_bot, phi_top));
        const double sum_e = std::exp(beta * (phi_side - mphi))
                            + std::exp(beta * (phi_bot  - mphi))
                            + std::exp(beta * (phi_top  - mphi));

        phi = mphi + (1.0 / beta) * std::log(sum_e);
        return;
    }

    // ---------------------------------
    // shape_id = 4 : Superellipsoid
    // ---------------------------------
    else if (shape_id == 4) {
        if (nParams < 4) fail("Superellipsoid needs params = [n; a; b; c].");

        const double n_raw = p[0];
        const double a = p[1], b = p[2], c = p[3];

        const int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) fail("Superellipsoid: n must be a positive integer.");

        const double u1 = y(0) / a, u2 = y(1) / b, u3 = y(2) / c;
        const double eps_u = eps_u_from_n(n);
        const double z1 = u1*u1 + eps_u, z2 = u2*u2 + eps_u, z3 = u3*u3 + eps_u;

        const double S = pow_int_pos(z1, n) + pow_int_pos(z2, n) + pow_int_pos(z3, n);
        const double inv_e = 1.0 / double(2*n);

        phi = phi_from_S(S, inv_e);
        return;
    }

    // ---------------------------------
    // shape_id = 5 : Superelliptic cylinder
    // ---------------------------------
    else if (shape_id == 5) {
        if (nParams < 3) fail("Superelliptic cylinder needs params = [n; R; h].");

        const double n_raw = p[0];
        const double R = p[1], h = p[2];
        if (R <= 0.0 || h <= 0.0) fail("Superelliptic cylinder: R, h must be > 0.");

        const int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) fail("Superelliptic cylinder: n must be a positive integer.");

        const double ua = y(0) / h;
        const double r2 = y(1)*y(1) + y(2)*y(2);
        const double qrad = r2 / (R * R);

        const double eps_u = eps_u_from_n(n);
        const double za = ua*ua + eps_u, zr = qrad + eps_u;

        const double S = pow_int_pos(za, n) + pow_int_pos(zr, n);
        const double inv_e = 1.0 / double(2*n);

        phi = phi_from_S(S, inv_e);
        return;
    }

    // ------------------------
    // shape_id = 6 : Custom user shape
    // ------------------------
    else if (shape_id == 6)
    {
        // custom_shape.hpp doesn't expose a phi-only path; fall back to the
        // full evaluator and discard the derivatives.
        Vector3d grad_dummy;
        Matrix3d hess_dummy;
        custom_shape_eval_local(y, params, phi, grad_dummy, hess_dummy);
        return;
    }

    else
        fail("Unknown shape_id (valid: 1..6).");
}

void shape_eval_local_phi_grad(
    const Vector3d& y,
    int shape_id,
    const VectorXd& params,
    double& phi,
    Vector3d& grad_phi)
{
    const double* p = params.data();
    const std::size_t nParams = static_cast<std::size_t>(params.size());

    phi = 0.0;
    grad_phi.setZero();
    const double eps = 1e-9;

    auto fail = [&](const char* msg){ throw std::runtime_error(msg); };

    // ------------------------
    // shape_id = 1 : Sphere
    // ------------------------
    if (shape_id == 1)
    {
        if (nParams < 1) fail("Sphere needs params(1) = radius.");

        double R = p[0];
        double R2inv = 1 / (R * R);
        double r2 = y.squaredNorm();

        phi = r2 * R2inv - 1;
        grad_phi = 2.0 * y * R2inv;
        return;
    }

    // -----------------------------------------
    // shape_id = 2 : Smooth polytope, smooth-max
    // -----------------------------------------
   else if (shape_id == 2)
    {
        if (nParams < 3) fail("Smooth polytope params too short.");

        const double beta   = p[0];
        const int    m      = static_cast<int>(p[1]);
        const double Lscale = p[2];

        if (m <= 0)          fail("Smooth polytope: m must be positive.");
        if (!(beta  > 0.0))  fail("Smooth polytope: beta must be > 0.");
        if (!(Lscale > 0.0)) fail("Smooth polytope: Lscale must be > 0.");

        const std::size_t expected = 3 + 4 * static_cast<std::size_t>(m);
        if (nParams != expected) fail("Smooth polytope params size must be 3 + 4*m.");

        const double* A_data = p + 3;
        const double* b_data = p + 3 + 3*m;

        // A is stored column-major in params by your packer. Make it explicit.
        using Matm3Col = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor>;
        Eigen::Map<const Matm3Col> A(A_data, m, 3);
        Eigen::Map<const Eigen::VectorXd> b(b_data, m);

        const double invL = 1.0 / Lscale;

        // z_hat = (a_i^T y - b_i)/Lscale
        double zmax = -std::numeric_limits<double>::infinity();
        for (int i = 0; i < m; ++i) {
            const double zi_hat = (A.row(i).dot(y) - b(i)) * invL;
            if (zi_hat > zmax) zmax = zi_hat;
        }

        double sum_ez = 0.0;
        grad_phi.setZero();

        for (int i = 0; i < m; ++i) {
            const double zi_hat = (A.row(i).dot(y) - b(i)) * invL;
            const double wi_unnorm = std::exp(beta * (zi_hat - zmax));
            sum_ez += wi_unnorm;
            // d(zi_hat)/dy = a_i / Lscale
            grad_phi.noalias() += wi_unnorm * (A.row(i).transpose() * invL);
        }

        // finalize
        phi = zmax + (1.0 / beta) * std::log(sum_ez);

        if (sum_ez > 0.0) {
            grad_phi /= sum_ez;
        } else {
            grad_phi.setZero();
        }

        return;
    }

    // ---------------------------------
    // shape_id = 3 : Smooth truncated cone
    // ---------------------------------
   else if (shape_id == 3)
    {
        if (nParams < 5)
            fail("Smooth truncated cone needs params = [beta; Rb; Rt; a; b].");
        
        double beta = p[0];
        double Rb   = p[1];
        double Rt   = p[2];
        double a    = p[3];
        double b    = p[4];

        if (Rb <= 0.0 || Rt <= 0.0 || a <= 0.0 || b <= 0.0 || beta <= 0.0)
            fail("Smooth truncated cone params must all be > 0.");

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        double r2 = x2 * x2 + x3 * x3;
        double h  = a + b;

        // linear interpolation of radius along x1
        double t  = (x1 + a) / h;        // in [0,1] ideally
        double Rx = Rb + (Rt - Rb) * t;
        double Rx2 = Rx * Rx;
        double Rx3 = Rx2 * Rx;

        double phi_side;
        Vector3d grad_side = Vector3d::Zero();

        if (Rx > 0.0 && r2 > 0.0) {
            // side surface implicit: r^2 / Rx^2 - 1 = 0
            phi_side = r2 / Rx2 - 1.0;

            double inv_Rx2 = 1.0 / Rx2;
            grad_side(1) = 2.0 * x2 * inv_Rx2;
            grad_side(2) = 2.0 * x3 * inv_Rx2;

            double dRx_dx1 = (Rt - Rb) / h;
            grad_side(0) = -2.0 * r2 / Rx3 * dRx_dx1;
        } else {
            // safely "inside" or degenerate case: treat side as inactive
            phi_side = -1.0;
        }

        double phi_bot = -x1 / a - 1;  // bottom plane
        double phi_top =  x1 / b - 1;  // top plane

        Vector3d grad_bot(-1.0 / a, 0.0, 0.0);
        Vector3d grad_top( 1.0 / b, 0.0, 0.0);

        // smooth-max of [phi_side, phi_bot, phi_top]
        double mphi = std::max(phi_side, std::max(phi_bot, phi_top));

        double e_side = std::exp(beta * (phi_side - mphi));
        double e_bot  = std::exp(beta * (phi_bot  - mphi));
        double e_top  = std::exp(beta * (phi_top  - mphi));

        double sum_e = e_side + e_bot + e_top;

        phi = mphi + (1.0 / beta) * std::log(sum_e);

        double inv_sum_e = 1.0 / sum_e;
        double w_side = e_side * inv_sum_e;
        double w_bot  = e_bot  * inv_sum_e;
        double w_top  = e_top  * inv_sum_e;

        grad_phi = w_side * grad_side + w_bot * grad_bot + w_top * grad_top;
        return;
    }

    // ---------------------------------
    // shape_id = 4 : Superellipsoid
    // ---------------------------------
    else if (shape_id == 4) {
        if (nParams < 4) fail("Superellipsoid needs params = [n; a; b; c].");

        const double n_raw = p[0];
        const double a = p[1], b = p[2], c = p[3];

        const int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) fail("Superellipsoid: n must be a positive integer.");

        const double inv_a = 1.0 / a, inv_b = 1.0 / b, inv_c = 1.0 / c;

        const double u1 = y(0) * inv_a;
        const double u2 = y(1) * inv_b;
        const double u3 = y(2) * inv_c;

        const double eps_u = eps_u_from_n(n);

        const double z1 = u1*u1 + eps_u;
        const double z2 = u2*u2 + eps_u;
        const double z3 = u3*u3 + eps_u;

        const double z1_n = pow_int_pos(z1, n);
        const double z2_n = pow_int_pos(z2, n);
        const double z3_n = pow_int_pos(z3, n);

        const double S = z1_n + z2_n + z3_n;

        const double inv_e = 1.0 / double(2*n);

        double q, qprime;
        phi_grad_from_S(S, inv_e, phi, q, qprime);

        // z^(n-1) = z^n / z (safe since z>0)
        const double z1_n1 = z1_n / z1;
        const double z2_n1 = z2_n / z2;
        const double z3_n1 = z3_n / z3;

        const double two_n = 2.0 * double(n);

        const double dSdx1 = two_n * u1 * z1_n1 * inv_a;
        const double dSdx2 = two_n * u2 * z2_n1 * inv_b;
        const double dSdx3 = two_n * u3 * z3_n1 * inv_c;

        grad_phi(0) = qprime * dSdx1;
        grad_phi(1) = qprime * dSdx2;
        grad_phi(2) = qprime * dSdx3;

        return;
    }


    // ---------------------------------
    // shape_id = 5 : Superelliptic cylinder
    // ---------------------------------
    else if (shape_id == 5) {
        if (nParams < 3) fail("Superelliptic cylinder needs params = [n; R; h].");

        const double n_raw = p[0];
        const double R = p[1], h = p[2];
        if (R <= 0.0 || h <= 0.0) fail("Superelliptic cylinder: R, h must be > 0.");

        const int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) fail("Superelliptic cylinder: n must be a positive integer.");

        const double inv_R  = 1.0 / R;
        const double inv_R2 = inv_R * inv_R;
        const double inv_h  = 1.0 / h;

        const double ua = y(0) * inv_h;

        const double x2 = y(1);
        const double x3 = y(2);
        const double r2 = x2*x2 + x3*x3;
        const double qrad = r2 * inv_R2;

        const double eps_u = eps_u_from_n(n);

        const double za = ua*ua + eps_u;
        const double zr = qrad + eps_u;

        const double Sa = pow_int_pos(za, n);
        const double Sr = pow_int_pos(zr, n);

        const double S = Sa + Sr;

        const double inv_e = 1.0 / double(2*n);

        double qS, qprime;
        phi_grad_from_S(S, inv_e, phi, qS, qprime);

        const double two_n = 2.0 * double(n);

        // axial: dSa/dx1
        const double za_n1 = Sa / za;                // za^(n-1)
        const double dSdx1 = two_n * ua * za_n1 * inv_h;

        // radial: Sr = (qrad+eps)^n, dqrad/dx2 = 2x2/R^2, dqrad/dx3 = 2x3/R^2
        const double zr_n1 = Sr / zr;                // zr^(n-1)
        const double dqdx2 = 2.0 * x2 * inv_R2;
        const double dqdx3 = 2.0 * x3 * inv_R2;

        const double dSdx2 = double(n) * zr_n1 * dqdx2;
        const double dSdx3 = double(n) * zr_n1 * dqdx3;

        grad_phi(0) = qprime * dSdx1;
        grad_phi(1) = qprime * dSdx2;
        grad_phi(2) = qprime * dSdx3;

        return;
    }

    // ------------------------
    // shape_id = 6 : Custom user shape
    // ------------------------
    else if (shape_id == 6)
    {
        Matrix3d hess_dummy;
        custom_shape_eval_local(y, params, phi, grad_phi, hess_dummy);
        return;
    }

    else
        fail("Unknown shape_id (valid: 1..6).");
}

void shape_eval_global_xa_phi_grad(
    const Matrix4d& g,
    const Vector3d& x,
    double alpha,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Vector4d& grad)
{
    if (alpha <= 0.0) throw std::runtime_error("shape_eval_global_ax_phi_grad: alpha must be > 0.");

    Matrix3d R = g.block<3,3>(0,0);
    Vector3d r = g.block<3,1>(0,3);

    // y = R' * (x - r) / alpha
    Vector3d y = R.transpose() * (x - r) / alpha;

    // local shape evaluation
    double phi_local = 0.0;
    Vector3d grad_y  = Vector3d::Zero();
    shape_eval_local_phi_grad(y, shape_id, params, phi_local, grad_y);

    phi = phi_local;

    // ---------------------------------------------------
    // gradient wrt x
    // dphi/dx = (1/alpha) * R * grad_y
    // ---------------------------------------------------
    Vector3d grad_x = (1.0/alpha) * (R * grad_y);

    // ---------------------------------------------------
    // gradient wrt alpha
    // dphi/dalpha = - (1/alpha) * y' * grad_y
    // ---------------------------------------------------
    double grad_alpha = -(1.0/alpha) * y.dot(grad_y);

    // pack
    grad.segment<3>(0) = grad_x;
    grad(3) = grad_alpha;
}

void shape_eval_local(
    const Vector3d& y,
    int shape_id,
    const VectorXd& params,
    double& phi,
    Vector3d& grad_phi,
    Matrix3d& hess_phi)
{
    const double* p = params.data();
    std::size_t nParams = params.size();

    phi = 0.0;
    grad_phi.setZero();
    hess_phi.setZero();
    const double eps = 1e-9;

    auto fail = [&](const char* msg){ throw std::runtime_error(msg); };

    // ------------------------
    // shape_id = 1 : Sphere
    // ------------------------
    if (shape_id == 1) {
        if (nParams < 1) {
            fail("Sphere needs params(1) = radius.");
        }
        double R = p[0];
        double R2inv = 1 / (R * R);
        double r2 = y.squaredNorm();
        phi = r2 * R2inv - 1;        // phi(y) = ||y||^2 / R^2 - 1
        grad_phi = 2.0 * y * R2inv;      // ∂phi/∂y = 2y / R^2
        hess_phi = 2.0 * Matrix3d::Identity() * R2inv; // ∂²phi/∂y² = 2I₃ / R^2

        return;
    }

    // -----------------------------------------
    // shape_id = 2 : Smooth polytope, smooth-max
    // -----------------------------------------
    else if (shape_id == 2) {
        if (nParams < 3) fail("Smooth polytope params too short.");

        const double beta   = p[0];
        const int    m      = static_cast<int>(p[1]);
        const double Lscale = p[2];

        if (m <= 0)          fail("Smooth polytope: m must be positive.");
        if (!(beta  > 0.0))  fail("Smooth polytope: beta must be > 0.");
        if (!(Lscale > 0.0)) fail("Smooth polytope: Lscale must be > 0.");

        const std::size_t expected = 3 + 4 * static_cast<std::size_t>(m);
        if (nParams != expected) fail("Smooth polytope params size must be 3 + 4*m.");

        const double* A_data = p + 3;
        const double* b_data = p + 3 + 3*m;

        Eigen::Map<const Eigen::Matrix<double, Eigen::Dynamic, 3>> A(A_data, m, 3);
        Eigen::Map<const Eigen::VectorXd> b(b_data, m);

        const double invL = 1.0 / Lscale;

        // -----------------------------
        // 1) Find zmax_hat = max_i ((a_i' y - b_i)/Lscale)
        // -----------------------------
        double zmax_hat = (A.row(0).dot(y) - b(0)) * invL;
        for (int i = 1; i < m; ++i) {
            const double zi_hat = (A.row(i).dot(y) - b(i)) * invL;
            if (zi_hat > zmax_hat) zmax_hat = zi_hat;
        }

        // -----------------------------
        // 2) Accumulate sum_u, grad, and M in normalized units
        //    u_i = exp(beta * (z_i_hat - zmax_hat))
        //    grad = sum u_i * (a_i/Lscale)
        //    M    = sum u_i * (a_i/Lscale)(a_i/Lscale)^T
        // -----------------------------
        double sum_u = 0.0;
        grad_phi.setZero();
        Eigen::Matrix3d M = Eigen::Matrix3d::Zero();

        for (int i = 0; i < m; ++i) {
            const double zi_hat = (A.row(i).dot(y) - b(i)) * invL;
            const double ui = std::exp(beta * (zi_hat - zmax_hat));

            // a_hat = a_i / Lscale
            const Eigen::Vector3d a_hat = A.row(i).transpose() * invL;

            sum_u += ui;
            grad_phi.noalias() += ui * a_hat;
            M.noalias()        += ui * (a_hat * a_hat.transpose());
        }

        if (sum_u <= 0.0) {
            phi = zmax_hat;
            grad_phi.setZero();
            hess_phi.setZero();
            return;
        }

        // -----------------------------
        // 3) Finalize phi, grad, hess (in normalized units)
        // -----------------------------
        phi = zmax_hat + (1.0 / beta) * std::log(sum_u);

        grad_phi /= sum_u;

        // Hessian of smooth-max: beta * (E[a a^T] - E[a]E[a]^T)
        hess_phi = (beta / sum_u) * M - beta * (grad_phi * grad_phi.transpose());

        return;
    }

    // ---------------------------------
    // shape_id = 3 : Smooth truncated cone
    // ---------------------------------
   else if (shape_id == 3) {
        if (nParams < 5) {
            fail("Smooth truncated cone needs params = [beta; Rb; Rt; a; b].");
        }
        
        double beta = p[0];
        double Rb   = p[1];
        double Rt   = p[2];
        double a    = p[3];
        double b    = p[4];

        if (Rb <= 0.0 || Rt <= 0.0 || a <= 0.0 || b <= 0.0 || beta <= 0.0) {
            fail("Smooth truncated cone: Rb, Rt, a, b, beta all must be > 0.");
        }

        double x1 = y(0);
        double x2 = y(1);
        double x3 = y(2);

        double r2 = x2 * x2 + x3 * x3;
        double h  = a + b;

        double t   = (x1 + a) / h;
        double Rx  = Rb + (Rt - Rb) * t;
        double Rx2 = Rx * Rx;
        double Rx3 = Rx2 * Rx;

        double phi_side = 0.0;
        Vector3d grad_side = Vector3d::Zero();
        Matrix3d hess_side = Matrix3d::Zero();

        if (Rx > 0.0 && r2 > 0.0) {
            double term_side = r2 / Rx2;   // r^2 / Rx^2
            phi_side = term_side - 1.0;

            double inv_Rx2 = 1.0 / Rx2;

            // grad wrt x2, x3
            grad_side(1) = 2.0 * x2 * inv_Rx2;
            grad_side(2) = 2.0 * x3 * inv_Rx2;

            // Rx depends linearly on x1
            double dRx_dx1 = (Rt - Rb) / h;

            // grad wrt x1
            grad_side(0) = -2.0 * r2 / Rx3 * dRx_dx1;

            // Hessian
            hess_side.setZero();
            double dRx_dx1_sq = dRx_dx1 * dRx_dx1;
            double inv_Rx4 = 1.0 / (Rx2 * Rx2);

            hess_side(0,0) = 6.0 * r2 * inv_Rx4 * dRx_dx1_sq;
            hess_side(1,1) = 2.0 * inv_Rx2;
            hess_side(2,2) = 2.0 * inv_Rx2;

            double H12 = -4.0 * x2 * dRx_dx1 / Rx3;
            double H13 = -4.0 * x3 * dRx_dx1 / Rx3;

            hess_side(0,1) = H12;
            hess_side(1,0) = H12;
            hess_side(0,2) = H13;
            hess_side(2,0) = H13;
        } else {
            phi_side = -1.0;
            grad_side.setZero();
            hess_side.setZero();
        }

        double phi_bot = -x1 / a - 1;
        double phi_top =  x1 / b - 1;

        Vector3d grad_bot(-1.0 / a, 0.0, 0.0);
        Vector3d grad_top( 1.0 / b, 0.0, 0.0);

        // smooth-max over [phi_side, phi_bot, phi_top]
        double mphi = std::max(phi_side, std::max(phi_top, phi_bot));

        double e_side = std::exp(beta * (phi_side - mphi));
        double e_bot  = std::exp(beta * (phi_bot  - mphi));
        double e_top  = std::exp(beta * (phi_top  - mphi));

        double sum_e = e_side + e_bot + e_top;

        phi = mphi + (1.0 / beta) * std::log(sum_e);

        double inv_sum_e = 1.0 / sum_e;
        double w_side = e_side * inv_sum_e;
        double w_bot  = e_bot  * inv_sum_e;
        double w_top  = e_top  * inv_sum_e;

        // gradient of smooth-max
        grad_phi = w_side * grad_side + w_bot * grad_bot + w_top * grad_top;

        // Hessian of smooth-max:
        // H = sum w_i H_i + beta * ( sum w_i g_i g_i^T - (sum w_i g_i)(sum w_i g_i)^T )
        Matrix3d H_sum = Matrix3d::Zero();
        H_sum += w_side * hess_side;  // only side has nonzero Hessian

        Matrix3d G2 = Matrix3d::Zero();
        G2 += w_side * (grad_side * grad_side.transpose());
        G2 += w_bot  * (grad_bot  * grad_bot.transpose());
        G2 += w_top  * (grad_top  * grad_top.transpose());

        hess_phi = H_sum + beta * (G2 - grad_phi * grad_phi.transpose());

        return;
    }

    // ---------------------------------
    // shape_id = 4 : Superellipsoid
    // ---------------------------------
    else if (shape_id == 4) {
        if (nParams < 4) fail("Superellipsoid needs params = [n; a; b; c].");

        const double n_raw = p[0];
        const double a = p[1], b = p[2], c = p[3];

        const int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) {
            fail("Superellipsoid: n must be a positive integer.");
        }

        const double inv_a = 1.0 / a;
        const double inv_b = 1.0 / b;
        const double inv_c = 1.0 / c;

        const double u1 = y(0) * inv_a;
        const double u2 = y(1) * inv_b;
        const double u3 = y(2) * inv_c;

        const double eps_u = eps_u_from_n(n);

        // z_i = u_i^2 + eps_u  (strictly positive)
        const double z1 = u1*u1 + eps_u;
        const double z2 = u2*u2 + eps_u;
        const double z3 = u3*u3 + eps_u;

        // t_i = z_i^n
        const double t1 = pow_int_pos(z1, n);
        const double t2 = pow_int_pos(z2, n);
        const double t3 = pow_int_pos(z3, n);

        const double S = t1 + t2 + t3;

        // phi = S^(1/e) - 1
        const double e = 2.0 * double(n);
        const double inv_e = 1.0 / e;

        const double S_eps  = 1e-16;
        const double S_safe = S + S_eps;

        const double q = std::pow(S_safe, inv_e);
        phi = q - 1.0;

        // q'(S), q''(S) without extra pow
        const double qprime  = inv_e * q / S_safe;
        const double qsecond = (inv_e - 1.0) * qprime / S_safe;

        // z^(n-1), z^(n-2) derived from z^n
        const double z1_n1 = t1 / z1;
        const double z2_n1 = t2 / z2;
        const double z3_n1 = t3 / z3;

        const double z1_n2 = z1_n1 / z1;   // = t1 / (z1*z1)
        const double z2_n2 = z2_n1 / z2;
        const double z3_n2 = z3_n1 / z3;

        const double two_n = 2.0 * double(n);
        const double two_n_minus_1 = 2.0 * double(n) - 1.0;

        // dS/dx_i
        const double dSdx1 = two_n * u1 * z1_n1 * inv_a;
        const double dSdx2 = two_n * u2 * z2_n1 * inv_b;
        const double dSdx3 = two_n * u3 * z3_n1 * inv_c;

        grad_phi(0) = qprime * dSdx1;
        grad_phi(1) = qprime * dSdx2;
        grad_phi(2) = qprime * dSdx3;

        // diagonal Hessian of S: d2S/dx_i^2
        // d2S/du^2 = 2n z^(n-2) [ eps + (2n-1) u^2 ]
        // then multiply by (du/dx)^2
        const double d2Sdx1 = two_n * z1_n2 * (eps_u + two_n_minus_1 * u1*u1) * (inv_a*inv_a);
        const double d2Sdx2 = two_n * z2_n2 * (eps_u + two_n_minus_1 * u2*u2) * (inv_b*inv_b);
        const double d2Sdx3 = two_n * z3_n2 * (eps_u + two_n_minus_1 * u3*u3) * (inv_c*inv_c);

        hess_phi.setZero();

        // q'(S)*Hess(S)
        hess_phi(0,0) += qprime * d2Sdx1;
        hess_phi(1,1) += qprime * d2Sdx2;
        hess_phi(2,2) += qprime * d2Sdx3;

        // q''(S)*gradS*gradS^T
        const Eigen::Vector3d gS(dSdx1, dSdx2, dSdx3);
        hess_phi.noalias() += qsecond * (gS * gS.transpose());

        return;
    }

    // ---------------------------------
    // shape_id = 5 : Superelliptic cylinder
    // ---------------------------------
   else if (shape_id == 5) {
        if (nParams < 3) fail("Superelliptic cylinder needs params = [n; R; h].");

        const double n_raw = p[0];
        const double R = p[1], h = p[2];
        if (R <= 0.0 || h <= 0.0) fail("Superelliptic cylinder: R, h must be > 0.");

        const int n = static_cast<int>(std::round(n_raw));
        if (n <= 0 || std::fabs(n_raw - n) > 1e-9) fail("Superelliptic cylinder: n must be a positive integer.");

        const double inv_R  = 1.0 / R;
        const double inv_R2 = inv_R * inv_R;
        const double inv_h  = 1.0 / h;

        const double x1 = y(0);
        const double x2 = y(1);
        const double x3 = y(2);

        const double ua = x1 * inv_h;

        const double r2 = x2*x2 + x3*x3;
        const double qrad = r2 * inv_R2;     // (r/R)^2

        const double eps_u = eps_u_from_n(n);

        // za = ua^2 + eps, zr = qrad + eps   (both > 0)
        const double za = ua*ua + eps_u;
        const double zr = qrad + eps_u;

        // Sa = za^n, Sr = zr^n
        const double Sa = pow_int_pos(za, n);
        const double Sr = pow_int_pos(zr, n);

        const double S = Sa + Sr;

        const double e = 2.0 * double(n);
        const double inv_e = 1.0 / e;

        const double S_eps  = 1e-16;
        const double S_safe = S + S_eps;

        const double qS = std::pow(S_safe, inv_e);
        phi = qS - 1.0;

        const double qprime  = inv_e * qS / S_safe;
        const double qsecond = (inv_e - 1.0) * qprime / S_safe;

        // derive (n-1),(n-2) powers by division
        const double za_n1 = Sa / za;
        const double za_n2 = za_n1 / za;

        const double zr_n1 = Sr / zr;
        const double zr_n2 = zr_n1 / zr;

        const double two_n = 2.0 * double(n);
        const double two_n_minus_1 = 2.0 * double(n) - 1.0;

        // ---- Gradient of S ----
        // axial: dSa/dx1 = 2n * ua * za^(n-1) * (1/h)
        const double dSdx1 = two_n * ua * za_n1 * inv_h;

        // radial: Sr = (qrad+eps)^n, dqrad/dx2 = 2x2/R^2, dqrad/dx3 = 2x3/R^2
        const double dqdx2 = 2.0 * x2 * inv_R2;
        const double dqdx3 = 2.0 * x3 * inv_R2;

        const double dSdx2 = double(n) * zr_n1 * dqdx2;
        const double dSdx3 = double(n) * zr_n1 * dqdx3;

        grad_phi(0) = qprime * dSdx1;
        grad_phi(1) = qprime * dSdx2;
        grad_phi(2) = qprime * dSdx3;

        // ---- Hessian of S ----
        // axial second derivative:
        // d2Sa/dx1^2 = 2n * za^(n-2) [ eps + (2n-1) ua^2 ] * (1/h^2)
        const double d2Sdx1 = two_n * za_n2 * (eps_u + two_n_minus_1 * ua*ua) * (inv_h * inv_h);

        // radial second derivatives:
        // Sr = (qrad+eps)^n
        // d2Sr/dxi^2 = n(n-1) zr^(n-2) (dq/dxi)^2 + n zr^(n-1) d2q/dxi^2
        // d2q/dx2^2 = d2q/dx3^2 = 2/R^2, d2q/dx2dx3 = 0
        const double d2q = 2.0 * inv_R2;

        const double nn1 = double(n) - 1.0;

        const double d2Sdx2 = double(n) * ( nn1 * zr_n2 * dqdx2 * dqdx2 + zr_n1 * d2q );
        const double d2Sdx3 = double(n) * ( nn1 * zr_n2 * dqdx3 * dqdx3 + zr_n1 * d2q );
        const double d2Sdx2dx3 = double(n) * nn1 * zr_n2 * dqdx2 * dqdx3;

        // ---- Hessian of phi ----
        hess_phi.setZero();

        hess_phi(0,0) += qprime * d2Sdx1;
        hess_phi(1,1) += qprime * d2Sdx2;
        hess_phi(2,2) += qprime * d2Sdx3;
        hess_phi(1,2) += qprime * d2Sdx2dx3;
        hess_phi(2,1) += qprime * d2Sdx2dx3;

        const Eigen::Vector3d gS(dSdx1, dSdx2, dSdx3);
        hess_phi.noalias() += qsecond * (gS * gS.transpose());

        return;
    }

    // ------------------------
    // shape_id = 6 : Custom user shape
    // ------------------------
    else if (shape_id == 6) {
        custom_shape_eval_local(y, params, phi, grad_phi, hess_phi);
        return;
    }

    else {
        fail("Unknown shape_id. Implemented: 1..6.");
    }
}

void shape_eval_global_xa(
    const Matrix4d& g,
    const Vector3d& x,
    double alpha,
    int shape_id,
    const VectorXd& params,
    double& phi,
    Vector4d& grad,
    Matrix4d& H)
{
    if (alpha <= 0.0) throw std::runtime_error("shape_eval_global_ax: alpha must be > 0.");

    Matrix3d R = g.block<3,3>(0,0);
    Vector3d r = g.block<3,1>(0,3);

    // y = R' * (x - r) / alpha
    Vector3d y = R.transpose() * (x - r) / alpha;

    // local eval
    double phi_local;
    Vector3d grad_y;
    Matrix3d H_y;
    shape_eval_local(y, shape_id, params, phi_local, grad_y, H_y);

    phi = phi_local;

    // chain rule
    // grad_x = (1/alpha) * R * grad_y
    Vector3d grad_x = (1.0/alpha) * (R * grad_y);

    // grad_alpha = -(1/alpha) * y' * grad_y
    double grad_alpha = -(1.0/alpha) * y.dot(grad_y);

    // H_xx = (1/alpha^2) * R * H_y * R'
    Matrix3d H_xx = (1.0/(alpha*alpha)) * (R * H_y * R.transpose());

    // H_xa = -(1/alpha^2) * R * (H_y * y + grad_y)
    Vector3d H_xa = -(1.0/(alpha*alpha)) * (R * (H_y * y + grad_y));

    // H_aa = (1/alpha^2) * ( y' H_y y + 2 y' grad_y )
    double H_aa = (1.0/(alpha*alpha)) *
                  ( y.transpose() * H_y * y + 2.0 * y.dot(grad_y) );

    // pack gradient [dphi/dx; dphi/dalpha]
    grad.segment<3>(0) = grad_x;
    grad(3) = grad_alpha;

    // pack Hessian [ H_xx  H_xa
    //                H_xa' H_aa ]
    H.setZero();
    H.block<3,3>(0,0) = H_xx;
    H.block<3,1>(0,3) = H_xa;
    H.block<1,3>(3,0) = H_xa.transpose();
    H(3,3) = H_aa;
}

// Hand-written partial-pivoting Gaussian elimination for the 4x4 support-point system. Returns false if the matrix is singular.
static inline bool solve_lu4(Matrix4d A, const Eigen::Vector4d& b, Eigen::Vector4d& x) {
    x = b;
    for (int k = 0; k < 3; ++k) {
        int piv = k;
        double amax = std::abs(A(k, k));
        for (int i = k + 1; i < 4; ++i) {
            const double a = std::abs(A(i, k));
            if (a > amax) { amax = a; piv = i; }
        }
        if (piv != k) {
            A.row(k).swap(A.row(piv));
            std::swap(x(k), x(piv));
        }
        const double pivval = A(k, k);
        if (pivval == 0.0) return false;
        const double invp = 1.0 / pivval;
        for (int i = k + 1; i < 4; ++i) {
            const double f = A(i, k) * invp;
            if (f != 0.0) {
                for (int j = k; j < 4; ++j) A(i, j) -= f * A(k, j);
                x(i) -= f * x(k);
            }
        }
    }
    if (A(3, 3) == 0.0) return false;
    for (int k = 3; k >= 0; --k) {
        double s = x(k);
        for (int j = k + 1; j < 4; ++j) s -= A(k, j) * x(j);
        x(k) = s / A(k, k);
    }
    return true;
}

// Ray-march from the origin along d_unit (unit vector) to the point y0 =
// t*d_unit with phi(y0) ~= 0 (bracket + safeguarded Newton). Requires
// phi(0) < 0 (origin inside), which is checked here since this is a public
// entry point in its own right, not just shape_support_point_local's seed.
bool ray_surface_intersection_local(
    const Vector3d& d_unit,
    int shape_id,
    const VectorXd& params,
    Vector3d& y0_out)
{
    {
        double phi0;
        shape_eval_local_phi(Vector3d::Zero(), shape_id, params, phi0);
        if (!(phi0 < 0.0))
            throw std::runtime_error("ray_surface_intersection_local: requires phi(0) < 0 (origin inside).");
    }

    const double t0 = 1e-3;
    const double grow = 2.0;
    const int max_grow_steps = 80;
    const int max_refine_iters = 80;
    const double phi_tol = 1e-10;

    double t_hi = t0;
    double phi_hi = 0.0;

    bool bracketed = false;
    for (int i = 0; i < max_grow_steps; ++i) {
        shape_eval_local_phi(t_hi * d_unit, shape_id, params, phi_hi);
        if (phi_hi >= 0.0) { bracketed = true; break; }
        t_hi *= grow;
    }
    if (!bracketed) return false;

    // Newton safeguarded by bisection (rtsafe): the 1D restriction
    // f(t) = phi(t*d_unit), f'(t) = grad_phi(t*d_unit).dot(d_unit) are both
    // already produced by shape_eval_local_phi_grad. Falls back to a bisection step whenever
    // Newton would leave the current bracket or the derivative is
    // degenerate, so it's never worse than pure bisection and typically
    // converges in far fewer iterations.
    //
    // Use Newton with bisection safeguard: tested faster and simpler than
    // pure bisection or Halley in our benchmarks.

    double a = 0.0, b = t_hi;
    double t = 0.5 * (a + b);

    for (int it = 0; it < max_refine_iters; ++it) {
        double phi_t;
        Vector3d grad_t;
        shape_eval_local_phi_grad(t * d_unit, shape_id, params, phi_t, grad_t);

        if (std::abs(phi_t) <= phi_tol) { a = b = t; break; }
        if (phi_t <= 0.0) a = t; else b = t;

        const double fp = grad_t.dot(d_unit);
        const double t_newton = t - phi_t / fp;

        t = (fp != 0.0 && std::isfinite(t_newton) && t_newton > a && t_newton < b)
                ? t_newton : 0.5 * (a + b);
    }

    y0_out = 0.5 * (a + b) * d_unit;
    return true;
}

// World-frame counterpart of ray_surface_intersection_local: marches from
// the shape's placed origin (g's translation) along d_world to the
// world-space point x0 on the boundary.
bool ray_surface_intersection_global(
    const Matrix4d& g,
    double alpha,
    const Vector3d& d_world,
    int shape_id,
    const VectorXd& params,
    Vector3d& x0_out)
{
    if (alpha <= 0.0) throw std::runtime_error("ray_surface_intersection_global: alpha must be > 0.");

    const double dnorm = d_world.norm();
    if (!(dnorm > 0.0)) throw std::runtime_error("ray_surface_intersection_global: d_world must be nonzero.");

    Matrix3d R = g.block<3,3>(0,0);
    Vector3d r = g.block<3,1>(0,3);
    Vector3d d_unit_local = (R.transpose() * d_world) / dnorm;

    Vector3d y0;
    if (!ray_surface_intersection_local(d_unit_local, shape_id, params, y0)) return false;

    x0_out = R * (alpha * y0) + r;
    return true;
}

// Vertex seed for the smooth polytope (shape_id 2, Ay<=b, rows of A unit-
// normalized): the 3 facets whose normals are most aligned with dd
// generically meet at the vertex that maximizes dd.y. Falls back to the
// single best-aligned facet's plane if that 3x3 intersection is degenerate.
static Eigen::Vector3d support_seed_polytope(const Eigen::Vector3d& dd, int m,
                                              const double* A_data, const double* b_data)
{
    using Matm3Col = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::ColMajor>;
    Eigen::Map<const Matm3Col> A(A_data, m, 3);
    Eigen::Map<const Eigen::VectorXd> b(b_data, m);

    int i0 = -1, i1 = -1, i2 = -1;
    double s0 = -std::numeric_limits<double>::infinity();
    double s1 = -std::numeric_limits<double>::infinity();
    double s2 = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < m; ++i) {
        const double s = A.row(i).dot(dd);
        if (s > s0)      { s2 = s1; i2 = i1; s1 = s0; i1 = i0; s0 = s; i0 = i; }
        else if (s > s1) { s2 = s1; i2 = i1; s1 = s;  i1 = i; }
        else if (s > s2) { s2 = s;  i2 = i; }
    }

    if (i0 >= 0 && i1 >= 0 && i2 >= 0) {
        Eigen::Matrix3d M;
        Eigen::Vector3d rhs;
        M.row(0) = A.row(i0); rhs(0) = b(i0);
        M.row(1) = A.row(i1); rhs(1) = b(i1);
        M.row(2) = A.row(i2); rhs(2) = b(i2);

        Eigen::Vector3d y = M.fullPivLu().solve(rhs);
        if (y.allFinite() && (M * y - rhs).norm() < 1e-6 * (1.0 + rhs.cwiseAbs().maxCoeff()))
            return y;
    }

    if (i0 >= 0) return A.row(i0).transpose() * b(i0);  // ||A.row(i0)|| == 1
    return Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
}

// Rim seed for the smooth truncated cone (shape_id 3): the lateral surface
// is developable (radius linear in x1), so dd.y restricted to it is linear
// too -- its max is always the top or bottom rim. Pick whichever is larger.
static Eigen::Vector3d support_seed_cone(const Eigen::Vector3d& dd, double Rb, double Rt, double a, double b)
{
    const double dr = std::sqrt(dd(1) * dd(1) + dd(2) * dd(2));
    const double val_top = dd(0) * b + Rt * dr;
    const double val_bot = -dd(0) * a + Rb * dr;

    double x1, R;
    if (val_top >= val_bot) { x1 = b;  R = Rt; }
    else                    { x1 = -a; R = Rb; }

    if (dr > 0.0) return Eigen::Vector3d(x1, R * dd(1) / dr, R * dd(2) / dr);
    return Eigen::Vector3d(x1, 0.0, 0.0);
}

// Closed-form support point of an Lp-ball {sum_i |y_i/a_i|^p <= 1} via
// Holder duality (q = p/(p-1)): w_i = a_i*|dd_i|, y_i = a_i*sign(dd_i)*
// w_i^(q-1)/C^(q-1), C = (sum_i w_i^q)^(1/q). Used by shape_id 4
// (superellipsoid) and 5 (superelliptic cylinder, via a 2D radial reduction).
static Eigen::Vector3d support_seed_lp_ball(const Eigen::Vector3d& dd, double p,
                                             double a1, double a2, double a3)
{
    const double q = p / (p - 1.0);
    const double qm1 = q - 1.0;

    const double w1 = a1 * std::abs(dd(0));
    const double w2 = a2 * std::abs(dd(1));
    const double w3 = a3 * std::abs(dd(2));

    const double Cq = std::pow(w1, q) + std::pow(w2, q) + std::pow(w3, q);
    if (!(Cq > 0.0)) return Eigen::Vector3d::Zero();
    const double C = std::pow(Cq, 1.0 / q);
    const double Cpow = std::pow(C, qm1);

    const auto comp = [&](double a, double w, double di) {
        return (Cpow > 0.0) ? a * (di >= 0.0 ? 1.0 : -1.0) * std::pow(w, qm1) / Cpow : 0.0;
    };

    return Eigen::Vector3d(comp(a1, w1, dd(0)), comp(a2, w2, dd(1)), comp(a3, w3, dd(2)));
}

bool shape_support_point_local(
    const Vector3d& d,
    int shape_id,
    const VectorXd& params,
    Vector3d& y_support,
    double& k_support,
    int max_iters,
    double tol)
{
    const double dnorm = d.norm();
    if (!(dnorm > 0.0)) throw std::runtime_error("shape_support_point_local: d must be nonzero.");
    const Vector3d d_unit = d / dnorm;

    {
        double phi0;
        Vector3d g0;
        shape_eval_local_phi_grad(Vector3d::Zero(), shape_id, params, phi0, g0);
        if (!(phi0 < 0.0))
            throw std::runtime_error("shape_support_point_local: requires phi(0) < 0 (origin inside).");
    }

    Vector3d y;
    if (shape_id == 2 && params.size() >= 3) {
        // Smooth polytope: params = [beta; m; Lscale; A(:); b] -> vertex from
        // the 3 most d-aligned facets.
        const double* p = params.data();
        const int m = static_cast<int>(p[1]);
        const std::size_t expected = 3 + 4 * static_cast<std::size_t>(m);
        if (m >= 1 && static_cast<std::size_t>(params.size()) == expected) {
            y = support_seed_polytope(d_unit, m, p + 3, p + 3 + 3 * m);
        } else {
            y = Vector3d::Constant(std::numeric_limits<double>::quiet_NaN());
        }
        if (!y.allFinite()) {
            if (!ray_surface_intersection_local(d_unit, shape_id, params, y)) return false;
        }
    } else if (shape_id == 3 && params.size() >= 5) {
        // Smooth truncated cone: params = [beta; Rb; Rt; a; b] -> rim closed form.
        y = support_seed_cone(d_unit, params[1], params[2], params[3], params[4]);
        if (!y.allFinite()) {
            if (!ray_surface_intersection_local(d_unit, shape_id, params, y)) return false;
        }
    } else if (shape_id == 4 && params.size() >= 4) {
        // Superellipsoid: params = [n; a; b; c] -> exact Lp-ball closed form.
        const double p = 2.0 * std::round(params[0]);
        y = support_seed_lp_ball(d_unit, p, params[1], params[2], params[3]);
        if (!y.allFinite() || y.isZero(0)) {
            if (!ray_surface_intersection_local(d_unit, shape_id, params, y)) return false;
        }
    } else if (shape_id == 5 && params.size() >= 3) {
        // Superelliptic cylinder: params = [n; R; h]. Reduce to a 2D Lp-ball
        // in (axial, radial) coordinates, then re-expand the radial component
        // onto the (x2,x3) circle in direction (d(1),d(2)).
        const double p = 2.0 * std::round(params[0]);
        const double R = params[1], h = params[2];
        const double dr = std::sqrt(d_unit(1) * d_unit(1) + d_unit(2) * d_unit(2));
        const Eigen::Vector3d d2(d_unit(0), dr, 0.0);
        Eigen::Vector3d y2 = support_seed_lp_ball(d2, p, h, R, 1.0);
        if (y2.allFinite() && !y2.isZero(0)) {
            double r_support = y2(1);
            if (dr > 0.0) y = Vector3d(y2(0), r_support * d_unit(1) / dr, r_support * d_unit(2) / dr);
            else          y = Vector3d(y2(0), 0.0, 0.0);
        } else {
            if (!ray_surface_intersection_local(d_unit, shape_id, params, y)) return false;
        }
    } else {
        if (!ray_surface_intersection_local(d_unit, shape_id, params, y)) return false;
    }

    double phi_seed;
    Vector3d grad_seed;
    shape_eval_local_phi_grad(y, shape_id, params, phi_seed, grad_seed);
    double k = grad_seed.dot(d_unit);

    Eigen::LLT<Matrix4d> llt;

    // Newton on F(y,k) = [grad_phi(y) - k*d_unit; phi(y)] = 0, backtracked on
    // ||F||^2, with a Levenberg-Marquardt fallback for when the Newton
    // direction isn't a descent direction at all. That happens routinely on
    // flat/near-planar facets of the smooth-max and high-order superquadric
    // shapes, where H is near-singular and the plain Newton step points the
    // wrong way rather than just overshooting (so no amount of step-halving
    // helps -- a regularized direction is required instead).
    for (int it = 0; it < max_iters; ++it) {
        double phi;
        Vector3d grad;
        Matrix3d H;
        shape_eval_local(y, shape_id, params, phi, grad, H);

        Eigen::Vector4d F;
        F.head<3>() = grad - k * d_unit;
        F(3) = phi;

        const double F2 = F.squaredNorm();
        if (F2 <= tol * tol) {
            // F=0 alone only says grad_phi(y) is parallel to d; k<0 is the
            // *other* branch (the minimizer of d.y, opposite normal). Only
            // k>0 is the actual support point.
            if (!(k > 0.0)) return false;
            y_support = y;
            k_support = k;
            return true;
        }

        Matrix4d J = Matrix4d::Zero();
        J.block<3,3>(0,0) = H;
        J.block<3,1>(0,3) = -d_unit;
        J.block<1,3>(3,0) = grad.transpose();

        // Backtrack a candidate direction; accept the first step that
        // actually decreases ||F||^2.
        auto try_direction = [&](const Eigen::Vector4d& dir,
                                  Vector3d& y_new, double& k_new, Eigen::Vector4d& F_new) -> bool {
            double step = 1.0;
            for (int ls = 0; ls < 20; ++ls) {
                y_new = y + step * dir.head<3>();
                k_new = k + step * dir(3);

                double phi_n;
                Vector3d grad_n;
                shape_eval_local_phi_grad(y_new, shape_id, params, phi_n, grad_n);
                F_new.head<3>() = grad_n - k_new * d_unit;
                F_new(3) = phi_n;

                if (F_new.allFinite() && F_new.squaredNorm() < F2) return true;
                step *= 0.5;
            }
            return false;
        };

        Vector3d y_new;
        double k_new = k;
        Eigen::Vector4d F_new;
        bool improved = false;

        Eigen::Vector4d delta;
        if (solve_lu4(J, -F, delta) && delta.allFinite()) {
            improved = try_direction(delta, y_new, k_new, F_new);
        }

        if (!improved) {
            // LM fallback: regularized Gauss-Newton on the normal equations
            // (J^T J + mu I) delta = -J^T F, growing mu until the resulting
            // direction is a genuine descent direction for the merit.
            const Eigen::Vector4d g_m = J.transpose() * F;
            const Matrix4d JTJ = J.transpose() * J;

            double mu = 1e-6 * std::max(1.0, JTJ.trace() / 4.0);
            for (int tr = 0; tr < 12 && !improved; ++tr) {
                Matrix4d A = JTJ;
                A.diagonal().array() += mu;
                llt.compute(A);
                if (llt.info() == Eigen::Success) {
                    Eigen::Vector4d delta_lm = llt.solve(-g_m);
                    if (delta_lm.allFinite()) improved = try_direction(delta_lm, y_new, k_new, F_new);
                }
                mu *= 10.0;
            }
        }

        if (!improved) return false;

        y = y_new;
        k = k_new;
    }

    // final check
    double phi_chk;
    Vector3d grad_chk;
    Matrix3d H_chk;
    shape_eval_local(y, shape_id, params, phi_chk, grad_chk, H_chk);
    Eigen::Vector4d F_chk;
    F_chk.head<3>() = grad_chk - k * d_unit;
    F_chk(3) = phi_chk;

    if (F_chk.norm() > tol) return false;
    if (!(k > 0.0)) return false;

    y_support = y;
    k_support = k;
    return true;
}

bool shape_support_point_global(
    const Matrix4d& g,
    double alpha,
    const Vector3d& d_world,
    int shape_id,
    const VectorXd& params,
    Vector3d& x_support,
    double& k_support,
    int max_iters,
    double tol)
{
    if (alpha <= 0.0) throw std::runtime_error("shape_support_point_global: alpha must be > 0.");

    Matrix3d R = g.block<3,3>(0,0);
    Vector3d r = g.block<3,1>(0,3);

    // maximizing d_world.x over x = R*alpha*y + r reduces (alpha>0, R orthonormal)
    // to maximizing (R'*d_world).y over y, i.e. a local support query.
    Vector3d d_local = R.transpose() * d_world;

    Vector3d y_support;
    if (!shape_support_point_local(d_local, shape_id, params, y_support, k_support, max_iters, tol))
        return false;

    x_support = R * (alpha * y_support) + r;
    return true;
}

