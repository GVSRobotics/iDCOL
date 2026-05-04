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


inline double eps_u_from_n(int n) {
    const double eps_min = 1e-12;   // enough to prevent 0/0, tiny geometry perturbation
    const double eps_max = 2e-3;
    // Geometry budget: max boundary shrink ~ eps/2 in normalized coords.
    // For max 0.1% change at n = 8 => eps_max = 0.002

    if (n <= 1) return eps_min;

    const double t = std::min(1.0, std::max(0.0, (double(n) - 1.0) / 7.0));
    return std::max(eps_min, eps_max * t * t);
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

