#include "idcol_newton.hpp"
#include <iostream>

namespace idcol {

// Hand-written partial-pivoting Gaussian elimination for a general
// (non-symmetric) 6x6 system A*x = b, done in place on a local copy of A.
// Eigen::PartialPivLU<Matrix6d> carries generic machinery (permutation
// bookkeeping, determinant-sign tracking) that's pure overhead at a fixed
// 6x6 next to a flat elimination loop. Uses the same (partial, row-only)
// pivoting as Eigen, so the chosen pivot -- and hence the Newton iterate
// path -- is unchanged. Returns false (instead of Eigen's silent NaN/Inf)
// on an exactly-zero pivot; not a behavior change since callers already
// gate on `dz_hat.allFinite()`.
static inline bool solve_lu6(Matrix6d A, const Vector6d& b, Vector6d& x) {
    x = b;
    for (int k = 0; k < 5; ++k) {
        int piv = k;
        double amax = std::abs(A(k, k));
        for (int i = k + 1; i < 6; ++i) {
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
        for (int i = k + 1; i < 6; ++i) {
            const double f = A(i, k) * invp;
            if (f != 0.0) {
                for (int j = k; j < 6; ++j) A(i, j) -= f * A(k, j);
                x(i) -= f * x(k);
            }
        }
    }
    if (A(5, 5) == 0.0) return false;
    for (int k = 5; k >= 0; --k) {
        double s = x(k);
        for (int j = k + 1; j < 6; ++j) s -= A(k, j) * x(j);
        x(k) = s / A(k, k);
    }
    return true;
}

NewtonResult solve_idcol_newton(
    const ProblemData& P,
    const Vector3d& x0,
    double alpha0,
    double lambda10,
    double lambda20,
    const NewtonOptions& opt_in)
{
    NewtonResult out;
    NewtonOptions opt = opt_in;

    auto clamp = [](double v, double lo, double hi) {
        return (v < lo) ? lo : (v > hi) ? hi : v;
    };
    auto clamp_s = [&](double s) {
        return clamp(s, opt.s_min, opt.s_max);
    };


    // Basic input sanity
    if (!(alpha0 > 0.0) || !std::isfinite(alpha0)) {
        out.converged = false;
        out.message = "alpha0 must be finite and > 0.";
        return out;
    }
    if (!(opt.L > 0.0) || !std::isfinite(opt.L)) opt.L = 1.0;

    double s0 = std::log(alpha0);
    s0 = clamp_s(s0);


    // scaling vector D for [x(3), s, lambda1, lambda2]
    Eigen::Matrix<double,6,1> D;
    D << opt.L, opt.L, opt.L, 1.0, 1.0, 1.0;

    const double max_step = opt.max_step_mult * opt.L;

    auto run_newton = [&](Vector3d& x, double& s, double& lambda1, double& lambda2,
                          Vector6d& F, Matrix6d& J,
                          int& iters_used) -> bool
    {
        bool solved = false;
        iters_used = 0;

        // Reuse decomposition objects and temporaries to avoid repeated allocations
        Eigen::LLT<Matrix6d> llt;
        Matrix6d A;

        for (int iter = 0; iter < opt.max_iters; ++iter) {
            iters_used = iter + 1;

            eval_F_J(x, s, lambda1, lambda2, P, F, J);

            const double Fn2 = F.squaredNorm();
            if (Fn2 < (opt.tol * opt.tol)) { solved = true; break; }

            const double m0 = idcol::merit_from_F(F);

            Vector6d g_m;
            g_m.noalias() = J.transpose() * F;

            // Build JD = J * diag(D)
            Matrix6d JD = J;
            for (int k = 0; k < 6; ++k) JD.col(k) *= D(k);

            // Solve JD * dz_hat = -F
            Vector6d dz_hat;
            bool have_step = false;

            const bool ok_lu = solve_lu6(JD, -F, dz_hat);

            if (ok_lu && dz_hat.allFinite() && dz_hat.squaredNorm() <= (opt.dz_hat_norm_max * opt.dz_hat_norm_max)) {
                have_step = true;
            }

            // LM fallback if LU step is bad
            if (!have_step) {
                Vector6d rhs;
                rhs.noalias() = -JD.transpose() * F;

                double mu = 1e-8;
                for (int k = 0; k < 10; ++k) {
                    A.noalias() = JD.transpose() * JD;
                    A.diagonal().array() += mu;

                    llt.compute(A);
                    if (llt.info() == Eigen::Success) {
                        dz_hat = llt.solve(rhs);
                        have_step = dz_hat.allFinite();
                        if (have_step) break;
                    }
                    mu *= 10.0;
                }
                if (!have_step) {
                    if (opt.verbose) std::cerr << "[idcol] LM fallback failed (LLT)\n";
                    break;
                }
            }

            // Unscale: dz = diag(D) * dz_hat
            Vector6d dz = D.array() * dz_hat.array();

            // Soft step limits
            if (std::isfinite(dz(3)) && std::abs(dz(3)) > opt.max_step_s)
                dz *= (opt.max_step_s / std::abs(dz(3)));

            double dz_norm = dz.norm();
            if (std::isfinite(dz_norm) && dz_norm > max_step)
                dz *= (max_step / dz_norm);

            // Ensure descent direction for merit
            double slope = g_m.dot(dz);
            if (!(slope < 0.0) || !std::isfinite(slope)) {
                Vector6d dz_sd = -g_m;
                double slope_sd = g_m.dot(dz_sd);

                if (!(slope_sd < 0.0) || !std::isfinite(slope_sd)) {
                    if (opt.verbose) std::cerr << "[idcol] Non-descent and invalid gradient\n";
                    break;
                }

                dz = dz_sd;
                slope = slope_sd;

                if (std::isfinite(dz(3)) && std::abs(dz(3)) > opt.max_step_s)
                    dz *= (opt.max_step_s / std::abs(dz(3)));

                dz_norm = dz.norm();
                if (std::isfinite(dz_norm) && dz_norm > max_step)
                    dz *= (max_step / dz_norm);
            }

            // Backtracking line search on m(z) = 0.5 ||F||^2

            double t_max_bound = 1.0;
            if (std::isfinite(dz(3)) && dz(3) != 0.0) {
                if (dz(3) > 0.0) t_max_bound = std::min(t_max_bound, (opt.s_max - s) / dz(3));
                else             t_max_bound = std::min(t_max_bound, (opt.s_min - s) / dz(3));
            }
            if (!(t_max_bound > 0.0)) t_max_bound = 0.0; // no feasible step in that direction


            double t = std::min(1.0, t_max_bound);
            bool accepted = false;

            Vector3d x_trial;
            double s_trial, l1_trial, l2_trial;
            Vector6d F_trial;

            for (int ls = 0; ls < opt.max_ls; ++ls) {
                x_trial = x + t * dz.segment<3>(0);
                s_trial = s + t * dz(3);
                if (opt.clamp_s_in_trials) s_trial = clamp_s(s_trial);

                l1_trial = lambda1 + t * dz(4);
                l2_trial = lambda2 + t * dz(5);

                eval_F(x_trial, s_trial, l1_trial, l2_trial, P, F_trial);

                const double m_trial = idcol::merit_from_F(F_trial);
                if (std::isfinite(m_trial) && (m_trial <= m0 + opt.c_armijo * t * slope)) {
                    accepted = true;
                    break;
                }
                t *= opt.beta_ls;
            }

            if (!accepted) {
                // Rescue: LM sweep on unscaled J 
                // reuse previously computed merit and gradient to avoid recomputation
                const double m0r = m0;

                // g_m holds J^T * F computed earlier; reuse it
                Vector6d& g = g_m;

                Matrix6d JTJ;
                JTJ.noalias() = J.transpose() * J;

                bool rescued = false;
                double mu = 1e-3;

                for (int tr = 0; tr < 10; ++tr) {
                    A = JTJ;
                    A.diagonal().array() += mu;

                    llt.compute(A);
                    if (llt.info() != Eigen::Success) {
                        mu *= 10.0;
                        continue;
                    }

                    Vector6d dz_r = llt.solve(-g);

                    if (std::isfinite(dz_r(3)) && std::abs(dz_r(3)) > opt.max_step_s)
                        dz_r *= (opt.max_step_s / std::abs(dz_r(3)));

                    double dzr_norm = dz_r.norm();
                    if (std::isfinite(dzr_norm) && dzr_norm > max_step)
                        dz_r *= (max_step / dzr_norm);

                    Vector3d x_r = x + dz_r.segment<3>(0);
                    double   s_r = s + dz_r(3);
                    if (opt.clamp_s_in_trials) s_r = clamp_s(s_r);

                    double   l1_r = lambda1 + dz_r(4);
                    double   l2_r = lambda2 + dz_r(5);

                    Vector6d F_r;
                    eval_F(x_r, s_r, l1_r, l2_r, P, F_r);

                    const double m_r = idcol::merit_from_F(F_r);
                    if (std::isfinite(m_r) && (m_r < m0r)) {
                        x = x_r; s = s_r; lambda1 = l1_r; lambda2 = l2_r;
                        rescued = true;
                        break;
                    }

                    mu *= 10.0;
                }

                if (!rescued) {
                    if (opt.verbose) std::cerr << "[idcol] Armijo failed; rescue failed\n";
                    break;
                }

                continue; // rescued step accepted
            }

            // Accept line search step
            x = x_trial;
            s = s_trial;
            lambda1 = l1_trial;
            lambda2 = l2_trial;

            if ((t * dz).norm() < 1e-14) break;
        }

        return solved;
    };

    // Initial guess
    Vector3d x = x0;
    double s = s0;
    double lambda1 = lambda10;
    double lambda2 = lambda20;

    Vector6d F;
    Matrix6d J;
    int iters_used = 0;

    bool converged = run_newton(x, s, lambda1, lambda2, F, J, iters_used);

    int attempts_used = 1;

    // Restart logic (optional)
    if (!converged && opt.enable_restarts) {
        Vector3d best_x = x;
        double   best_s = s;
        double   best_l1 = lambda1, best_l2 = lambda2;
        double   best_Fn = F.squaredNorm();

        auto attempt = [&](double s_init) -> bool {
            ++attempts_used;

            Vector3d x_try = x0;
            double s_try = clamp_s(s_init);
            double l1_try = lambda10;
            double l2_try = lambda20;

            Vector6d F_try;
            Matrix6d J_try;
            int it_try = 0;

            bool ok = run_newton(x_try, s_try, l1_try, l2_try, F_try, J_try, it_try);

            const double Fn2 = F_try.squaredNorm();
            if (std::isfinite(Fn2) && Fn2 < best_Fn) {
                best_Fn = Fn2;
                best_x = x_try; best_s = s_try; best_l1 = l1_try; best_l2 = l2_try;
            }

            if (ok) {
                x = x_try; s = s_try; lambda1 = l1_try; lambda2 = l2_try;
                F = F_try; J = J_try;
                iters_used = it_try;
            }
            return ok;
        };

        converged = attempt(s0 + opt.s_restart_delta);
        if (!converged) converged = attempt(s0 - opt.s_restart_delta);

        if (!converged) {
            const double dir = (best_s > s0) ? 1.0 : ((best_s < s0) ? -1.0 : 0.0);
            if (dir != 0.0) converged = attempt(s0 + opt.s_restart_extra * dir);
        }

        if (!converged) {
            x = best_x; s = best_s; lambda1 = best_l1; lambda2 = best_l2;
        }
    }

    // Final eval for returned outputs
    // If the solver converged (or an attempt succeeded) then `F` and `J` already
    // correspond to the current (x,s,lambda) and we can reuse them. Only call
    // `eval_F_J` when we are returning a best result from restarts (not converged).
    if (!converged) {
        eval_F_J(x, s, lambda1, lambda2, P, out.F, out.J);
    } else {
        out.F = F;
        out.J = J;
    }

    out.x = x;
    out.s = s;
    out.alpha = std::exp(s);
    out.lambda1 = lambda1;
    out.lambda2 = lambda2;

    out.J.col(3) /= out.alpha; //bring back to alpha space
    
    out.converged = converged;
    out.iters_used = iters_used;
    out.attempts_used = attempts_used;
    out.final_F_norm = out.F.norm();

    if (converged) {
        out.message = "converged";
    } else {
        out.message = opt.enable_restarts
            ? "not converged (returned best from restarts)"
            : "not converged (no restarts)";
    }

    return out;
}

} // namespace idcol
