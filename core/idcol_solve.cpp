// core/idcol_solve.cpp

#include "idcol_solve.hpp"

#include <cmath>
#include <limits>
#include <utility>  // for std::move if needed

namespace idcol {

// ---------------- internal helpers ----------------

static const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();


static bool shape_uses_n(int shape_id) {
    return (shape_id == 4) || (shape_id == 5); // SE or SEC
}

static bool shape_uses_beta(int shape_id) {
    return (shape_id == 2) || (shape_id == 3); // Poly or TC
}

static int get_integer_n_or_default(const Eigen::VectorXd& params, int default_n = 2) {
    if (params.size() == 0) return default_n;
    const double n_raw = params[0];
    const int n = static_cast<int>(std::round(n_raw));
    if (n <= 0 || std::fabs(n_raw - n) > 1e-9) return default_n;
    return n;
}

static void set_n_in_params(Eigen::VectorXd& params, int n) {
    if (params.size() > 0) params[0] = double(n);
}

// params scaling helper
static void idcol_scale_params_inplace(int shape_id, Eigen::VectorXd& params, double s) {
    switch (shape_id) {
        case 1: { // Sphere: [R]
            params(0) *= s;
            break;
        }
        case 2: { // Polytope: [beta; m; Lscale; A(:); b]
            const int m = static_cast<int>(std::llround(params(1)));
            params(2) *= s;

            const int b_start = 3 + 3 * m;
            params.segment(b_start, m) *= s;
            break;
        }
        case 3: { // Truncated cone: [beta; Rb; Rt; a; b]
            params.segment(1, 4) *= s;
            break;
        }
        case 4: { // Superellipsoid: [n; a; b; c]
            params.segment(1, 3) *= s;
            break;
        }
        case 5: { // Superelliptic cylinder: [n; R; h]
            params.segment(1, 2) *= s;
            break;
        }
        default:
            break;
    }
}

// map guess: original -> surrogate
static Guess map_guess_to_surrogate(const Guess& g, double scale_factor) {
    Guess gs = g;
    gs.x       = g.x * scale_factor;
    gs.alpha   = g.alpha * scale_factor;
    gs.lambda1 = g.lambda1 * scale_factor;
    gs.lambda2 = g.lambda2 * scale_factor;
    return gs;
}

// map solution: surrogate -> original
static void map_solution_to_original(NewtonResult& res, double scale_factor) {
    res.x       /= scale_factor;
    res.alpha   /= scale_factor;
    res.lambda1 /= scale_factor;
    res.lambda2 /= scale_factor;
    res.J *= scale_factor; // surrogate problem has same F.
}

// ---------------- implementation ----------------

SolveResult idcol_solve(const SolveData& S,
                        const std::optional<Guess>& user_guess,
                        const NewtonOptions& opt_in,
                        const SurrogateOptions& sopt)
{
    const ProblemData& P_in = S.P;
    const RadialBounds& bounds1 = S.bounds1;
    const RadialBounds& bounds2 = S.bounds2;

    // recursion guard (continuation calls idcol_solve again)
    static thread_local int solve_depth = 0;

    struct SolveDepthGuard {
        int& d;
        explicit SolveDepthGuard(int& d_) : d(d_) { ++d; }
        ~SolveDepthGuard() { --d; }
    } _guard(solve_depth);

    SolveResult out;

    // Work on a local copy because we overwrite P.g for the surrogate problem.
    // Reused per-recursion-depth (not a single shared static, so the
    // recursive continuation calls below don't clobber the caller's scratch)
    // instead of a fresh ProblemData every call: params1/params2 keep the
    // same size across repeated solves for a given shape pair, so
    // VectorXd::operator= has nothing to reallocate after the first call.
    constexpr int kMaxPooledDepth = 4;
    static thread_local ProblemData P_pool[kMaxPooledDepth];
    ProblemData& P = P_pool[std::min(solve_depth, kMaxPooledDepth) - 1];

    P.shape_id1 = P_in.shape_id1;
    P.shape_id2 = P_in.shape_id2;
    P.params1   = P_in.params1;
    P.params2   = P_in.params2;
    P.g         = P_in.g;

    // ------------------------------------------------------------------
    // Unit scaling (NOT surrogate scaling)
    //  - scales translations in g1,g2; bounds; params; user_guess.x
    //  - alpha and lambdas unchanged
    // ------------------------------------------------------------------
    double Lscale = 1.0;
    if (sopt.enable_scaling) {
        if (sopt.scale_mode == "sumRout") {
            Lscale = bounds1.Rout + bounds2.Rout;
        } else { // default "maxRout"
            Lscale = std::max(bounds1.Rout, bounds2.Rout);
        }
        if (!std::isfinite(Lscale) || Lscale <= 0.0) Lscale = 1.0;
    }
    const double invLscale = 1.0 / Lscale;

    RadialBounds b1 = bounds1;
    RadialBounds b2 = bounds2;

    if (sopt.enable_scaling && invLscale != 1.0) {
        b1.Rin  *= invLscale;  b1.Rout *= invLscale;
        b2.Rin  *= invLscale;  b2.Rout *= invLscale;

        P.g.topRightCorner<3,1>() *= invLscale;

        idcol_scale_params_inplace(P.shape_id1, P.params1, invLscale);
        idcol_scale_params_inplace(P.shape_id2, P.params2, invLscale);
    }

    const Eigen::Matrix4d g = P.g;
    const Eigen::Vector3d r0 = g.topRightCorner<3,1>();
    const double d0 = r0.norm();

    if (!std::isfinite(d0) || d0 <= 1e-15) {
        out.newton.converged    = false;
        out.newton.iters_used   = 0;
        out.newton.final_F_norm = std::numeric_limits<double>::infinity();
        out.newton.message =
            "idcol_solve: degenerate relative translation (||r|| ~ 0). "
            "Cannot build surrogate (u = r/||r||).";

        out.used_surrogate   = false;
        out.fS_used          = (sopt.fS_values.empty() ? 1 : sopt.fS_values.front());
        out.fS_attempts_used = 0;
        return out;
    }

    const Eigen::Vector3d u = r0 / d0;

    const double alpha_min = d0 / (b1.Rout + b2.Rout);
    const double alpha_max = d0 / (b1.Rin  + b2.Rin);

    auto attempt = [&](int fS) -> NewtonResult {
        NewtonOptions opt = opt_in;

        const double scale_factor = static_cast<double>(fS) / alpha_min;

        P.g = g;
        const Eigen::Vector3d rS = u * (b1.Rout + b2.Rout) * static_cast<double>(fS);
        P.g.topRightCorner<3,1>() = rS;

        const double alpha_min_scaled = static_cast<double>(fS);
        const double alpha_max_scaled = (alpha_max / alpha_min) * static_cast<double>(fS);

        opt.s_min = std::log(alpha_min_scaled);
        opt.s_max = std::log(alpha_max_scaled);

        Guess g0;

        if (user_guess.has_value()) {
            Guess g_user = *user_guess;
            if (sopt.enable_scaling && invLscale != 1.0) {
                g_user.x *= invLscale;
            }
            g0 = map_guess_to_surrogate(g_user, scale_factor);
        } else {
            g0.x     = b1.Rout * u;
            g0.alpha = std::sqrt(alpha_min_scaled * alpha_max_scaled);

            double phi_tmp;
            Eigen::Vector4d grad_tmp;

            shape_eval_global_xa_phi_grad(I4, g0.x, g0.alpha,
                                          P.shape_id1, P.params1, phi_tmp, grad_tmp);
            const double denom1 = rS.dot(grad_tmp.head<3>());
            g0.lambda1 = (std::abs(denom1) < 1e-14 || !std::isfinite(denom1)) ? 1.0 : (g0.alpha / denom1);

            shape_eval_global_xa_phi_grad(P.g, g0.x, g0.alpha,
                                          P.shape_id2, P.params2, phi_tmp, grad_tmp);
            const double denom2 = rS.dot(grad_tmp.head<3>());
            g0.lambda2 = (std::abs(denom2) < 1e-14 || !std::isfinite(denom2)) ? 1.0 : (-(g0.alpha) / denom2);
        }

        NewtonResult resS = solve_idcol_newton(P, g0.x, g0.alpha, g0.lambda1, g0.lambda2, opt);

        map_solution_to_original(resS, scale_factor);

        if (sopt.enable_scaling && invLscale != 1.0) {
            resS.x *= Lscale;

            resS.F.segment<3>(2) *= invLscale;   //lambda_1*phi1_x+lambda_2*phi2_x
            resS.J.block<3,6>(2,0) *= invLscale; //from residual scaling above
            resS.J.block<6,3>(0,0) *= invLscale; //z = [x,alpha,lambda1,lambda2]
        }

        return resS;
    };

    out.used_surrogate   = true;
    out.fS_attempts_used = 0;

    if (sopt.fS_values.empty()) {
        out.newton = attempt(1);
        out.fS_used = 1;
        out.fS_attempts_used = 1;
        return out;
    }

    for (std::size_t k = 0; k < sopt.fS_values.size(); ++k) {
        const int fS = sopt.fS_values[k];

        out.newton = attempt(fS);
        out.fS_used = fS;
        out.fS_attempts_used = static_cast<int>(k) + 1;

        if (out.newton.converged) break;
    }

    // Continuation strategies only at top-level call
    if (solve_depth == 1 && !out.newton.converged) {

        // ---- n-continuation (SE/SEC) ----
        const bool body1_has_n = shape_uses_n(P_in.shape_id1);
        const bool body2_has_n = shape_uses_n(P_in.shape_id2);

        const int n1_target = body1_has_n ? get_integer_n_or_default(P_in.params1, 2) : 0;
        const int n2_target = body2_has_n ? get_integer_n_or_default(P_in.params2, 2) : 0;
        const int n_target  = std::max(n1_target, n2_target);

        if (n_target > 2 && (body1_has_n || body2_has_n)) {
            std::vector<int> n_schedule;
            n_schedule.push_back(2);
            for (int nk = 4; nk < n_target; nk += 2) n_schedule.push_back(nk);
            if (n_schedule.back() != n_target) n_schedule.push_back(n_target);

            SolveResult best_out = out;
            double best_norm = out.newton.final_F_norm;

            std::optional<Guess> guess_k = std::nullopt;

            for (int nk : n_schedule) {
                if (nk > n_target) break;

                SolveData Sk = S;
                if (body1_has_n) set_n_in_params(Sk.P.params1, std::min(nk, n1_target));
                if (body2_has_n) set_n_in_params(Sk.P.params2, std::min(nk, n2_target));

                SolveResult out_k = idcol_solve(Sk, guess_k, opt_in, sopt);

                if (std::isfinite(out_k.newton.final_F_norm) && out_k.newton.final_F_norm < best_norm) {
                    best_out = out_k;
                    best_norm = out_k.newton.final_F_norm;
                }

                if (out_k.newton.converged) {
                    Guess g;
                    g.x = out_k.newton.x;
                    g.alpha = out_k.newton.alpha;
                    g.lambda1 = out_k.newton.lambda1;
                    g.lambda2 = out_k.newton.lambda2;
                    guess_k = g;

                    if (nk == n_target) {
                        out = out_k;
                        out.newton.message += " | converged via n-continuation";
                        return out;
                    }
                } else {
                    Guess g;
                    g.x = best_out.newton.x;
                    g.alpha = best_out.newton.alpha;
                    g.lambda1 = best_out.newton.lambda1;
                    g.lambda2 = best_out.newton.lambda2;
                    guess_k = g;
                }
            }

            out = best_out;
            out.newton.message += " | attempted n-continuation (fallback)";
        }

        // ---- beta-continuation (poly / tc) ----
        if (!out.newton.converged) {
            const bool body1_has_beta = shape_uses_beta(P_in.shape_id1);
            const bool body2_has_beta = shape_uses_beta(P_in.shape_id2);

            const double beta1 = body1_has_beta ? P_in.params1(0) : 0.0;
            const double beta2 = body2_has_beta ? P_in.params2(0) : 0.0;
            const double beta_target = std::max(beta1, beta2);

            if ((body1_has_beta || body2_has_beta) &&
                std::isfinite(beta_target) && beta_target > 1.0)
            {
                std::vector<double> beta_schedule = { beta_target/4.0, beta_target/2.0, beta_target };

                SolveResult best_out = out;
                double best_norm = out.newton.final_F_norm;

                std::optional<Guess> guess_k = std::nullopt;

                for (double bk : beta_schedule) {
                    if (!(bk > 0.0) || !std::isfinite(bk)) continue;

                    SolveData Sk = S;
                    if (body1_has_beta) Sk.P.params1(0) = std::min(P_in.params1(0), bk);
                    if (body2_has_beta) Sk.P.params2(0) = std::min(P_in.params2(0), bk);

                    SolveResult out_k = idcol_solve(Sk, guess_k, opt_in, sopt);

                    if (std::isfinite(out_k.newton.final_F_norm) && out_k.newton.final_F_norm < best_norm) {
                        best_out = out_k;
                        best_norm = out_k.newton.final_F_norm;
                    }

                    if (out_k.newton.converged) {
                        Guess g;
                        g.x = out_k.newton.x;
                        g.alpha = out_k.newton.alpha;
                        g.lambda1 = out_k.newton.lambda1;
                        g.lambda2 = out_k.newton.lambda2;
                        guess_k = g;

                        if (bk == beta_target) {
                            out = out_k;
                            out.newton.message += " | converged via beta-continuation";
                            return out;
                        }
                    } else {
                        Guess g;
                        g.x = best_out.newton.x;
                        g.alpha = best_out.newton.alpha;
                        g.lambda1 = best_out.newton.lambda1;
                        g.lambda2 = best_out.newton.lambda2;
                        guess_k = g;
                    }
                }

                out = best_out;
                out.newton.message += " | attempted beta-continuation (fallback)";
            }
        }
    }

    return out;
}

// ---- overloads ----

SolveResult idcol_solve(const SolveData& S, const NewtonOptions& opt) {
    return idcol_solve(S, std::nullopt, opt, SurrogateOptions{});
}

SolveResult idcol_solve(const SolveData& S, const Guess& guess) {
    return idcol_solve(S, std::optional<Guess>(guess), NewtonOptions{}, SurrogateOptions{});
}

SolveResult idcol_solve(const SolveData& S, const Guess& guess, const NewtonOptions& opt) {
    return idcol_solve(S, std::optional<Guess>(guess), opt, SurrogateOptions{});
}

} // namespace idcol
