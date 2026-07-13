#pragma once

#include <Eigen/Dense>
#include <cstdint>
#include <string>
#include <cmath>
#include "idcol_kkt.hpp"

namespace idcol {

using Matrix6d = Eigen::Matrix<double, 6, 6>;
using Vector6d = Eigen::Matrix<double, 6, 1>;
using Vector3d = Eigen::Vector3d;

struct NewtonOptions {
    // Scaling for x
    double L = 1.0;

    // Termination
    int    max_iters = 30;
    double tol = 1e-10;

    // Line search
    int    max_ls = 8;
    double beta_ls = 0.5;
    double c_armijo = 1e-4;

    // Soft trust-ish limits
    double max_step_s = 1.0;        // limit |Δs|
    double max_step_mult = 1.0;     // max_step = max_step_mult * L

    // Step sanity threshold in scaled coords
    double dz_hat_norm_max = 1e6;

    // s-restart logic
    bool   enable_restarts = true;
    double s_restart_delta = 0.1;   // try s0 ± 0.1
    double s_restart_extra = 0.2;   // optional extra push in best direction

    // Logging (optional)
    bool   verbose = false;

    double s_min = -std::numeric_limits<double>::infinity();
    double s_max =  std::numeric_limits<double>::infinity();
    bool   clamp_s_in_trials = true; // optional convenience

};

struct NewtonResult {
    Vector3d x = Vector3d::Zero();
    double   s = 0.0;
    double   alpha = 1.0;
    double   lambda1 = 1.0;
    double   lambda2 = 1.0;

    Vector6d F = Vector6d::Zero();
    Matrix6d J = Matrix6d::Zero();
    Matrix6d T = Matrix6d::Zero(); // T = -(J^-1)*G to be implemented later

    bool converged = false;
    int  iters_used = 0;
    int  attempts_used = 1;

    double final_F_norm = 0.0;
    std::string message;
};

// Pure C++ solver
NewtonResult solve_idcol_newton(
    const ProblemData& P,
    const Vector3d& x0,
    double alpha0,
    double lambda10,
    double lambda20,
    const NewtonOptions& opt = NewtonOptions()
);

} // namespace idcol
