// core/idcol_solve.hpp
#pragma once

#include <Eigen/Dense>
#include <optional>
#include <string>
#include <vector>

#include "radial_bounds.hpp"
#include "idcol_newton.hpp"   // for NewtonOptions + NewtonResult (and solve_idcol_newton)
#include "shape_core.hpp"     // for ProblemData

namespace idcol {

struct SolveData {
    ProblemData P;
    RadialBounds bounds1;
    RadialBounds bounds2;
};

struct Guess {
    Eigen::Vector3d x = Eigen::Vector3d::Zero();   // in body-1 frame
    double alpha   = 1.0;
    double lambda1 = 0.0;
    double lambda2 = 0.0;
};

struct SurrogateOptions {
    std::vector<int> fS_values = {1, 3, 5, 7};

    bool enable_scaling = true;
    // 'maxRout' or 'sumRout'
    std::string scale_mode = "maxRout";
};

struct SolveResult {
    NewtonResult newton;
    int  fS_used = 1;
    int  fS_attempts_used = 0;
    bool used_surrogate = false;
};

// Main API
SolveResult idcol_solve(const SolveData& S,
                        const std::optional<Guess>& user_guess = std::nullopt,
                        const NewtonOptions& opt_in = NewtonOptions{},
                        const SurrogateOptions& sopt = SurrogateOptions{});

// Convenience overloads
SolveResult idcol_solve(const SolveData& S, const NewtonOptions& opt);
SolveResult idcol_solve(const SolveData& S, const Guess& guess);
SolveResult idcol_solve(const SolveData& S, const Guess& guess, const NewtonOptions& opt);

} // namespace idcol
