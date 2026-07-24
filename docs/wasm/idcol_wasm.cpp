// docs/wasm/idcol_wasm.cpp
//
// Emscripten/WASM bindings exposing the REAL iDCOL C++ solver
// (core/idcol_solve.cpp, idcol_newton.cpp, shape_core.cpp, radial_bounds.cpp)
// to the "solve" demo on the project website. This file only INCLUDES core/
// headers/sources and adds a thin binding layer -- core/ itself is untouched.
//
// Four shapes are pre-registered (indices 0-3), using the exact same
// dimensions as examples/main.cpp:
//   0: Smooth Polytope        beta=20, A/b as in examples/main.cpp
//   1: Smooth Truncated Cone  beta=20, Rb=1.0, Rt=1.5, a=1.5, b=1.5
//   2: Superellipsoid         n=8, a=0.5, b=1.0, c=1.5
//   3: Superelliptic Cylinder n=8, r=1.0, h=2.0
//
// SolveSession exposes:
//   setup(keyA, keyB, rx,ry,rz, r00..r22)  -- (re)initializes a cold start
//     (Eq. 14) exactly as experiments/init_sensitivity/init_sensitivity.cpp
//     independently verified against ContactPair's own cold-start path.
//   stepOnce()   -- exactly ONE real Newton iteration, via
//                   idcol::solve_idcol_newton with max_iters=1, continuing
//                   from the previous iterate (surrogate frame).
//   solveFull()  -- the library's normal end-to-end path: a real
//                   idcol::ContactPair, solved with the standard
//                   fS in {1,3,5,7,9} surrogate schedule + continuation.
//
// Build (from docs/wasm/):
//   em++ idcol_wasm.cpp ../../core/idcol_solve.cpp ../../core/idcol_newton.cpp \
//        ../../core/idcol_kkt.cpp ../../core/shape_core.cpp ../../core/radial_bounds.cpp \
//        -I../../ -I<eigen include dir> -lembind -O3 \
//        -s MODULARIZE=1 -s EXPORT_NAME=IdcolModule -s ENVIRONMENT=web \
//        -s ALLOW_MEMORY_GROWTH=1 -s SINGLE_FILE=1 \
//        -o idcol_wasm.js

#include <emscripten/bind.h>
#include <emscripten/val.h>
#include <Eigen/Dense>
#include <cmath>
#include <vector>

#include "../../core/idcol_implicitfamily.hpp"
#include "../../core/idcol_contactpair.hpp"
#include "../../core/idcol_solve.hpp"
#include "../../core/shape_core.hpp"

using namespace idcol;
using namespace emscripten;

// ---------------------------------------------------------------------
// Shape registry (indices 0-3), same dimensions as examples/main.cpp
// ---------------------------------------------------------------------
static ShapeSpec buildShape(int key) {
    RadialBoundsOptions optr;
    optr.num_starts = 1000;

    switch (key) {
        case 0: { // Smooth Polytope
            Eigen::MatrixXd A(8, 3);
            A << 1, 1, 1,
                 1, -1, -1,
                -1, 1, -1,
                -1, -1, 1,
                -1, -1, -1,
                -1, 1, 1,
                 1, -1, 1,
                 1, 1, -1;
            Eigen::VectorXd b(8);
            b << 1.0, 1.0, 1.0, 1.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0;
            return make_poly(20.0, A, b, optr);
        }
        case 1: // Smooth Truncated Cone
            return make_tc(20.0, 1.0, 1.5, 1.5, 1.5, optr);
        case 2: // Superellipsoid
            return make_se(8, 0.5, 1.0, 1.5, optr);
        case 3: // Superelliptic Cylinder
        default:
            return make_sec(8, 1.0, 2.0, optr);
    }
}

static ShapeSpec& getShape(int key) {
    static ShapeSpec shapes[4] = { buildShape(0), buildShape(1), buildShape(2), buildShape(3) };
    return shapes[key < 0 || key > 3 ? 3 : key];
}

val getBoundsJs(int key) {
    const ShapeSpec& s = getShape(key);
    val out = val::object();
    out.set("rin", s.bounds.Rin);
    out.set("rout", s.bounds.Rout);
    return out;
}

double evalPhiJs(int key, double x, double y, double z) {
    const ShapeSpec& s = getShape(key);
    double phi;
    shape_eval_local_phi(Eigen::Vector3d(x, y, z), s.shape_id, s.params, phi);
    return phi;
}

// ---------------------------------------------------------------------
// Cold start (Eq. 14) -- copied from experiments/init_sensitivity's
// independently-verified compute_cold_start (fS=1 case), not from core/
// itself (which keeps this identical logic private to idcol_solve.cpp).
// ---------------------------------------------------------------------
struct ColdStart {
    ProblemData Ps;
    Vector6d z0;
    double s_min = 0.0, s_max = 0.0;
    double scale_factor = 1.0;
};

static ColdStart compute_cold_start(const ProblemData& P, const RadialBounds& b1,
                                     const RadialBounds& b2, int fS = 1) {
    ColdStart out;
    const Eigen::Vector3d r0 = P.g.topRightCorner<3, 1>();
    const double d0 = r0.norm();
    const Eigen::Vector3d u = r0 / d0;

    const double alpha_min = d0 / (b1.Rout + b2.Rout);
    const double alpha_max = d0 / (b1.Rin + b2.Rin);
    out.scale_factor = double(fS) / alpha_min;

    out.Ps = P;
    const Eigen::Vector3d rS = u * (b1.Rout + b2.Rout) * double(fS);
    out.Ps.g.topRightCorner<3, 1>() = rS;

    const double alpha_min_scaled = double(fS);
    const double alpha_max_scaled = (alpha_max / alpha_min) * double(fS);
    out.s_min = std::log(alpha_min_scaled);
    out.s_max = std::log(alpha_max_scaled);

    const Eigen::Vector3d x0 = b1.Rout * u;
    const double alpha0 = std::sqrt(alpha_min_scaled * alpha_max_scaled);

    static const Eigen::Matrix4d I4 = Eigen::Matrix4d::Identity();
    double phi_tmp;
    Eigen::Vector4d grad_tmp;

    shape_eval_global_xa_phi_grad(I4, x0, alpha0, P.shape_id1, P.params1, phi_tmp, grad_tmp);
    const double denom1 = rS.dot(grad_tmp.head<3>());
    const double lambda10 = (std::abs(denom1) < 1e-14 || !std::isfinite(denom1)) ? 1.0 : (alpha0 / denom1);

    shape_eval_global_xa_phi_grad(out.Ps.g, x0, alpha0, P.shape_id2, P.params2, phi_tmp, grad_tmp);
    const double denom2 = rS.dot(grad_tmp.head<3>());
    const double lambda20 = (std::abs(denom2) < 1e-14 || !std::isfinite(denom2)) ? 1.0 : (-alpha0 / denom2);

    out.z0.head<3>() = x0;
    out.z0(3) = std::log(alpha0);
    out.z0(4) = lambda10;
    out.z0(5) = lambda20;
    return out;
}

static Eigen::Matrix4d makeG(double rx, double ry, double rz,
                              double r00, double r01, double r02,
                              double r10, double r11, double r12,
                              double r20, double r21, double r22) {
    Eigen::Matrix4d g = Eigen::Matrix4d::Identity();
    g(0, 0) = r00; g(0, 1) = r01; g(0, 2) = r02;
    g(1, 0) = r10; g(1, 1) = r11; g(1, 2) = r12;
    g(2, 0) = r20; g(2, 1) = r21; g(2, 2) = r22;
    g(0, 3) = rx;  g(1, 3) = ry;  g(2, 3) = rz;
    return g;
}

// ---------------------------------------------------------------------
// SolveSession: stateful wrapper driving the real solver step by step
// ---------------------------------------------------------------------
class SolveSession {
public:
    void setup(int keyA, int keyB, double rx, double ry, double rz,
               double r00, double r01, double r02,
               double r10, double r11, double r12,
               double r20, double r21, double r22) {
        keyA_ = keyA; keyB_ = keyB;
        const ShapeSpec& sA = getShape(keyA);
        const ShapeSpec& sB = getShape(keyB);

        P_.shape_id1 = sA.shape_id; P_.params1 = sA.params;
        P_.shape_id2 = sB.shape_id; P_.params2 = sB.params;
        P_.g = makeG(rx, ry, rz, r00, r01, r02, r10, r11, r12, r20, r21, r22);

        cs_ = compute_cold_start(P_, sA.bounds, sB.bounds, /*fS=*/1);
        z_ = cs_.z0;
        iters_ = 0;
        converged_ = false;
    }

    val stepOnce() {
        NewtonOptions opt;
        opt.L = 1;
        opt.max_iters = 1;
        opt.tol = 1e-10;
        opt.s_min = cs_.s_min;
        opt.s_max = cs_.s_max;

        NewtonResult r = solve_idcol_newton(cs_.Ps, z_.head<3>(), std::exp(z_(3)), z_(4), z_(5), opt);
        z_.head<3>() = r.x;
        z_(3) = std::log(r.alpha);
        z_(4) = r.lambda1;
        z_(5) = r.lambda2;
        iters_++;
        converged_ = r.converged;
        return packResult(r, iters_);
    }

    val solveFull() {
        NewtonOptions opt;
        opt.L = 1; opt.max_iters = 30; opt.tol = 1e-10;
        SurrogateOptions sopt;
        sopt.fS_values = {1, 3, 5, 7, 9};
        sopt.enable_scaling = false;

        ContactPair pair(getShape(keyA_), getShape(keyB_), opt, sopt);
        SolveResult out = pair.solve(P_.g);

        iters_ = out.newton.iters_used;
        converged_ = out.newton.converged;

        val res = val::object();
        res.set("converged", out.newton.converged);
        res.set("x", val::array(std::vector<double>{out.newton.x(0), out.newton.x(1), out.newton.x(2)}));
        res.set("alpha", out.newton.alpha);
        res.set("lambda1", out.newton.lambda1);
        res.set("lambda2", out.newton.lambda2);
        res.set("iters", out.newton.iters_used);
        res.set("normF", out.newton.final_F_norm);
        res.set("message", out.newton.message);
        return res;
    }

private:
    val packResult(const NewtonResult& r, int iters) {
        // surrogate frame -> original units (matches idcol_solve.cpp's
        // map_solution_to_original: x, alpha, lambda1, lambda2 all /= scale_factor)
        Eigen::Vector3d x_orig = r.x / cs_.scale_factor;
        double alpha_orig = r.alpha / cs_.scale_factor;
        double lambda1_orig = r.lambda1 / cs_.scale_factor;
        double lambda2_orig = r.lambda2 / cs_.scale_factor;

        val res = val::object();
        res.set("converged", r.converged);
        res.set("x", val::array(std::vector<double>{x_orig(0), x_orig(1), x_orig(2)}));
        res.set("alpha", alpha_orig);
        res.set("lambda1", lambda1_orig);
        res.set("lambda2", lambda2_orig);
        res.set("iters", iters);
        res.set("normF", r.final_F_norm);
        return res;
    }

    int keyA_ = 0, keyB_ = 0;
    ProblemData P_;
    ColdStart cs_;
    Vector6d z_;
    int iters_ = 0;
    bool converged_ = false;
};

EMSCRIPTEN_BINDINGS(idcol_module) {
    function("getBounds", &getBoundsJs);
    function("evalPhi", &evalPhiJs);

    class_<SolveSession>("SolveSession")
        .constructor<>()
        .function("setup", &SolveSession::setup)
        .function("stepOnce", &SolveSession::stepOnce)
        .function("solveFull", &SolveSession::solveFull);
}
