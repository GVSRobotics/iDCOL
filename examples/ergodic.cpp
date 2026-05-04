// examples/main.cpp
#include <Eigen/Dense>
#include <Eigen/Geometry>
#include <Eigen/SVD>
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include <numeric>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <string>
#include <optional>

#include "core/idcol_implicitfamily.hpp"
#include "core/idcol_contactpair.hpp"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------------
// Ergodic pose generator 
// ---------------
Eigen::Matrix4d getSystematicPose(double t, double r_min, double r_max) {
    const double f1 = std::sqrt(2.0);
    const double f2 = std::sqrt(3.0);
    const double f3 = std::sqrt(5.0);
    const double f4 = std::sqrt(7.0);
    const double f5 = std::sqrt(11.0);
    const double f6 = std::sqrt(13.0);
    const double f7 = std::sqrt(17.0);

    const double TWO_PI = 2.0 * M_PI;

    // Translation
    double u = 0.5 * (1.0 + std::sin(TWO_PI * f1 * t));
    double r_val = std::cbrt(
        r_min*r_min*r_min +
        (r_max*r_max*r_max - r_min*r_min*r_min) * u
    );

    double theta = (M_PI / 2.0) * std::sin(TWO_PI * f2 * t);
    double phi   = TWO_PI * f3 * t;

    Eigen::Vector3d pos;
    pos << r_val * std::cos(theta) * std::cos(phi),
           r_val * std::cos(theta) * std::sin(phi),
           r_val * std::sin(theta);

    // Orientation
    double v1 = std::sin(TWO_PI * f4 * t);
    double v2 = std::cos(TWO_PI * f5 * t);
    double v3 = std::sin(TWO_PI * f6 * t);
    double v4 = std::cos(TWO_PI * f7 * t);

    Eigen::Quaterniond q(v1, v2, v3, v4); // (w, x, y, z)
    q.normalize();

    Eigen::Matrix4d g2 = Eigen::Matrix4d::Identity();
    g2.topLeftCorner<3,3>() = q.toRotationMatrix();
    g2.topRightCorner<3,1>() = pos;
    return g2;
}


// ------------------------------
// Benchmark options: Pass A / Pass B
// ------------------------------
struct BenchOptions {
    bool use_warm_start = false;

    // Pass A (timing): set both false
    bool write_csv = false;

    // Pass B (diagnostics): enable and maybe subsample
    bool compute_svd = false;
    int  svd_stride  = 1000;  // compute SVD once every K successful solves (>=1)

    // Control the sweep
    double t_max = 100.0;
    double dt    = 1e-4;

    // For reporting: if true, compute median from sorted durations
    // (always true here; it’s cheap)
    bool report_median = true;
};

struct BenchResult {
    double avg_us = 0.0;
    double median_us = 0.0;
    double stddev_us = 0.0;
    double avg_iters = 0.0;
    double min_sigma_ratio = 0.0;
    double success_pct = 0.0;
    std::size_t succ = 0;
    std::size_t failed = 0;
};

static BenchResult run_case(const idcol::ShapeSpec& s1, const idcol::ShapeSpec& s2, const BenchOptions& bo)
{
    using namespace idcol;
    using Clock = std::chrono::steady_clock;

    NewtonOptions opt;
    opt.L = 1;
    opt.max_iters = 30;
    opt.tol = 1e-10;
    opt.verbose = false;

    SurrogateOptions sopt;
    sopt.fS_values = {1, 3, 5, 7, 9};
    sopt.enable_scaling = false;

    // Contact pair 
    ContactPair pair(s1, s2, opt, sopt);

    const double r_min = 0.1 * std::min(s1.bounds.Rin,  s2.bounds.Rin);
    const double r_max = 2.0 * std::max(s1.bounds.Rout, s2.bounds.Rout);

    const int N = static_cast<int>(std::round(bo.t_max / bo.dt)) + 1;

    std::vector<double> durations_us;
    durations_us.reserve(N);
    std::vector<int> iters_used;
    iters_used.reserve(N);
    std::vector<double> j_ratios;
    if (bo.compute_svd) j_ratios.reserve(N / std::max(1, bo.svd_stride));

    std::ofstream csv;
    bool csv_header_written = false;
    if (bo.write_csv) {
        const std::string csv_name = s1.name + std::string("-") + s2.name +
                                     (bo.use_warm_start ? "_warm" : "_cold") + ".csv";
        csv.open(csv_name);
        if (!csv) std::cerr << "Warning: could not open CSV file '" << csv_name << "' for writing\n";
    }

    std::size_t failed = 0;

    int svd_counter = 0;

    for (int i = 0; i < N; ++i) {
        const double t = i * bo.dt;
        const Eigen::Matrix4d g = getSystematicPose(t, r_min, r_max); // relative ergodic pose g(t)

        if (!bo.use_warm_start) pair.reset_guess();

        const auto solve_t0 = Clock::now();
        SolveResult out = pair.solve(g);
        const auto solve_t1 = Clock::now();

        const double dur_us =
            std::chrono::duration_cast<std::chrono::duration<double, std::micro>>(solve_t1 - solve_t0).count();

        if (!out.newton.converged) {
            failed++;
            continue;
        }

        // Pass A timing collection: only store on success
        durations_us.push_back(dur_us);
        iters_used.push_back(out.newton.iters_used);

        // Optional CSV (kept out of timed region)
        if (bo.write_csv && csv.is_open()) {
            if (!csv_header_written) {
                const int xlen = static_cast<int>(out.newton.x.size());
                csv << "t";
                for (int k = 0; k < xlen; ++k) csv << ",x" << k;
                csv << ",alpha\n";
                csv_header_written = true;
            }
            csv << std::setprecision(15) << t;
            for (int k = 0; k < out.newton.x.size(); ++k)
                csv << "," << std::setprecision(15) << out.newton.x(k);
            csv << "," << std::setprecision(15) << out.newton.alpha << "\n";
        }

        // Optional diagnostics (SVD) with subsampling to avoid distorting runtime environment
        if (bo.compute_svd) {
            svd_counter++;
            if (svd_counter >= std::max(1, bo.svd_stride)) {
                svd_counter = 0;

                // If J is always 6x6 in your solver, consider changing its type there.
                Eigen::MatrixXd Jmat = out.newton.J;
                Eigen::JacobiSVD<Eigen::MatrixXd> svd(Jmat, Eigen::ComputeThinU | Eigen::ComputeThinV);
                const Eigen::VectorXd svals = svd.singularValues();
                const double sigma_min = svals.minCoeff();
                const double sigma_max = svals.maxCoeff();
                const double ratio = (sigma_max > 0.0) ? (sigma_min / sigma_max) : 0.0;
                j_ratios.push_back(ratio);
            }
        }

    }

    if (bo.write_csv && csv.is_open()) csv.close();

    BenchResult R;
    R.succ = durations_us.size();
    R.failed = failed;
    R.success_pct = 100.0 * double(R.succ) / double(N);

    if (!durations_us.empty()) {
        const double sum = std::accumulate(durations_us.begin(), durations_us.end(), 0.0);
        R.avg_us = sum / double(durations_us.size());

        std::sort(durations_us.begin(), durations_us.end());
        R.median_us = durations_us[durations_us.size() / 2]; // N is odd in your setup

        double sq = 0.0;
        for (double v : durations_us) sq += (v - R.avg_us) * (v - R.avg_us);
        R.stddev_us = std::sqrt(sq / double(durations_us.size()));
    }

    if (!iters_used.empty()) {
        const double sumi = std::accumulate(iters_used.begin(), iters_used.end(), 0.0);
        R.avg_iters = sumi / double(iters_used.size());
    }

    if (!j_ratios.empty()) {
        R.min_sigma_ratio = *std::min_element(j_ratios.begin(), j_ratios.end());
    } else {
        R.min_sigma_ratio = 0.0;
    }

    std::cout << "CASE " << s1.name << "-" << s2.name
          << (bo.use_warm_start ? " [warm]" : " [cold]")
          << " | avg_us=" << R.avg_us
          << " | median_us=" << R.median_us
          << " | stddev_us=" << R.stddev_us
          << " | avg_iters=" << R.avg_iters;

    if (bo.compute_svd) {
        std::cout << " | min_sigma_ratio=" << R.min_sigma_ratio;
    }

    std::cout << " | success=" << R.success_pct
            << "% (" << R.succ << " succ, " << R.failed << " failed)\n";


    return R;
}

int main() {
    // For stable timing runs
    Eigen::setNbThreads(1);

    // ----------------- Polytope--------------
    Eigen::MatrixXd A1(8,3);
    A1 <<  1,  1,  1,
           1, -1, -1,
          -1,  1, -1,
          -1, -1,  1,
          -1, -1, -1,
          -1,  1,  1,
           1, -1,  1,
           1,  1, -1;

    Eigen::VectorXd b1(8);
    b1 << 1.0, 1.0, 1.0, 1.0,
          5.0/3.0, 5.0/3.0, 5.0/3.0, 5.0/3.0;

    // ----------------- Smooth Truncated Cone--------------
    const double rb = 1.0;
    const double rt = 1.5;
    const double ac = 1.5;
    const double bc = 1.5;

    // ----------------- Superellipsoid--------------
    const double a = 0.5;
    const double b = 1.0;
    const double c = 1.5;

    // ----------------- Superelliptic Cylinder--------------
    const double r = 1.0;
    const double h = 2.0; // half-height

    const double beta = 20.0;
    int n = 8;

    RadialBoundsOptions optr;
    optr.num_starts = 1000;

    auto poly = idcol::make_poly(beta, A1, b1, optr);        // note arg order
    auto tc   = idcol::make_tc(beta, rb, rt, ac, bc, optr);
    auto se   = idcol::make_se(n, a, b, c, optr);
    auto sec  = idcol::make_sec(n, r, h, optr);

    Eigen::VectorXd custom_params(0);
    auto custom = idcol::make_custom(custom_params, optr);

    std::vector<idcol::ShapeSpec> shapes = {poly, tc, se, sec, custom};

    // -------------------------
    // Pass A: TIMING ONLY
    // -------------------------
    {
        BenchOptions bo;
        bo.use_warm_start = false; // you can loop both below
        bo.write_csv = false;
        bo.compute_svd = false;    // IMPORTANT: no SVD in timing pass
        bo.t_max = 100.0;
        bo.dt    = 1e-4;

        for (bool warm : {false, true}) {
            bo.use_warm_start = warm;
            std::cout << "\n=== PASS A (timing) warm_start=" << warm << " ===\n";
            for (const auto& s1 : shapes)
                for (const auto& s2 : shapes)
                    run_case(s1, s2, bo);
        }

        // DCOL comparison cases (ellipsoid is n=1 in idcol convention)
        n = 1;
        auto ellip = idcol::make_se(n, a, b, c, optr);

        for (bool warm : {false, true}) {
            bo.use_warm_start = warm;
            std::cout << "\n=== PASS A (timing, DCOL subset) warm_start=" << warm << " ===\n";
            run_case(poly,  poly,  bo);
            run_case(poly,  ellip, bo);
            run_case(ellip, ellip, bo);
        }
    }

    // -------------------------
    // Pass B: DIAGNOSTICS (SVD, optional CSV), subsampled
    // -------------------------
    {
        BenchOptions bo;
        bo.use_warm_start = false;
        bo.write_csv = false;      // set true only if you really want per-sample outputs
        bo.compute_svd = true;
        bo.svd_stride  = 1000;     // SVD every 1000 successful solves (change as you like)
        bo.t_max = 100.0;
        bo.dt    = 1e-4;

        std::cout << "\n=== PASS B (diagnostics) ===\n";
        run_case(se, se, bo);      // examples
        run_case(se, sec, bo);
        run_case(sec, sec, bo);
        run_case(custom, custom, bo);
        run_case(custom, poly, bo);
    }

    return 0;
}
