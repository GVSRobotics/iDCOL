#include "idcol_kkt.hpp"

namespace idcol {

void eval_F_J(
    const Vector3d& x,
    double s,
    double lambda1,
    double lambda2,
    const ProblemData& P,
    Vector6d& F,
    Matrix6d& J)
{
    const double alpha = std::exp(s);

    double   phi1;
    Vector4d grad1;
    Matrix4d H1;
    shape_eval_global_xa(Eigen::Matrix4d::Identity(), x, alpha, P.shape_id1, P.params1, phi1, grad1, H1);

    double   phi2;
    Vector4d grad2;
    Matrix4d H2;
    shape_eval_global_xa(P.g, x, alpha, P.shape_id2, P.params2, phi2, grad2, H2);

    // ==========================================
    // OPTIMIZATION 1: Direct Block Assignment
    // Avoids creating temporary vectors/matrices (e.g., Vector3d g1x = grad1.head<3>())
    // ==========================================

    // Residual F(z)
    F(0) = phi1;
    F(1) = phi2;
    F.segment<3>(2) = lambda1 * grad1.head<3>() + lambda2 * grad2.head<3>();
    F(5) = 1.0 + lambda1 * grad1(3) + lambda2 * grad2(3);

    // ==========================================
    // OPTIMIZATION 2: Vectorized Jacobian Construction
    // Removed the manual for-loop. Eigen can now SIMD-vectorize the entire 3x3 block.
    // ==========================================
    
    // Rows 0 and 1
    J.block<1,3>(0,0) = grad1.head<3>().transpose();
    J(0,3) = grad1(3) * alpha;
    J(0,4) = 0.0; J(0,5) = 0.0; // Ensure cleanliness

    J.block<1,3>(1,0) = grad2.head<3>().transpose();
    J(1,3) = grad2(3) * alpha;
    J(1,4) = 0.0; J(1,5) = 0.0;

    // Rows 2..4: Fx = λ1 ∇x φ1 + λ2 ∇x φ2
    J.block<3,3>(2,0) = lambda1 * H1.block<3,3>(0,0) + lambda2 * H2.block<3,3>(0,0);
    J.block<3,1>(2,3) = (lambda1 * H1.block<3,1>(0,3) + lambda2 * H2.block<3,1>(0,3)) * alpha;
    J.block<3,1>(2,4) = grad1.head<3>();
    J.block<3,1>(2,5) = grad2.head<3>();

    // Row 5: F6 = 1 + λ1 g1a + λ2 g2a
    J.block<1,3>(5,0) = lambda1 * H1.block<1,3>(3,0) + lambda2 * H2.block<1,3>(3,0);
    J(5,3) = (lambda1 * H1(3,3) + lambda2 * H2(3,3)) * alpha;
    J(5,4) = grad1(3);
    J(5,5) = grad2(3);
}

void eval_F(
    const Vector3d& x,
    double s,
    double lambda1,
    double lambda2,
    const ProblemData& P,
    Vector6d& F)
{
    const double alpha = std::exp(s);

    double   phi1;
    Vector4d grad1;
    shape_eval_global_xa_phi_grad(Eigen::Matrix4d::Identity(), x, alpha, P.shape_id1, P.params1, phi1, grad1);

    double   phi2;
    Vector4d grad2;
    shape_eval_global_xa_phi_grad(P.g, x, alpha, P.shape_id2, P.params2, phi2, grad2);

    F(0) = phi1;
    F(1) = phi2;
    F.segment<3>(2) = lambda1 * grad1.head<3>() + lambda2 * grad2.head<3>();
    F(5) = 1.0 + lambda1 * grad1(3) + lambda2 * grad2(3);
}

} // namespace idcol