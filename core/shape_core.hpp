#pragma once
#include <Eigen/Dense>

void shape_eval_local_phi( //phi only, no gradient/Hessian
    const Eigen::Vector3d& y,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi);

void shape_eval_local_phi_grad( //no Hessian
    const Eigen::Vector3d& y,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector3d& grad_phi);

void shape_eval_global_xa_phi_grad( //no Hessian
    const Eigen::Matrix4d& g,
    const Eigen::Vector3d& x,
    double alpha,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector4d& grad);

void shape_eval_local(
    const Eigen::Vector3d& y,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector3d& grad_phi,
    Eigen::Matrix3d& hess_phi);

void shape_eval_global_xa(
    const Eigen::Matrix4d& g,
    const Eigen::Vector3d& x,
    double alpha,
    int shape_id,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector4d& grad,      // [dphi/dx; dphi/dalpha]
    Eigen::Matrix4d& H);        // Hessian wrt [x; alpha]

// Ray-surface intersection in local (shape) frame: the point y0 = t*d_unit
// (t>0) on the boundary (phi(y0)=0) along direction d_unit from the origin.
// d_unit must be a unit vector. Requires phi(0) < 0 (origin inside).
bool ray_surface_intersection_local(
    const Eigen::Vector3d& d_unit,
    int shape_id,
    const Eigen::VectorXd& params,
    Eigen::Vector3d& y0_out);

// Ray-surface intersection in world frame for the shape placed by g
// (rotation R, origin r) and uniform scale alpha: x = R*alpha*y + r.
// d_world need not be unit.
bool ray_surface_intersection_global(
    const Eigen::Matrix4d& g,
    double alpha,
    const Eigen::Vector3d& d_world,
    int shape_id,
    const Eigen::VectorXd& params,
    Eigen::Vector3d& x0_out);

// Support point in local (shape) frame: the boundary point y (phi(y)=0)
// whose outward normal grad_phi(y) is parallel to direction d, i.e. the
// maximizer of d.y over the shape. Solved via Newton's method (quadratic
// convergence) on F(y,k) = [grad_phi(y) - k*d_unit; phi(y)] = 0, seeded by
// marching a ray from the origin along d_unit to the surface. Requires
// phi(0) < 0 (origin inside), same as compute_radial_bounds_local.
bool shape_support_point_local(
    const Eigen::Vector3d& d,
    int shape_id,
    const Eigen::VectorXd& params,
    Eigen::Vector3d& y_support,
    double& k_support,
    int max_iters = 30,
    double tol = 1e-10);

// Support point in world frame for the shape placed by g (rotation R,
// origin r) and uniform scale alpha: x = R*alpha*y + r. d_world need not
// be unit. k_support is the local-frame KKT multiplier (grad_phi(y) = k*d_unit).
bool shape_support_point_global(
    const Eigen::Matrix4d& g,
    double alpha,
    const Eigen::Vector3d& d_world,
    int shape_id,
    const Eigen::VectorXd& params,
    Eigen::Vector3d& x_support,
    double& k_support,
    int max_iters = 30,
    double tol = 1e-10);
