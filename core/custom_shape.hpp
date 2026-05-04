#pragma once

#include <Eigen/Dense>

// User-editable custom implicit shape for shape_id = 6.
//
// Define the local implicit function and its derivatives here:
//   phi(y, params)        = 0 on the surface
//   grad_phi(y, params)   = nabla phi
//   hess_phi(y, params)   = nabla^2 phi
//
// The engine evaluates this local shape and then applies the same global
// pose/scale chain rule used by the built-in shapes.
inline void custom_shape_eval_local(
    const Eigen::Vector3d& y,
    const Eigen::VectorXd& params,
    double& phi,
    Eigen::Vector3d& grad_phi,
    Eigen::Matrix3d& hess_phi)
{
    (void)params; // This example is fully hard-coded.

    Eigen::Matrix3d A;
    A << 2.3, 0.4, 0.2,
         0.4, 1.7, 0.3,
         0.2, 0.3, 1.2;

    const Eigen::Vector3d b(0.25, -0.15, 0.10);
    const double c = 1.0;

    phi = y.dot(A * y) + b.dot(y) - c;
    grad_phi = 2.0 * A * y + b;
    hess_phi = 2.0 * A;
}
