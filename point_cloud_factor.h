#pragma once

#include <ceres/autodiff_cost_function.h>
#include <ceres/cost_function.h>
#include <Eigen/Core>
#include <Eigen/Geometry>

#include "base/point_cloud/point_cloud_constraint.h"

namespace offline_localization {
namespace point_cloud {

struct EdgeFactor {
  EdgeFactor(const EdgeConstraint& edge_constraint) : edge_constraint_(edge_constraint) {}

  template <typename T>
  bool operator()(const T* q, const T* t, T* residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> rotation(q);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(t);

    const Eigen::Matrix<T, 3, 1> point = edge_constraint_.point.cast<T>();
    const Eigen::Matrix<T, 3, 1> edge_point1 = edge_constraint_.edge_point1.cast<T>();
    const Eigen::Matrix<T, 3, 1> edge_point2 = edge_constraint_.edge_point2.cast<T>();

    const Eigen::Matrix<T, 3, 1> point_w = rotation * point + translation;
    const Eigen::Matrix<T, 3, 1> point_cross = (point_w - edge_point1).cross(point_w - edge_point2);
    const T edge_length = (edge_point1 - edge_point2).norm();

    residual[0] = point_cross.x() / edge_length;
    residual[1] = point_cross.y() / edge_length;
    residual[2] = point_cross.z() / edge_length;

    return true;
  }

  static ceres::CostFunction* Create(const EdgeConstraint& edge_constraint) {
    return (new ceres::AutoDiffCostFunction<EdgeFactor, 3, 4, 3>(new EdgeFactor(edge_constraint)));
  }

  const EdgeConstraint edge_constraint_;
};

struct PlaneFactor {
  PlaneFactor(const PlaneConstraint& plane_constraint) : plane_constraint_(plane_constraint) {
    plane_unit_norm_ = (plane_constraint.plane_point1 - plane_constraint.plane_point2)
                           .cross(plane_constraint.plane_point1 - plane_constraint.plane_point3);
    plane_unit_norm_.normalize();
  }

  template <typename T>
  bool operator()(const T* q, const T* t, T* residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> rotation(q);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(t);

    const Eigen::Matrix<T, 3, 1> point = plane_constraint_.point.cast<T>();
    const Eigen::Matrix<T, 3, 1> plane_point1 = plane_constraint_.plane_point1.cast<T>();

    const Eigen::Matrix<T, 3, 1> point_w = rotation * point + translation;

    residual[0] = (point_w - plane_point1).dot(plane_unit_norm_);

    return true;
  }

  static ceres::CostFunction* Create(const PlaneConstraint& plane_constraint) {
    return (
        new ceres::AutoDiffCostFunction<PlaneFactor, 1, 4, 3>(new PlaneFactor(plane_constraint)));
  }

  const PlaneConstraint plane_constraint_;
  Eigen::Vector3d plane_unit_norm_ = Eigen::Vector3d::Zero();
};

struct PlaneNormFactor {
  PlaneNormFactor(const PlaneNormConstraint& plane_norm_constraint)
      : plane_norm_constraint_(plane_norm_constraint) {}

  template <typename T>
  bool operator()(const T* q, const T* t, T* residual) const {
    Eigen::Map<const Eigen::Quaternion<T>> rotation(q);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> translation(t);

    const Eigen::Matrix<T, 3, 1> point = plane_norm_constraint_.point.cast<T>();
    const Eigen::Matrix<T, 3, 1> plane_unit_norm = plane_norm_constraint_.plane_unit_norm.cast<T>();
    const T oa_dot_norm = static_cast<T>(plane_norm_constraint_.oa_dot_norm);

    const Eigen::Matrix<T, 3, 1> point_w = rotation * point + translation;

    residual[0] = point_w.dot(plane_unit_norm) + oa_dot_norm;
    return true;
  }

  static ceres::CostFunction* Create(const PlaneNormConstraint& plane_norm_constraint) {
    return (new ceres::AutoDiffCostFunction<PlaneNormFactor, 1, 4, 3>(
        new PlaneNormFactor(plane_norm_constraint)));
  }

  const PlaneNormConstraint plane_norm_constraint_;
};

}  // namespace point_cloud
}  // namespace offline_localization
