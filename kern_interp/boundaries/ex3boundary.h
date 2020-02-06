// Copyright 2019 John Paul Ryan
#ifndef KERN_INTERP_BOUNDARIES_EX3BOUNDARY_H_
#define KERN_INTERP_BOUNDARIES_EX3BOUNDARY_H_

#include <vector>
#include <memory>
#include "kern_interp/boundaries/boundary.h"

namespace kern_interp {

class Ex3Boundary : public CubicBoundary {
 public:
  void initialize(int N, BoundaryCondition bc) override;

  void get_spline_points(std::vector<double>* x0_spline_points,
                         std::vector<double>* x1_spline_points) override;

  void get_star_spline_points(double x, double y,
                              std::vector<double>* x0_points,
                              std::vector<double>* x1_points);
  std::unique_ptr<Boundary> clone() const override {
    return std::make_unique<Ex3Boundary>(*this);
  }
};

}  // namespace kern_interp

#endif  // KERN_INTERP_BOUNDARIES_EX3BOUNDARY_H_
