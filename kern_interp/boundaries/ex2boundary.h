// Copyright 2019 John Paul Ryan
#ifndef KERN_INTERP_BOUNDARIES_EX2BOUNDARY_H_
#define KERN_INTERP_BOUNDARIES_EX2BOUNDARY_H_

#include <vector>
#include <memory>
#include "kern_interp/boundaries/boundary.h"

namespace kern_interp {

class Ex2Boundary : public CubicBoundary {
 public:
  void initialize(int N, BoundaryCondition bc) override;

  void get_spline_points(std::vector<double>* outer_x0_spline_points,
                         std::vector<double>* outer_x1_spline_points) override;
  std::unique_ptr<Boundary> clone() const override {
    return std::make_unique<Ex2Boundary>(*this);
  }
};

}  // namespace kern_interp

#endif  // KERN_INTERP_BOUNDARIES_EX2BOUNDARY_H_
