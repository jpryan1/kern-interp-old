// Copyright 2019 John Paul Ryan
#include <cmath>
#include <iostream>
#include "kern_interp/boundaries/uniformdist.h"

namespace kern_interp {

void UniformDist::initialize(int N, BoundaryCondition bc) {
  points.clear();
  for (int i = 0; i < sqrt(N); i++) {
    for (int j = 0; j < sqrt(N); j++) {
      points.push_back(0.00123 + i / (sqrt(N)));
      points.push_back(0.00123 + j / (sqrt(N)));
    }
  }

  boundary_values = ki_Mat::rand_vec(points.size()/2);
}


bool UniformDist::is_in_domain(const PointVec& a) const {
  return true;
}

}  // namespace kern_interp
