// Copyright 2019 John Paul Ryan
#include <cmath>
#include <iostream>
#include "kern_interp/boundaries/sphere.h"
#include "kern_interp/legendre.h"

namespace kern_interp {

void Sphere::initialize(int num_circumf_points, BoundaryCondition bc) {
  points.clear();
  normals.clear();
  weights.clear();
  curvatures.clear();
  holes.clear();

  int num_phi_points = num_circumf_points / 2;
  int total_points = num_circumf_points * num_phi_points;

  double phis[num_phi_points];
  double phi_weights[num_phi_points];
  double phi_start = 0.;
  double phi_end = M_PI;
  cgqf(num_phi_points, 1, 0.0, 0.0, phi_start, phi_end, phis, phi_weights);

  // Weight at north and south pole = 0?
  for (int i = 0; i < num_circumf_points; i++) {
    double theta = 2 * M_PI * i * (1.0 / num_circumf_points);
    for (int j = 0; j < num_phi_points; j++) {   // modify this for annulus proxy
      // double phi = M_PI * j * (1.0 / (num_phi_points)); //phis[j]; 
      double phi = phis[j]; 
      points.push_back(0.5 + r * sin(phi) * cos(theta));
      points.push_back(0.5 + r * sin(phi) * sin(theta));
      points.push_back(0.5 + r * cos(phi));
      normals.push_back(sin(phi) * cos(theta));
      normals.push_back(sin(phi) * sin(theta));
      normals.push_back(cos(phi));
      // weights.push_back(pow(r, 2) * sin(phi)*(M_PI/num_phi_points)//phi_weights[j]*
      weights.push_back(pow(r, 2) * sin(phi) * phi_weights[j]
                        *(( 2* M_PI )/num_circumf_points));
    }
  }


  if (bc == BoundaryCondition::DEFAULT) {
    boundary_values = ki_Mat(total_points, 1);
    apply_boundary_condition(0, total_points, ELECTRON_3D);
  } else {
    set_boundary_values_size(bc);
    apply_boundary_condition(0, total_points, bc);
  }
}

bool Sphere::is_in_domain(const PointVec& a) const {
  PointVec center(0.5, 0.5, 0.5);
  double eps = 1e-2;
  double dist = (center - a).norm();
  if (dist + eps > r) return false;
  return true;
}

}  // namespace kern_interp
