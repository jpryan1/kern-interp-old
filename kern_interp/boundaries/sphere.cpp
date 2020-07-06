// Copyright 2019 John Paul Ryan
#include <cmath>
#include <iostream>
#include "kern_interp/boundaries/sphere.h"
#include "kern_interp/legendre.h"

namespace kern_interp {

void Sphere::initialize(int sz_param, BoundaryCondition bc) {
  points.clear();
  normals.clear();
  weights.clear();
  curvatures.clear();

  if (holes.size() == 0) {
    Hole hole;
    hole.center = PointVec(0.3, 0.5, 0.5);
    hole.radius = 0.1;
    holes.push_back(hole);
    hole.center = PointVec(0.7, 0.5, 0.5);
    hole.radius = 0.1;
    holes.push_back(hole);
  }

  int num_holes = holes.size();

  double x = sqrt(sz_param * 2.0 / (9.0 + num_holes));
  int hole_circumf_points = (int) x;
  int num_circumf_points = 3 * hole_circumf_points;

  int num_phi_points = num_circumf_points / 2;
  num_outer_nodes = num_circumf_points * num_phi_points;

  int hole_phi_points = hole_circumf_points / 2;


  int each_hole_points = hole_circumf_points * hole_phi_points;
  for (int i = 0; i < holes.size(); i++) {
    holes[i].num_nodes = each_hole_points;
  }

  int total_points = num_circumf_points * num_phi_points + 
                     (num_holes * each_hole_points);

  double phis[num_phi_points];
  double phi_weights[num_phi_points];
  double phi_start = 0.;
  double phi_end = M_PI;
  cgqf(num_phi_points, 1, 0.0, 0.0, phi_start, phi_end, phis, phi_weights);
  for (int i = 0; i < num_circumf_points; i++) {
    double theta = 2 * M_PI * i * (1.0 / num_circumf_points);
    for (int j = 0; j < num_phi_points; j++) {
      double phi = phis[j];
      points.push_back(0.5 + r * sin(phi) * cos(theta));
      points.push_back(0.5 + r * sin(phi) * sin(theta));
      points.push_back(0.5 + r * cos(phi));
      normals.push_back(sin(phi) * cos(theta));
      normals.push_back(sin(phi) * sin(theta));
      normals.push_back(cos(phi));
      weights.push_back(pow(r, 2) * sin(phi) * phi_weights[j]
                        * ((2 * M_PI) / num_circumf_points));
    }
  }
  double hole_phis[hole_phi_points];
  double hole_phi_weights[hole_phi_points];
  cgqf(hole_phi_points, 1, 0.0, 0.0, phi_start, phi_end, hole_phis,
       hole_phi_weights);
  for (Hole hole : holes) {
    for (int i = 0; i < hole_circumf_points; i++) {
      double theta = 2 * M_PI * i * (1.0 / hole_circumf_points);
      for (int j = 0; j < hole_phi_points; j++) {
        double phi = hole_phis[j];
        points.push_back(hole.center.a[0] + hole.radius * sin(phi) * cos(theta));
        points.push_back(hole.center.a[1] + hole.radius * sin(phi) * sin(theta));
        points.push_back(hole.center.a[2] + hole.radius * cos(phi));
        normals.push_back(-sin(phi) * cos(theta));
        normals.push_back(-sin(phi) * sin(theta));
        normals.push_back(-cos(phi));
        weights.push_back(pow(hole.radius, 2) * sin(phi) * hole_phi_weights[j]
                          * ((2 * M_PI) / hole_circumf_points));
      }
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
  double eps = 0.1;
  double dist = (center - a).norm();
  if (dist + eps > r) return false;
  for (Hole hole : holes) {
    dist = (hole.center - a).norm();
    if (dist - eps < hole.radius) return false;
  }
  return true;
}

}  // namespace kern_interp
