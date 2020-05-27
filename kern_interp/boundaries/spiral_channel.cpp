// Copyright 2019 John Paul Ryan
#include <omp.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>
#include "kern_interp/boundaries/spiral_channel.h"

namespace kern_interp {


void SpiralChannel::get_spline_points(std::vector<double>* x0_points,
                                      std::vector<double>* x1_points) {

  // CAP should be multiple of 5

  int CAP = 45;

  // First semicircle
  for (int i = 6; i <= 10; i++) {
    double ang = 2 * M_PI * (i / 16.0);
    double x = 0.5 * cos(ang);
    double y = 0.5 * sin(ang) - 0.5;
    x0_points->push_back(x);
    x1_points->push_back(y);
  }
  // Quartercircle
  for (int i = 6; i <= 7; i++) {
    double ang = 2 * M_PI * (i / 8.0);
    double x =  cos(ang);
    double y =  sin(ang);
    x0_points->push_back(x);
    x1_points->push_back(y);
  }
  // outer spiral
  for (int t = 0; t <= CAP; t++) {
    double theta = 2 * M_PI * (t / 20.0);
    double r = 1 + (theta / M_PI);
    x0_points->push_back(r * cos(theta));
    x1_points->push_back(r * sin(theta));
  }
  // second semicircle
  for (int i = 6; i <= 10; i++) {
    double ang = 2 * M_PI * (i / 16.0);
    double x = 0.5 * cos(ang);
    double y = 0.5 * sin(ang) + 5;
    x0_points->push_back(x);
    x1_points->push_back(y);
  }
  // inner spiral
  for (int t = CAP; t >= 0; t--) {
    double theta = 2 * M_PI * (t / 20.0);
    double r = (theta / M_PI);
    x0_points->push_back(r * cos(theta));
    x1_points->push_back(r * sin(theta));
  }
  std::cout << "total " << x0_points->size() << std::endl;
  for (int i = 0; i < x0_points->size(); i++) {
    std::cout << (*x0_points)[i] << " " << (*x1_points)[i] << std::endl;
  }
}



void SpiralChannel::initialize(int N, BoundaryCondition bc) {
  points.clear();
  normals.clear();
  weights.clear();
  curvatures.clear();
  holes.clear();

  std::vector<double> outer_x0_spline_points, outer_x1_spline_points;
  get_spline_points(&outer_x0_spline_points, &outer_x1_spline_points);

  int OUTER_NUM_SPLINE_POINTS = outer_x0_spline_points.size();
  int OUTER_NODES_PER_SPLINE = N / OUTER_NUM_SPLINE_POINTS;
  num_outer_nodes = OUTER_NUM_SPLINE_POINTS * OUTER_NODES_PER_SPLINE;

  std::vector<std::vector<double>> outer_x0_cubics, outer_x1_cubics;
  get_cubics(outer_x0_spline_points, outer_x1_spline_points,
             &outer_x0_cubics, &outer_x1_cubics);

  interpolate(false, OUTER_NODES_PER_SPLINE, outer_x0_cubics, outer_x1_cubics);

  if (bc == BoundaryCondition::DEFAULT) {
    boundary_values = ki_Mat(weights.size() * 2, 1);
    int b1 = OUTER_NUM_SPLINE_POINTS * OUTER_NODES_PER_SPLINE;
    apply_boundary_condition(0, b1, TANGENT_VEC);
  } else {
    set_boundary_values_size(bc);
    apply_boundary_condition(0, weights.size(), bc);
  }
}


}  // namespace kern_interp
