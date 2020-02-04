// Copyright 2019 John Paul Ryan
#include <omp.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include <vector>
#include "kern_interp/boundaries/ex3boundary.h"

namespace kern_interp {


void Ex3Boundary::get_spline_points(std::vector<double>* x0_spline_points,
                                    std::vector<double>* x1_spline_points) {
  for (int i = 0; i < 20; i++) {
    double ang = 2 * M_PI * (i / (20.));

    double x =  0.375 * cos(ang) * (sin(5 * ang) + 4);
    double y =  0.375 * sin(ang) * (sin(5 * ang) + 4);

    x0_spline_points->push_back(0.5 + x);
    x1_spline_points->push_back(0.5 + y);
  }
}


void Ex3Boundary::get_star_spline_points(double x, double y,
    std::vector<double>* x0_points, std::vector<double>* x1_points) {
  for (int i = 0; i < 20; i++) {
    double ang = 2 * M_PI * (i / (20.));

    double xc =  0.06 * cos(ang) * (sin(5 * ang) + 4);
    double yc =  0.06 * sin(ang) * (sin(5 * ang) + 4);

    x0_points->push_back(x + xc);
    x1_points->push_back(y + yc);
  }
}


void Ex3Boundary::initialize(int N, BoundaryCondition bc) {
  points.clear();
  normals.clear();
  weights.clear();
  curvatures.clear();
  holes.clear();

  int OUTER_NUM_SPLINE_POINTS = 20;
  int STAR_NUM_SPLINE_POINTS = 20;

  int OUTER_NODES_PER_SPLINE = (2 * N / 3) / OUTER_NUM_SPLINE_POINTS;
  int STAR_NODES_PER_SPLINE = (N / 6) / STAR_NUM_SPLINE_POINTS;

  num_outer_nodes = OUTER_NUM_SPLINE_POINTS * OUTER_NODES_PER_SPLINE;

  if (perturbation_parameters.size() == 0) {
    perturbation_parameters.push_back(0);
    perturbation_parameters.push_back(M_PI);
  }
  Hole star1, star2;

  double ang1 = perturbation_parameters[0];
  double x1 =  0.2 * cos(ang1) * (sin(5 * ang1) + 4);
  double y1 =  0.2 * sin(ang1) * (sin(5 * ang1) + 4);
  star1.center = Vec2(0.5 + x1, 0.5 + y1);
  star1.radius = 0.3;
  star1.num_nodes =  STAR_NUM_SPLINE_POINTS * STAR_NODES_PER_SPLINE;
  holes.push_back(star1);
  double ang2 = perturbation_parameters[1];
  double x2 =  0.2 * cos(ang2) * (sin(5 * ang2) + 4);
  double y2 =  0.2 * sin(ang2) * (sin(5 * ang2) + 4);
  star2.center = Vec2(0.5 + x2, 0.5 + y2);
  star2.radius = 0.3;
  star2.num_nodes =  STAR_NUM_SPLINE_POINTS * STAR_NODES_PER_SPLINE;
  holes.push_back(star2);


  std::vector<double> outer_x0_spline_points, outer_x1_spline_points;
  std::vector<std::vector<double>> outer_x0_cubics, outer_x1_cubics;

  get_spline_points(&outer_x0_spline_points, &outer_x1_spline_points);
  get_cubics(outer_x0_spline_points, outer_x1_spline_points, &outer_x0_cubics,
             &outer_x1_cubics);

  interpolate(false,  OUTER_NODES_PER_SPLINE, outer_x0_cubics, outer_x1_cubics);

  std::vector<double> star_x0_points, star_x1_points;
  get_star_spline_points(star1.center.a[0], star1.center.a[1], &star_x0_points,
                         &star_x1_points);

  std::vector<std::vector<double>> star_x0_cubics, star_x1_cubics;
  get_cubics(star_x0_points, star_x1_points,
             &star_x0_cubics, &star_x1_cubics);

  interpolate(true, STAR_NODES_PER_SPLINE, star_x0_cubics, star_x1_cubics);

  std::vector<double> star2_x0_points, star2_x1_points;
  get_star_spline_points(star2.center.a[0], star2.center.a[1], &star2_x0_points,
                         &star2_x1_points);

  std::vector<std::vector<double>> star2_x0_cubics, star2_x1_cubics;
  get_cubics(star2_x0_points, star2_x1_points,
             &star2_x0_cubics, &star2_x1_cubics);

  interpolate(true, STAR_NODES_PER_SPLINE, star2_x0_cubics, star2_x1_cubics);

  if (bc == BoundaryCondition::EX3A) {
    boundary_values = ki_Mat(weights.size(), 1);
    int b1 =  OUTER_NUM_SPLINE_POINTS * OUTER_NODES_PER_SPLINE;
    int b2 = b1 + STAR_NUM_SPLINE_POINTS * STAR_NODES_PER_SPLINE;
    int b3 = b2 + STAR_NUM_SPLINE_POINTS * STAR_NODES_PER_SPLINE;
    apply_boundary_condition(0, b1, BoundaryCondition::ALL_ZEROS);
    apply_boundary_condition(b1, b2, BoundaryCondition::ALL_ONES);
    apply_boundary_condition(b2, b3, BoundaryCondition::ALL_NEG_ONES);
  } else if (bc == BoundaryCondition::EX3B) {
    boundary_values = ki_Mat(2 * weights.size(), 1);
    int b1 =  OUTER_NUM_SPLINE_POINTS * OUTER_NODES_PER_SPLINE;
    int b2 = b1 + STAR_NUM_SPLINE_POINTS * STAR_NODES_PER_SPLINE;
    int b3 = b2 + STAR_NUM_SPLINE_POINTS * STAR_NODES_PER_SPLINE;
    apply_boundary_condition(0, b1, BoundaryCondition::HORIZONTAL_VEC);
    apply_boundary_condition(b1, b2, BoundaryCondition::REVERSE_NORMAL_VEC);
    apply_boundary_condition(b2, b3, BoundaryCondition::NORMAL_VEC);
  } else if (bc == BoundaryCondition::DEFAULT) {
    std::cout << "No default boundary condition for ex3boundary" << std::endl;
    exit(0);
  } else {
    set_boundary_values_size(bc);
    apply_boundary_condition(0, weights.size(), bc);
  }
}


}  // namespace kern_interp
