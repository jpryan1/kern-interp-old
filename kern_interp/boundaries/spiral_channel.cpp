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


  int NUM_OF_TURNS = num_turns;
  int num_semi_circles = NUM_OF_TURNS * 2;

  // First semicircle
  for (int i = 3; i <= 5; i++) {
    double ang = 2 * M_PI * (i / 8.0);
    double x = 0.5 * cos(ang);
    double y = 0.5 * sin(ang) - 0.5;
    x0_points->push_back(x);
    x1_points->push_back(y);
  }

  // Quartercircle
  for (int i = 9; i <= 11; i++) {
    double ang = 2 * M_PI * (i / 12.0);
    double x =  cos(ang);
    double y =  sin(ang);
    x0_points->push_back(x);
    x1_points->push_back(y);
  }

  int div_idx = 12;

  for (int semi = 0; semi < num_semi_circles; semi++) {
    for (int t = semi * (div_idx / 2); t < (semi + 1) * (div_idx / 2); t++) {
      double theta = (2 * M_PI * t) / div_idx;
      double r = 1 + (theta / M_PI);
      x0_points->push_back(r * cos(theta));
      x1_points->push_back(r * sin(theta));
    }
    div_idx += 6;
  }


  // SEMICIRCLE
  for (int i = 0; i <= 4; i++) {
    double ang = 2 * M_PI * (i / 8.0);
    double x = 0.5 * cos(ang) + 0.5 + (2 * NUM_OF_TURNS);
    double y = 0.5 * sin(ang);
    x0_points->push_back(x);
    x1_points->push_back(y);
  }

  for (int semi = num_semi_circles - 1; semi >= 0; semi--) {
    div_idx -= 6;
    for (int t = ((semi + 1) * (div_idx / 2)) - 1;
         t >= semi * (div_idx / 2); t--) {
      double theta = (2 * M_PI * t) / div_idx;
      double r = (theta / M_PI);
      x0_points->push_back(r * cos(theta));
      x1_points->push_back(r * sin(theta));
    }
  }


  double xmin = (*x0_points)[0];
  double xmax = (*x0_points)[0];
  double ymin = (*x1_points)[0];
  double ymax = (*x1_points)[0];
  for (int i = 0; i < x0_points->size(); i++) {
    xmin = std::min(xmin, (*x0_points)[i]);
    xmax = std::max(xmax, (*x0_points)[i]);
    ymin = std::min(ymin, (*x1_points)[i]);
    ymax = std::max(ymax, (*x1_points)[i]);
  }

  for (int i = 0; i < x0_points->size(); i++) {
    (*x0_points)[i]  = 1000 * ((*x0_points)[i] - xmin) / (xmax - xmin);
    (*x1_points)[i]  = 1000 * ((*x1_points)[i] - ymin) / (ymax - ymin);
  }

  hole_center = PointVec((-xmin)/(xmax - xmin),( -0.5-ymin) / (ymax - ymin));
  hole_rad = 0.25 / (std::max(ymax - ymin, xmax - xmin));

  // std::cout << "total " << x0_points->size() << std::endl;
  // for (int i = 0; i < x0_points->size(); i++) {
  //   std::cout << (*x0_points)[i] << " " << (*x1_points)[i] << std::endl;
  // }
}



void SpiralChannel::initialize(int N, BoundaryCondition bc) {
  points.clear();
  normals.clear();
  weights.clear();
  curvatures.clear();
  holes.clear();

  // int num_hole_pts = 100;

  std::vector<double> outer_x0_spline_points, outer_x1_spline_points;
  get_spline_points(&outer_x0_spline_points, &outer_x1_spline_points);

  int OUTER_NUM_SPLINE_POINTS = outer_x0_spline_points.size();
  int OUTER_NODES_PER_SPLINE = N / OUTER_NUM_SPLINE_POINTS;
  num_outer_nodes = (OUTER_NUM_SPLINE_POINTS * OUTER_NODES_PER_SPLINE);

  std::vector<std::vector<double>> outer_x0_cubics, outer_x1_cubics;
  get_cubics(outer_x0_spline_points, outer_x1_spline_points,
             &outer_x0_cubics, &outer_x1_cubics);

  interpolate(false, OUTER_NODES_PER_SPLINE, outer_x0_cubics, outer_x1_cubics);


  // Hole circle;
  // circle.radius = hole_rad;
  // circle.center = hole_center;
  // circle.num_nodes = num_hole_pts;
  // holes.push_back(circle);

  // for (int i = 0; i < num_hole_pts; i++) {
  //   double ang = (2.0 * M_PI * i) / num_hole_pts;
  //   points.push_back(circle.center.a[0] + circle.radius * cos(ang));
  //   points.push_back(circle.center.a[1] + circle.radius * sin(ang));
  //   normals.push_back(-cos(ang));
  //   normals.push_back(-sin(ang));
  //   curvatures.push_back(-1.0 / circle.radius);
  //   weights.push_back(2.0 * M_PI * circle.radius / num_hole_pts);
  // }


  if (bc == BoundaryCondition::DEFAULT) {
    boundary_values = ki_Mat(weights.size() * 2, 1);
    int b1 = OUTER_NUM_SPLINE_POINTS * OUTER_NODES_PER_SPLINE;
    // int b2 = b1 + num_hole_pts;
    apply_boundary_condition(0, b1, TANGENT_VEC);
    // apply_boundary_condition(b1, b2, TANGENT_VEC);
  } else {
    set_boundary_values_size(bc);
    apply_boundary_condition(0, weights.size(), bc);
  }
}


}  // namespace kern_interp
