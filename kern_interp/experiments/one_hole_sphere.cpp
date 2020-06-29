// Copyright 2019 John Paul Ryan
#include <omp.h>
#include <string.h>
#include <fstream>
#include <memory>
#include <iostream>
#include <cmath>
#include <cassert>
#include "kern_interp/ki_mat.h"
#include "kern_interp/skel_factorization/skel_factorization.h"
#include "kern_interp/quadtree/quadtree.h"
#include "kern_interp/kernel/kernel.h"
#include "kern_interp/linear_solve.h"
#include "kern_interp/boundaries/sphere.h"
#include "kern_interp/boundaries/donut.h"

namespace kern_interp {


void run_one_hole_sphere() {

  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Sphere());

  Hole hole;
  hole.center = PointVec(0.0,0.2,0.7);
  hole.radius = 0.25;
  boundary->holes.push_back(hole);  
  Hole hole2;
  hole2.center = PointVec(1.0,0.8,0.3);
  hole2.radius = 0.25;
  boundary->holes.push_back(hole2);  
  boundary->initialize(96,  BoundaryCondition::STOKES_3D_MIX);

  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), 3, 3);
  std::vector<double>  domain_points;
  get_domain_points3d(10, &domain_points, 0.25, 1);

  Kernel kernel(3, 3, Kernel::Pde::STOKES, boundary.get(), domain_points);
  kernel.compute_diag_entries_3dstokes(boundary.get());

  ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
                                       1e-6, 8, domain_points);

// double err = stokes_err_3d(sol, domain_points, boundary.get(), 
//     hole.radius, STOKES_MIXER);
//   std::cout<<"err of "<<err<<std::endl;
  std::ofstream sol_out;
  sol_out.open("output/data/ie_solver_solution.txt");
  int points_index = 0;
  for (int i = 0; i < sol.height(); i += 3) {
    sol_out << domain_points[points_index] << "," <<
            domain_points[points_index+1] << ","<<
            domain_points[points_index+2]<<",";
    points_index += 3;
    sol_out << sol.get(i, 0) << "," << 
               sol.get(i + 1, 0)<<","<<
               sol.get(i + 2, 0)<< std::endl;
  }
  sol_out.close();
  // std::ofstream bound_out;
  // bound_out.open("output/data/ie_solver_boundary.txt");
  // for (int i = 0; i < boundary2d->points.size(); i += 2) {
  //   bound_out << boundary2d->points[i] << "," << boundary2d->points[i + 1]
  //             << std::endl;
  // }
  // bound_out.close();
 
 // conclusion of last timing test - lvl 2 has some boxes that take forever
}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
  openblas_set_num_threads(1);
  kern_interp::run_one_hole_sphere();
  return 0;
}

