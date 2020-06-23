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

namespace kern_interp {


void run_two_hole_sphere() {
  // double start = omp_get_wtime();
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Sphere());
  boundary->initialize(96,  BoundaryCondition::STOKES_3D);

  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), 3, 3);
  std::vector<double> old_domain_points, domain_points;
  get_domain_points3d(10, &old_domain_points, quadtree.min,
                    quadtree.max);

  for(int i=0; i<old_domain_points.size(); i+=3){
    if(boundary->is_in_domain(PointVec(old_domain_points[i],
                                      old_domain_points[i+1],
                                      old_domain_points[i+2] ))){
      domain_points.push_back(old_domain_points[i]);
      domain_points.push_back(old_domain_points[i+1]);
      domain_points.push_back(old_domain_points[i+2]);
    }

  }

  Kernel kernel(3, 3, Kernel::Pde::STOKES, boundary.get(), domain_points);
  double cstart=omp_get_wtime();
  kernel.compute_diag_entries_3dstokes(boundary.get());
  double cend=omp_get_wtime();
  std::cout<<"computer diag "<<(cend-cstart)<<std::endl;
  ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
                                       1e-6, 8, domain_points);

    std::ofstream sol_out;
  sol_out.open("ie_solver_solution.txt");
  int points_index = 0;
  for (int i = 0; i < sol.height(); i += 3) {
    sol_out << domain_points[points_index] << "," <<
            domain_points[points_index + 1] << ","<<domain_points[points_index+2]<<",";
    points_index += 3;
    sol_out << sol.get(i, 0) << "," << sol.get(i + 1, 0)<<","<<sol.get(i+2,0)
            << std::endl;
  }
  sol_out.close();
  // std::ofstream bound_out;
  // bound_out.open("output/data/ie_solver_boundary.txt");
  // for (int i = 0; i < boundary->points.size(); i += 2) {
  //   bound_out << boundary->points[i] << "," << boundary->points[i + 1]
  //             << std::endl;
  // }
  // bound_out.close();

 // conclusion of last timing test - lvl 2 has some boxes that take forever
}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
  openblas_set_num_threads(1);
  kern_interp::run_two_hole_sphere();
  return 0;
}
