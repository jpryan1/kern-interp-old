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


void run_3d() {
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
                                       1e-3, 4, domain_points);


}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
  kern_interp::run_3d();
  return 0;
}

