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
#include "kern_interp/boundaries/uniformdist.h"
#include "kern_interp/boundaries/ex1boundary.h"
#include "kern_interp/boundaries/circle.h"
#include "kern_interp/boundaries/donut.h"

namespace kern_interp {


void run_covar_exp() {
  double id_tol = 1e-6;
  Kernel::Pde pde = Kernel::Pde::GAUSS;
  int num_boundary_points = pow(2, 13);
  int solution_dimension = 1;
  int domain_dimension = 2;
  int fact_threads = 4;
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new UniformDist());
  boundary->initialize(num_boundary_points, BoundaryCondition::DEFAULT);
  num_boundary_points = boundary->points.size() / domain_dimension;

  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), solution_dimension,
                           domain_dimension);
  Kernel kernel(solution_dimension, domain_dimension,
                pde, boundary.get(), std::vector<double>());
  SkelFactorization skel_factorization(id_tol, fact_threads);
  skel_factorization.skeletonize(kernel, &quadtree);

  ki_Mat mu(num_boundary_points, 1);
  linear_solve(skel_factorization, quadtree, boundary->boundary_values, &mu,
               nullptr);

  std::vector<int> all_inds;
  for (int i = 0; i < num_boundary_points; i++) {
    all_inds.push_back(i);
  }

  ki_Mat dense = kernel(all_inds, all_inds);

  ki_Mat result = dense * mu;
  ki_Mat err = result - boundary->boundary_values;
  // std::cout << "Err " << (err.vec_two_norm() /
  //                         boundary->boundary_values.vec_two_norm())
  //           << std::endl;
}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
  openblas_set_num_threads(1);

  kern_interp::run_covar_exp();
  return 0;
}

