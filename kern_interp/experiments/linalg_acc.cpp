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
#include "kern_interp/boundaries/ex3boundary.h"

namespace kern_interp {


void run_linalg_acc() {
  Kernel::Pde pde = Kernel::Pde::STOKES;
  int num_boundary_points = pow(2, 13);
  int domain_size = 100;
  int domain_dimension = 2;
  int solution_dimension = 2;
  int fact_threads = 4;
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Ex3Boundary());
  boundary->initialize(num_boundary_points, BoundaryCondition::EX3B);

  Kernel kernel(solution_dimension, domain_dimension,
                pde, boundary.get(), std::vector<double>());

  for (int epspow = -3; epspow >= -9; epspow--) {
    double id_tol =  pow(10, epspow);
    std::cout << "id_tol " << id_tol << " err: " << solve_err(kernel,
              boundary.get(), id_tol) << std::endl;
  }
}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
    openblas_set_num_threads(1);

  kern_interp::run_linalg_acc();
  return 0;
}

