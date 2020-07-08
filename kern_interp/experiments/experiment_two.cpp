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
#include "kern_interp/boundaries/ex2boundary.h"

namespace kern_interp {


void run_experiment2() {
  double id_tol = 1e-6;
  Kernel::Pde pde = Kernel::Pde::STOKES;
  int num_boundary_points = pow(2, 12);
  // int domain_size = 100;
  int domain_size = 350;
  int domain_dimension = 2;
  int solution_dimension = 2;
  int fact_threads = 4;

  std::unique_ptr<Boundary> boundary
    =  std::unique_ptr<Boundary>(new Ex2Boundary());
  boundary->initialize(num_boundary_points, BoundaryCondition::DEFAULT);

  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), solution_dimension,
                           domain_dimension);
  double x_min = boundary->points[0], x_max = boundary->points[0],
         y_min = boundary->points[1], y_max = boundary->points[1];
  for (int i = 0; i < boundary->points.size(); i += 2) {
    x_min = std::min(x_min, boundary->points[i]);
    x_max = std::max(x_max, boundary->points[i]);
    y_min = std::min(y_min, boundary->points[i + 1]);
    y_max = std::max(y_max, boundary->points[i + 1]);
  }
  std::vector<double> domain_points;
  get_domain_points(domain_size, &domain_points, x_min, x_max, y_min,
                    y_max);

  Kernel kernel(solution_dimension, domain_dimension,
                pde, boundary.get(), domain_points);
  ki_Mat solution = boundary_integral_solve(kernel, *(boundary.get()),
                    &quadtree, id_tol, fact_threads, domain_points);

  // int FRAME_CAP = 100;
  // for (int frame = 0; frame < FRAME_CAP; frame++) {
  //   int rand_idx = floor(8 * (rand() / (0. + RAND_MAX)));
  //   boundary->perturbation_parameters[rand_idx] = 0.35
  //       + 0.3 * (rand() / (0. + RAND_MAX));
  //   boundary->initialize(num_boundary_points, BoundaryCondition::DEFAULT);
  //   quadtree.perturb(*boundary.get());
  //   kernel.update_data(boundary.get());
  //   solution = boundary_integral_solve(kernel,  *(boundary.get()), &quadtree,
  //                                      id_tol, fact_threads, domain_points);
  // }

  std::ofstream sol_out;
  sol_out.open("output/data/ie_solver_solution.txt");
  int points_index = 0;
  for (int i = 0; i < solution.height(); i += 2) {
    sol_out << domain_points[points_index] << "," <<
            domain_points[points_index + 1] << ",";
    points_index += 2;
    sol_out << solution.get(i, 0) << "," << solution.get(i + 1, 0)
            << std::endl;
  }
  sol_out.close();
  std::ofstream bound_out;
  bound_out.open("output/data/ie_solver_boundary.txt");
  for (int i = 0; i < boundary->points.size(); i += 2) {
    bound_out << boundary->points[i] << "," << boundary->points[i + 1]
              << std::endl;
  }
  bound_out.close();
}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
    openblas_set_num_threads(1);

  kern_interp::run_experiment2();
  return 0;
}
