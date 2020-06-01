// Copyright 2019 John Paul Ryan
#include <omp.h>
#include <string.h>
#include <fstream>
#include <memory>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <cassert>
#include "kern_interp/ki_mat.h"
#include "kern_interp/skel_factorization/skel_factorization.h"
#include "kern_interp/quadtree/quadtree.h"
#include "kern_interp/kernel/kernel.h"
#include "kern_interp/linear_solve.h"
#include "kern_interp/boundaries/donut.h"

namespace kern_interp {


void run_pde_acc() {
  double id_tol = 1e-13;
  Kernel::Pde pde = Kernel::Pde::STOKES;
  int num_boundary_points = pow(2, 11);
  int domain_size = 200;
  int domain_dimension = 2;
  int solution_dimension = 2;
  int fact_threads = 4;
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Donut());
  boundary->initialize(num_boundary_points, BoundaryCondition::DEFAULT);


  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), solution_dimension,
                           domain_dimension);
  std::vector<double> domain_points;
  get_domain_points(domain_size, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min, quadtree.max);

  ki_Mat stokes_sol = stokes_true_sol(domain_points, boundary.get(), -(1.0 / 3.0),
                                      (10.0 / 3.0));


  Kernel kernel(solution_dimension, domain_dimension,
                pde, boundary.get(), domain_points);
  ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()),
                                       &quadtree, id_tol, fact_threads,
                                       domain_points);

  ki_Mat error = stokes_sol - sol;

  // std::vector<double> err_norms;
  // for (int i = 0; i < error.height(); i += 2) {
  //   double err_mag = sqrt(pow(error.get(i, 0), 2) + pow(error.get(i + 1, 0), 2));
  //   if (err_mag != 0) {
  //     double truth_mag = sqrt(pow(stokes_sol.get(i, 0), 2)
  //                             + pow(stokes_sol.get(i + 1, 0), 2));
  //     err_norms.push_back(err_mag / truth_mag);
  //   }
  // }
  // std::sort(err_norms.begin(), err_norms.end());
  // int tenth = err_norms.size() / 10;
  // int med = err_norms.size() / 2;
  // int ninetieth = 9 * err_norms.size() / 10;
  // std::cout << "percentiles " << err_norms[tenth] << " "
  //           << err_norms[med] << " " << err_norms[ninetieth] << std::endl;





  std::ofstream sol_out;
  sol_out.open("output/data/ie_solver_solution.txt");
  int points_index = 0;
  for (int i = 0; i < error.height(); i += 2) {
    sol_out << domain_points[points_index] << "," <<
            domain_points[points_index + 1] << ",";
    points_index += 2;
    sol_out << error.get(i, 0) << "," << error.get(i + 1, 0)
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
  kern_interp::run_pde_acc();
  return 0;
}

