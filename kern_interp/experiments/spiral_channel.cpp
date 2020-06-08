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
#include "kern_interp/boundaries/spiral_channel.h"

namespace kern_interp {


void run_spiral_channel(int num_boundary_points) {
  double id_tol = 1e-6;
  Kernel::Pde pde = Kernel::Pde::STOKES;
  // int num_boundary_points = pow(2, 13);
  int domain_size = 100;
  int domain_dimension = 2;
  int solution_dimension = 2;
  int fact_threads = 4;
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new SpiralChannel());
  boundary->initialize(num_boundary_points, BoundaryCondition::DEFAULT);
  // double start = omp_get_wtime();

  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), solution_dimension,
                           domain_dimension);
  std::vector<double> domain_points;
  get_domain_points(domain_size, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min,
                    quadtree.max);
  Kernel kernel(solution_dimension, domain_dimension,
                pde, boundary.get(), domain_points);

  // ki_Mat solution = boundary_integral_solve(kernel, *(boundary.get()),
  //                   &quadtree, id_tol, fact_threads, domain_points);
  // double end = omp_get_wtime();
  // std::cout << "Elapsed: " << (end - start) << std::endl;

  SkelFactorization skel_factorization(id_tol, fact_threads);

  ki_Mat f = boundary->boundary_values;
  ki_Mat U = initialize_U_mat(kernel.pde, boundary->holes, boundary->points);
  ki_Mat Psi = initialize_Psi_mat(kernel.pde, boundary->holes, *(boundary.get()));

  ki_Mat solution((kernel.domain_points.size() / 2)*
                  kernel.solution_dimension, 1);
  quadtree.U = U;
  quadtree.Psi = Psi;

  double start = omp_get_wtime();
  skel_factorization.skeletonize(kernel, &quadtree);
  double end = omp_get_wtime();
  std::cout << "skel " << (end - start) << std::endl;
  ki_Mat mu, alpha;
  mu = ki_Mat(f.height(), 1);
  alpha = ki_Mat(quadtree.U.width(), 1);

  start = omp_get_wtime();
  skel_factorization.multiply_connected_solve(quadtree, &mu, &alpha, f);
  end = omp_get_wtime();
  std::cout << "solve" << (end - start) << std::endl;
  // ki_Mat K_domain = kernel.forward();

  // ki_Mat U_forward = initialize_U_mat(kernel.pde, boundary->holes,
  //                                     kernel.domain_points);

  // solution = (K_domain * mu) + (U_forward * alpha);
  // for (int i = 0; i < kernel.domain_points.size(); i += 2) {
  //   PointVec point = PointVec(kernel.domain_points[i], kernel.domain_points[i + 1]);
  //   if (!boundary->is_in_domain(point)) {
  //     solution.set(i, 0, 0.);
  //     solution.set(i + 1, 0, 0.);
  //   }
  // }
  // std::ofstream sol_out;
  // sol_out.open("output/data/ie_solver_solution.txt");
  // int points_index = 0;
  // for (int i = 0; i < solution.height(); i += 2) {
  //   sol_out << domain_points[points_index] << "," <<
  //           domain_points[points_index + 1] << ",";
  //   points_index += 2;
  //   sol_out << solution.get(i, 0) << "," << solution.get(i + 1, 0)
  //           << std::endl;
  // }
  // sol_out.close();
  // std::ofstream bound_out;
  // bound_out.open("output/data/ie_solver_boundary.txt");
  // for (int i = 0; i < boundary->points.size(); i += 2) {
  //   bound_out << boundary->points[i] << "," << boundary->points[i + 1]
  //             << std::endl;
  // }
  // bound_out.close();
}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
  for (int power = 1; power <= 10; power++) {
    std::cout << "pts 1k* to pow " << power << std::endl;
    for (int trial = 0; trial < 3; trial++) {
      kern_interp::run_spiral_channel(1000 * power);
    }
  }
  return 0;
}

