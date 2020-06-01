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
#include "kern_interp/boundaries/annulus.h"

namespace kern_interp {


void run_xrr_test() {
  double id_tol = 1e-6;
  Kernel::Pde pde = Kernel::Pde::STOKES;
  int num_boundary_points = pow(2, 10);
  int domain_size = 100;
  int domain_dimension = 2;
  int solution_dimension = 2;
  int fact_threads = 1;
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(num_boundary_points, BoundaryCondition::TANGENT_VEC);
  double start = omp_get_wtime();

  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), solution_dimension,
                           domain_dimension);
  std::vector<double> domain_points;
  get_domain_points(domain_size, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min,
                    quadtree.max);
  Kernel kernel(solution_dimension, domain_dimension,
                pde, boundary.get(), domain_points);
  ki_Mat solution = boundary_integral_solve(kernel, *(boundary.get()),
                    &quadtree, id_tol, fact_threads, domain_points);
  double end = omp_get_wtime();
  std::cout << "Elapsed: " << (end - start) << std::endl;

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

  std::ofstream tree_out;
  tree_out.open("output/data/ie_solver_tree.txt");
  if (tree_out.is_open()) {
    for (QuadTreeLevel* level : quadtree.levels) {
      for (QuadTreeNode* node : level->nodes) {
        tree_out << node->corners[0] << "," << node->corners[1] << ","
                 << node->side_length << "," << "-1" << ","   <<
                 node->dof_lists.original_box.size() << std::endl;
      }
    }
    tree_out.close();
  }
}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
  kern_interp::run_xrr_test();
  return 0;
}
