// Copyright John Paul Ryan 2019
#include <omp.h>
#include <string.h>
#include <fstream>
#include <memory>
#include <iostream>
#include <cmath>
#include <cassert>
#include "kern_interp/ki_mat.h"
#include "kern_interp/boundaries/circle.h"
#include "kern_interp/boundaries/annulus.h"
#include "kern_interp/boundaries/donut.h"
#include "kern_interp/boundaries/ex1boundary.h"
#include "kern_interp/boundaries/ex2boundary.h"
#include "kern_interp/boundaries/ex3boundary.h"
#include "kern_interp/boundaries/cubic_spline.h"
#include "kern_interp/skel_factorization/skel_factorization.h"
#include "kern_interp/quadtree/quadtree.h"
#include "kern_interp/kernel/kernel.h"
#include "kern_interp/linear_solve.h"
#include "gtest/gtest.h"

namespace kern_interp {


void check_solve_err(const Kernel& kernel, Boundary* boundary) {
  int num_threads = 4;
  double id_tol = 1e-6;

  QuadTree quadtree;
  quadtree.initialize_tree(boundary, kernel.solution_dimension,
                           kernel.domain_dimension);

  SkelFactorization skel_factorization(id_tol, num_threads);

  double max = 0.;
  ki_Mat dense;

  if (boundary->holes.size() > 0
      && kernel.pde != Kernel::Pde::LAPLACE_NEUMANN) {
    ki_Mat U = initialize_U_mat(kernel.pde, boundary->holes, boundary->points);
    ki_Mat Psi = initialize_Psi_mat(kernel.pde, boundary->holes, *boundary);
    quadtree.U = U;
    quadtree.Psi = Psi;
    skel_factorization.skeletonize(kernel, &quadtree);

    ki_Mat mu, alpha;
    linear_solve(skel_factorization, quadtree, boundary->boundary_values, &mu,
                 &alpha);
    ki_Mat stacked(mu.height() + alpha.height(), 1);
    stacked.set_submatrix(0, mu.height(), 0, 1, mu);
    stacked.set_submatrix(mu.height(), stacked.height(), 0, 1, alpha);

    std::vector<int> all_dofs;
    for (int i = 0; i < kernel.solution_dimension * boundary->weights.size();
         i++) {
      all_dofs.push_back(i);
    }
    ki_Mat kern = kernel(all_dofs, all_dofs);
    int hole_factor = kernel.solution_dimension == 2 ? 3 : 1;
    int added = hole_factor * boundary->holes.size();
    dense = ki_Mat(all_dofs.size() + added, all_dofs.size() + added);
    ki_Mat ident(added, added);
    ident.eye(added);
    dense.set_submatrix(0, all_dofs.size(), 0, all_dofs.size(), kern);
    dense.set_submatrix(0, all_dofs.size(), all_dofs.size(), dense.width(), U);
    dense.set_submatrix(all_dofs.size(), dense.height(),
                        0, all_dofs.size(), Psi);
    dense.set_submatrix(all_dofs.size(), dense.height(),
                        all_dofs.size(), dense.width(), -ident);

    ki_Mat fzero_prime = dense * stacked;

    ki_Mat err1 = (fzero_prime(0, mu.height(), 0, 1)
                   - boundary->boundary_values);
    ki_Mat err2 = (fzero_prime(mu.height(), fzero_prime.height(), 0, 1));

    double err = sqrt(pow(err1.vec_two_norm(), 2) + pow(err1.vec_two_norm(), 2));
    EXPECT_LE(err / boundary->boundary_values.vec_two_norm(), 10 * id_tol);
  } else {
    skel_factorization.skeletonize(kernel, &quadtree);
    ki_Mat mu;
    linear_solve(skel_factorization, quadtree, boundary->boundary_values, &mu);
    std::vector<int> all_dofs;
    for (int i = 0; i < kernel.solution_dimension * boundary->weights.size();
         i++) {
      all_dofs.push_back(i);
    }

    ki_Mat err = (kernel(all_dofs, all_dofs) * mu) - boundary->boundary_values;

    EXPECT_LE(err.vec_two_norm() / boundary->boundary_values.vec_two_norm(),
              10 * id_tol);
  }
}


TEST(IeSolverTest, LaplaceCircleBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Circle());
  boundary->initialize(pow(2, 10),  BoundaryCondition::SINGLE_ELECTRON);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, LaplaceNeumannCircleBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Circle());
  boundary->initialize(pow(2, 10),  BoundaryCondition::ALL_ONES);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, StokesCircleBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Circle());
  boundary->initialize(pow(2, 10),  BoundaryCondition::TANGENT_VEC);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, LaplaceStarfishBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new CubicSpline());
  boundary->initialize(pow(2, 10),  BoundaryCondition::SINGLE_ELECTRON);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, LaplaceNeumannStarfishBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new CubicSpline());
  boundary->initialize(pow(2, 10),  BoundaryCondition::ALL_ONES);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, StokesStarfishBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new CubicSpline());
  boundary->initialize(pow(2, 10),  BoundaryCondition::TANGENT_VEC);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}



TEST(IeSolverTest, LaplaceAnnulusBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(pow(2, 10),  BoundaryCondition::SINGLE_ELECTRON);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, LaplaceNeumannAnnulusBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(pow(2, 10),  BoundaryCondition::DEFAULT);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, StokesAnnulusBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(pow(2, 10),  BoundaryCondition::TANGENT_VEC);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}

TEST(IeSolverTest, BigLaplaceCircleBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Circle());
  boundary->initialize(pow(2, 13),  BoundaryCondition::SINGLE_ELECTRON);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, BigLaplaceNeumannCircleBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Circle());
  boundary->initialize(pow(2, 13),  BoundaryCondition::ALL_ONES);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, BigStokesCircleBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Circle());
  boundary->initialize(pow(2, 13),  BoundaryCondition::TANGENT_VEC);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, BigLaplaceStarfishBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new CubicSpline());
  boundary->initialize(pow(2, 13),  BoundaryCondition::SINGLE_ELECTRON);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, BigLaplaceNeumannStarfishBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new CubicSpline());
  boundary->initialize(pow(2, 13),  BoundaryCondition::ALL_ONES);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, BigStokesStarfishBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new CubicSpline());
  boundary->initialize(pow(2, 13),  BoundaryCondition::TANGENT_VEC);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}



TEST(IeSolverTest, BigLaplaceAnnulusBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(pow(2, 13),  BoundaryCondition::SINGLE_ELECTRON);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, BigLaplaceNeumannAnnulusBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(pow(2, 13),  BoundaryCondition::DEFAULT);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


TEST(IeSolverTest, BigStokesAnnulusBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(pow(2, 13),  BoundaryCondition::TANGENT_VEC);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


double laplace_error(const ki_Mat& domain,
                     const std::vector<double>& domain_points,
                     Boundary * boundary) {
  double diff_norm = 0;
  double norm_of_true = 0;
  for (int i = 0; i < domain_points.size(); i += 2) {
    double x0 = domain_points[i];
    double x1 = domain_points[i + 1];
    Vec2 x(x0, x1);
    if (!boundary->is_in_domain(x)) {
      continue;
    }
    double potential = log(sqrt(pow(x0 + 3, 2) + pow(x1 + 2, 2))) / (2 * M_PI);

    diff_norm += pow(potential - domain.get(i / 2, 0), 2);
    norm_of_true += pow(potential, 2);
  }
  diff_norm = sqrt(diff_norm) / sqrt(norm_of_true);
  return diff_norm;
}


double laplace_neumann_error(const ki_Mat& domain,
                             const std::vector<double>& domain_points,
                             Boundary * boundary) {
  std::vector<double> res;
  std::vector<double> truth;

  for (int i = 0; i < domain_points.size(); i += 2) {
    double x0 = domain_points[i];
    double x1 = domain_points[i + 1];
    Vec2 x(x0, x1);
    if (boundary->is_in_domain(x)) {
      res.push_back(domain.get(i / 2, 0));
      double r = sqrt(pow(x0 - 0.5, 2) + pow(x1 - 0.5, 2));
      truth.push_back(log(r));
    }
  }
  double res_avg = 0.;
  double truth_avg = 0.;
  for (int i = 0; i < res.size(); i++) {
    res_avg += res[i];
    truth_avg += truth[i];
  }
  res_avg /= res.size();
  truth_avg /= truth.size();

  for (int i = 0; i < res.size(); i++) {
    res[i] -= res_avg;
    truth[i] -= truth_avg;
  }

  double diff_norm = 0.;
  double truth_norm = 0.;
  for (int i = 0; i < res.size(); i++) {
    diff_norm += pow(res[i] - truth[i], 2);
    truth_norm += pow(truth[i], 2);
  }
  return sqrt(diff_norm) / sqrt(truth_norm);
}


double stokes_error(const ki_Mat& domain,
                    const std::vector<double>& domain_points,
                    Boundary * boundary) {
  double diff_norm;
  double norm_of_true;
  for (int i = 0; i < domain_points.size(); i += 2) {
    double x0 = domain_points[i];
    double x1 = domain_points[i + 1];
    Vec2 x(x0, x1);
    if (!boundary->is_in_domain(x)) {
      continue;
    }
    Vec2 center(0.5, 0.5);
    Vec2 r = x - center;
    Vec2 sol = Vec2(domain.get(i, 0), domain.get(i + 1, 0));
    Vec2 truth = Vec2(-r.a[1], r.a[0]);
    switch (boundary->holes.size()) {
      case 0:  // circle
        truth = truth * (r.norm() / truth.norm()) * 4.;
        break;
      case 1:  // donut
      default:
        double p = (-1. / r.norm()) + 2 * r.norm();
        truth = truth * (1. / truth.norm()) * p;
        break;
    }
    diff_norm += pow((sol - truth).norm(), 2);
    norm_of_true += pow(truth.norm(), 2);
  }
  return sqrt(diff_norm) / sqrt(norm_of_true);
}


TEST(IeSolverTest, LaplaceCircleAnalyticAgreementElectron) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Circle());
  boundary->initialize(pow(2, 12),  BoundaryCondition::SINGLE_ELECTRON);
  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), 1, 2);
  std::vector<double> domain_points;
  get_domain_points(20, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min,
                    quadtree.max);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(), domain_points);
  ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
                                       1e-13, 4, domain_points);
  double err = laplace_error(sol, domain_points, boundary.get());
  EXPECT_LE(err, 10 * 1e-13);
}


TEST(IeSolverTest, LaplaceAnnulusAnalyticAgreementElectron) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(pow(2, 12),  BoundaryCondition::SINGLE_ELECTRON);
  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), 1, 2);
  std::vector<double> domain_points;
  get_domain_points(20, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min,
                    quadtree.max);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(), domain_points);
  ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
                                       1e-13,
                                       4, domain_points);
  double err = laplace_error(sol, domain_points, boundary.get());
  EXPECT_LE(err, 10 * 1e-13);
}


TEST(IeSolverTest, LaplaceNeumannDonutAnalyticAgreement) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Donut());
  boundary->initialize(pow(2, 12),  BoundaryCondition::DEFAULT);
  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), 1, 2);
  std::vector<double> domain_points;
  get_domain_points(20, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min,
                    quadtree.max);
  Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
                domain_points);
  ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
                                       1e-13,
                                       4, domain_points);
  double err = laplace_neumann_error(sol, domain_points, boundary.get());
  EXPECT_LE(err, 10 * 1e-13);
}


TEST(IeSolverTest, StokesCircleAnalyticAgreementTangent) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Circle());
  boundary->initialize(pow(2, 12),  BoundaryCondition::TANGENT_VEC);
  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), 2, 2);
  std::vector<double> domain_points;
  get_domain_points(20, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min,
                    quadtree.max);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(), domain_points);
  ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
                                       1e-13,
                                       4, domain_points);
  double err = stokes_error(sol, domain_points, boundary.get());
  EXPECT_LE(err, 10 * 1e-13);
}


TEST(IeSolverTest, StokesDonutAnalyticAgreementTangent) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Donut());
  boundary->initialize(pow(2, 12),  BoundaryCondition::TANGENT_VEC);
  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), 2, 2);
  std::vector<double> domain_points;
  get_domain_points(20, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min, quadtree.max);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(), domain_points);
  ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
                                       1e-13,
                                       4, domain_points);
  double err = stokes_error(sol, domain_points, boundary.get());
  EXPECT_LE(err, 10 * 1e-13);
}


TEST(IeSolverTest, Ex3UpdateLosesNoAcc) {
  double id_tol = 1e-6;
  Kernel::Pde pde = Kernel::Pde::STOKES;
  int num_boundary_points = pow(2, 12);
  int domain_size = 20;
  int domain_dimension = 2;
  int solution_dimension = 2;
  int fact_threads = 4;

  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Ex3Boundary());
  boundary->initialize(num_boundary_points,
                       BoundaryCondition::EX3B);
  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), solution_dimension,
                           domain_dimension);
  std::vector<double> domain_points;
  get_domain_points(domain_size, &domain_points, quadtree.min,
                    quadtree.max, quadtree.min,
                    quadtree.max);
  Kernel kernel(solution_dimension, domain_dimension,
                pde, boundary.get(), domain_points);

  for (int frame = 0; frame < 10; frame++) {
    double ang = (frame / 10.) * 2 * M_PI;

    boundary->perturbation_parameters[0] = ang;
    boundary->perturbation_parameters[1] = ang + M_PI;
    boundary->initialize(num_boundary_points, BoundaryCondition::EX3B);
    quadtree.perturb(*boundary.get());
    kernel.update_boundary(boundary.get());
    ki_Mat solution = boundary_integral_solve(kernel, *(boundary.get()),
                      &quadtree, id_tol, fact_threads, domain_points);

    QuadTree fresh;
    fresh.initialize_tree(boundary.get(), 2,  2);
    ki_Mat new_sol = boundary_integral_solve(kernel, *(boundary.get()), &fresh,
                     id_tol, fact_threads, domain_points);

    ASSERT_LE((new_sol - solution).vec_two_norm() / new_sol.vec_two_norm(),
              20 * id_tol);
  }
}


TEST(IeSolverTest, TreeCopyGivesSameAnswer) {
  double id_tol = 1e-6;
  Kernel::Pde pde = Kernel::Pde::STOKES;
  int num_boundary_points = pow(2, 12);
  int domain_size = 20;
  int domain_dimension = 2;
  int solution_dimension = 2;
  int fact_threads = 4;

  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Ex1Boundary());
  boundary->initialize(num_boundary_points,
                       BoundaryCondition::DEFAULT);
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

  QuadTree fresh;
  quadtree.copy_into(&fresh);
  ki_Mat new_sol = boundary_integral_solve(kernel, *(boundary.get()), &fresh,
                   id_tol,
                   fact_threads, domain_points);
  ASSERT_LE((new_sol - solution).vec_two_norm(), 1e-15);
}

}  // namespace kern_interp
