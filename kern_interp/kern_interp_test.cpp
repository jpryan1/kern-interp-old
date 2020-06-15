// Copyright John Paul Ryan 2019
#include <omp.h>
#include <string.h>
#include <fstream>
#include <memory>
#include <unordered_map>
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
#include "kern_interp/boundaries/sphere.h"
#include "kern_interp/boundaries/cubic_spline.h"
#include "kern_interp/skel_factorization/skel_factorization.h"
#include "kern_interp/quadtree/quadtree.h"
#include "kern_interp/kernel/kernel.h"
#include "kern_interp/linear_solve.h"
#include "gtest/gtest.h"

namespace kern_interp {


void check_solve_err(const Kernel& kernel, Boundary* boundary) {
  double id_tol = 1e-6;

  EXPECT_LE(solve_err(kernel, boundary, id_tol), 10 * id_tol);
}


// TEST(IeSolverTest, LaplaceCircleBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Circle());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::SINGLE_ELECTRON);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, LaplaceNeumannCircleBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Circle());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::ALL_ONES);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, StokesCircleBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Circle());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::TANGENT_VEC);
//   Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, LaplaceStarfishBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new CubicSpline());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::SINGLE_ELECTRON);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, LaplaceNeumannStarfishBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new CubicSpline());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::ALL_ONES);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, StokesStarfishBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new CubicSpline());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::TANGENT_VEC);
//   Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }



// TEST(IeSolverTest, LaplaceAnnulusBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Annulus());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::SINGLE_ELECTRON);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, LaplaceNeumannAnnulusBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Annulus());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::DEFAULT);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, StokesAnnulusBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Annulus());
//   boundary->initialize(pow(2, 10),  BoundaryCondition::TANGENT_VEC);
//   Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }

// TEST(IeSolverTest, BigLaplaceCircleBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Circle());
//   boundary->initialize(pow(2, 13),  BoundaryCondition::SINGLE_ELECTRON);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, BigLaplaceNeumannCircleBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Circle());
//   boundary->initialize(pow(2, 13),  BoundaryCondition::ALL_ONES);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, BigStokesCircleBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Circle());
//   boundary->initialize(pow(2, 13),  BoundaryCondition::TANGENT_VEC);
//   Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, BigLaplaceStarfishBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new CubicSpline());
//   boundary->initialize(pow(2, 13),  BoundaryCondition::SINGLE_ELECTRON);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, BigLaplaceNeumannStarfishBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new CubicSpline());
//   boundary->initialize(pow(2, 13),  BoundaryCondition::ALL_ONES);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, BigStokesStarfishBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new CubicSpline());
//   boundary->initialize(pow(2, 13),  BoundaryCondition::TANGENT_VEC);
//   Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }



// TEST(IeSolverTest, BigLaplaceAnnulusBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Annulus());
//   boundary->initialize(pow(2, 13),  BoundaryCondition::SINGLE_ELECTRON);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


// TEST(IeSolverTest, BigLaplaceNeumannAnnulusBackwardError) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Annulus());
//   boundary->initialize(pow(2, 13),  BoundaryCondition::DEFAULT);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
//                 std::vector <double>());
//   check_solve_err(kernel, boundary.get());
// }


TEST(IeSolverTest, BigStokesAnnulusBackwardError) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Annulus());
  boundary->initialize(pow(2, 13),  BoundaryCondition::TANGENT_VEC);
  Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(),
                std::vector <double>());
  check_solve_err(kernel, boundary.get());
}


// double laplace_error(const ki_Mat& domain,
//                      const std::vector<double>& domain_points,
//                      Boundary * boundary) {
//   double diff_norm = 0;
//   double norm_of_true = 0;
//   for (int i = 0; i < domain_points.size(); i += 2) {
//     double x0 = domain_points[i];
//     double x1 = domain_points[i + 1];
//     PointVec x(x0, x1);
//     if (!boundary->is_in_domain(x)) {
//       continue;
//     }
//     double potential = log(sqrt(pow(x0 + 3, 2) + pow(x1 + 2, 2))) / (2 * M_PI);

//     diff_norm += pow(potential - domain.get(i / 2, 0), 2);
//     norm_of_true += pow(potential, 2);
//   }
//   diff_norm = sqrt(diff_norm) / sqrt(norm_of_true);
//   return diff_norm;
// }


double laplace_error3d(const ki_Mat& domain,
                     const std::vector<double>& domain_points,
                     Boundary * boundary) {
  double diff_norm = 0;
  double norm_of_true = 0;
  int in_domain=0;
  for (int i = 0; i < domain_points.size(); i += 3) {
    double x0 = domain_points[i];
    double x1 = domain_points[i + 1];
    double x2 = domain_points[i + 2];
    PointVec x(x0, x1, x2);
    if (!boundary->is_in_domain(x)) {
      continue;
    }
in_domain++;
    double potential = -1.0/(4.0*M_PI*sqrt(pow(x0 + 3, 2) + pow(x1 + 2, 2) + pow(x2 + 2, 2)));

    // std::cout<<x0<<" "<<x1<<" "<<x2<<std::endl;
    if(i<10) std::cout<<domain.get(i/3,0)<<" vs "<<potential<<std::endl;
    diff_norm += pow(potential - domain.get(i / 3, 0), 2);
    norm_of_true += pow(potential, 2);
  }
  diff_norm = sqrt(diff_norm) / sqrt(norm_of_true);
  return diff_norm;
}


double stokes_err_3d(const ki_Mat& domain,
                     const std::vector<double>& domain_points,
                     Boundary * boundary){

  ki_Mat truth(domain_points.size(), 1);
  for (int i = 0; i < domain_points.size(); i += 3) {
    double x0 = domain_points[i];
    double x1 = domain_points[i + 1];
    double x2 = domain_points[i + 2];
    PointVec x(x0, x1, x2);
    if (!boundary->is_in_domain(x)) {
      truth.set(i, 0, 0.0);
      truth.set(i + 1, 0, 0.0);
      truth.set(i + 2, 0, 0.0);
      continue;
    }
    truth.set(i, 0, 1);
    truth.set(i + 1, 0, 0);
    truth.set(i + 2, 0, 0);
  }
std::cout<<"printing some diffs"<<std::endl;
  for(int i=0; i<12; i+=3){
    std::cout<<domain.get(i, 0)<<" should be 1"<<std::endl;
    std::cout<<domain.get(i+1, 0)<<" should be 0"<<std::endl;
    std::cout<<domain.get(i+2, 0)<<" should be 0"<<std::endl;
  }
  return (domain-truth).vec_two_norm()/truth.vec_two_norm();
}

// double laplace_neumann_error(const ki_Mat& domain,
//                              const std::vector<double>& domain_points,
//                              Boundary * boundary) {
//   std::vector<double> res;
//   std::vector<double> truth;

//   for (int i = 0; i < domain_points.size(); i += 2) {
//     double x0 = domain_points[i];
//     double x1 = domain_points[i + 1];
//     PointVec x(x0, x1);
//     if (boundary->is_in_domain(x)) {
//       res.push_back(domain.get(i / 2, 0));
//       double r = sqrt(pow(x0 - 0.5, 2) + pow(x1 - 0.5, 2));
//       truth.push_back(log(r));
//     }
//   }
//   double res_avg = 0.;
//   double truth_avg = 0.;
//   for (int i = 0; i < res.size(); i++) {
//     res_avg += res[i];
//     truth_avg += truth[i];
//   }
//   res_avg /= res.size();
//   truth_avg /= truth.size();

//   for (int i = 0; i < res.size(); i++) {
//     res[i] -= res_avg;
//     truth[i] -= truth_avg;
//   }

//   double diff_norm = 0.;
//   double truth_norm = 0.;
//   for (int i = 0; i < res.size(); i++) {
//     diff_norm += pow(res[i] - truth[i], 2);
//     truth_norm += pow(truth[i], 2);
//   }
//   return sqrt(diff_norm) / sqrt(truth_norm);
// }


// TEST(IeSolverTest, LaplaceCircleAnalyticAgreementElectron) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Circle());
//   boundary->initialize(pow(2, 12),  BoundaryCondition::SINGLE_ELECTRON);
//   QuadTree quadtree;
//   quadtree.initialize_tree(boundary.get(), 1, 2);
//   std::vector<double> domain_points;
//   get_domain_points(20, &domain_points, quadtree.min,
//                     quadtree.max, quadtree.min,
//                     quadtree.max);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(), domain_points);
//   ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
//                                        1e-13, 4, domain_points);
//   double err = laplace_error(sol, domain_points, boundary.get());
//   EXPECT_LE(err, 10 * 1e-13);
// }


// TEST(IeSolverTest, LaplaceSphereAnalyticAgreementElectron) {
//   srand(0);
//   int num_threads = 4;
//   double id_tol=1e-6;
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Sphere());
//   boundary->initialize(pow(2, 7),  BoundaryCondition::ELECTRON_3D);

//   QuadTree quadtree;
//   quadtree.initialize_tree(boundary.get(), 1, 3);
//   std::vector<double> domain_points;
//   get_domain_points3d(3, &domain_points, quadtree.min,
//                     quadtree.max);
//   Kernel kernel(1, 3, Kernel::Pde::LAPLACE, boundary.get(), domain_points);

//   double start = omp_get_wtime();
//   kernel.compute_diag_entries_3dlaplace(boundary.get());
//     double end = omp_get_wtime();
// std::cout<<"diags took "<<(end-start)<<std::endl;
//   ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
//                                        id_tol, num_threads, domain_points);
//   // std::vector<int> all_dofs;
//   //   for (int i = 0; i < kernel.solution_dimension * boundary->weights.size();
//   //        i++) {
//   //   all_dofs.push_back(i);
//   // }
//   // ki_Mat dense = kernel(all_dofs, all_dofs); 

//   // // for(int row=0; row<dense.height();row++){
//   // //   double total=0.;
//   // //   for(int entry=0;entry<dense.height(); entry++){
//   // //     if(entry==row) continue;
//   // //     total+=dense.get(row, entry);
//   // //   }
//   // //   dense.set(row, row, 1-total);
//   // // }
//   // ki_Mat mu(dense.height(), 1);
//   // dense.left_multiply_inverse(boundary->boundary_values, &mu);
//   // ki_Mat sol = kernel.forward()*mu;

//   double err = laplace_error3d(sol, domain_points, boundary.get());
//   EXPECT_LE(err, 10 * 1e-13);
// }


TEST(IeSolverTest, StokesSphereAnalyticAgreement) {
  srand(0);
  std::unique_ptr<Boundary> boundary =
    std::unique_ptr<Boundary>(new Sphere());
  boundary->initialize(pow(2, 5),  BoundaryCondition::STOKES_3D);

  QuadTree quadtree;
  quadtree.initialize_tree(boundary.get(), 3, 3);
  std::vector<double> domain_points;
  get_domain_points3d(3, &domain_points, quadtree.min,
                    quadtree.max);
  Kernel kernel(3, 3, Kernel::Pde::STOKES, boundary.get(), domain_points);
  kernel.compute_diag_entries_3dstokes(boundary.get());
  // ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
  //                                      1e-33, 4, domain_points);

  // std::vector<int> all_dofs;
  //   for (int i = 0; i < kernel.solution_dimension * boundary->weights.size();
  //        i++) {
  //   all_dofs.push_back(i);
  // }
  // ki_Mat dense = kernel(all_dofs, all_dofs); 

  ki_Mat forw=kernel.forward();
  // std::cout<<"printing topleft 12x12 of forward"<<std::endl;

  // for(int i=0; i<12;i++){
  //   for(int j=0; j<12; j++){
  //     std::cout<<forw.get(i,j)<<" ";
  //   }std::cout<<"\n\n"<<std::endl;
  // }
  ki_Mat mu(kernel.solution_dimension * boundary->weights.size(), 1);
  // dense.left_multiply_inverse(boundary->boundary_values, &mu);

  for(int i=0; i<mu.height(); i+=3){
    mu.set(i, 0, 1);
    mu.set(i+1, 0, 0);
    mu.set(i+2, 0, 0);
  }
  ki_Mat sol = forw*mu;

  double err = stokes_err_3d(sol, domain_points, boundary.get());
  EXPECT_LE(err, 10 * 1e-13);
}


// TEST(IeSolverTest, LaplaceAnnulusAnalyticAgreementElectron) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Annulus());
//   boundary->initialize(pow(2, 12),  BoundaryCondition::SINGLE_ELECTRON);
//   QuadTree quadtree;
//   quadtree.initialize_tree(boundary.get(), 1, 2);
//   std::vector<double> domain_points;
//   get_domain_points(20, &domain_points, quadtree.min,
//                     quadtree.max, quadtree.min,
//                     quadtree.max);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE, boundary.get(), domain_points);
//   ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
//                                        1e-13,
//                                        4, domain_points);
//   double err = laplace_error(sol, domain_points, boundary.get());
//   EXPECT_LE(err, 10 * 1e-13);
// }


// TEST(IeSolverTest, LaplaceNeumannDonutAnalyticAgreement) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Donut());
//   boundary->initialize(pow(2, 12),  BoundaryCondition::DEFAULT);

//   //TODO(John) fix this hack
//   int hole_nodes = pow(2, 12) / 5;
//   int num_points = pow(2, 12) + hole_nodes * 1;

//   boundary->boundary_values = ki_Mat(num_points, 1);
//   for (int i = 0; i < pow(2, 12); i++) {
//      boundary->boundary_values.set(i, 0, 1);
//   }
//   for (int i = pow(2, 12); i < num_points; i++) {
//      boundary->boundary_values.set(i, 0, -2.);
//   }

//   QuadTree quadtree;
//   quadtree.initialize_tree(boundary.get(), 1, 2);
//   std::vector<double> domain_points;
//   get_domain_points(20, &domain_points, quadtree.min,
//                     quadtree.max, quadtree.min,
//                     quadtree.max);
//   Kernel kernel(1, 2, Kernel::Pde::LAPLACE_NEUMANN, boundary.get(),
//                 domain_points);
//   ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
//                                        1e-13,
//                                        4, domain_points);
//   double err = laplace_neumann_error(sol, domain_points, boundary.get());
//   EXPECT_LE(err, 10 * 1e-13);
// }


// TEST(IeSolverTest, StokesCircleAnalyticAgreementTangent) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Circle());
//   boundary->initialize(pow(2, 12),  BoundaryCondition::TANGENT_VEC);
//   QuadTree quadtree;
//   quadtree.initialize_tree(boundary.get(), 2, 2);
//   std::vector<double> domain_points;
//   get_domain_points(20, &domain_points, quadtree.min,
//                     quadtree.max, quadtree.min,
//                     quadtree.max);
//   Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(), domain_points);
//   ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
//                                        1e-13,
//                                        4, domain_points);
//   ki_Mat stokes_sol = stokes_true_sol(domain_points, boundary.get(), -1, 2);

//   double err = (stokes_sol - sol).vec_two_norm() / stokes_sol.vec_two_norm();
//   EXPECT_LE(err, 10 * 1e-13);
// }


// TEST(IeSolverTest, StokesDonutAnalyticAgreementTangent) {
//   srand(0);
//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Donut());
//   boundary->initialize(pow(2, 12),  BoundaryCondition::TANGENT_VEC);
//   QuadTree quadtree;
//   quadtree.initialize_tree(boundary.get(), 2, 2);
//   std::vector<double> domain_points;
//   get_domain_points(20, &domain_points, quadtree.min,
//                     quadtree.max, quadtree.min, quadtree.max);
//   Kernel kernel(2, 2, Kernel::Pde::STOKES, boundary.get(), domain_points);
//   ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
//                                        1e-13,
//                                        4, domain_points);
//   ki_Mat stokes_sol = stokes_true_sol(domain_points, boundary.get(), -1, 2);

//   double err = (stokes_sol - sol).vec_two_norm() / stokes_sol.vec_two_norm();
//   EXPECT_LE(err, 10 * 1e-13);
// }


// TEST(IeSolverTest, Ex3UpdateLosesNoAcc) {
//   double id_tol = 1e-6;
//   Kernel::Pde pde = Kernel::Pde::STOKES;
//   int num_boundary_points = pow(2, 12);
//   int domain_size = 20;
//   int domain_dimension = 2;
//   int solution_dimension = 2;
//   int fact_threads = 4;

//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Ex3Boundary());
//   boundary->initialize(num_boundary_points,
//                        BoundaryCondition::EX3B);
//   QuadTree quadtree;
//   quadtree.initialize_tree(boundary.get(), solution_dimension,
//                            domain_dimension);
//   std::vector<double> domain_points;
//   get_domain_points(domain_size, &domain_points, quadtree.min,
//                     quadtree.max, quadtree.min,
//                     quadtree.max);
//   Kernel kernel(solution_dimension, domain_dimension,
//                 pde, boundary.get(), domain_points);

//   for (int frame = 0; frame < 10; frame++) {
//     double ang = (frame / 10.) * 2 * M_PI;

//     boundary->perturbation_parameters[0] = ang;
//     boundary->perturbation_parameters[1] = ang + M_PI;
//     boundary->initialize(num_boundary_points, BoundaryCondition::EX3B);
//     quadtree.perturb(*boundary.get());
//     kernel.update_data(boundary.get());
//     ki_Mat solution = boundary_integral_solve(kernel, *(boundary.get()),
//                       &quadtree, id_tol, fact_threads, domain_points);

//     QuadTree fresh;
//     fresh.initialize_tree(boundary.get(), 2,  2);
//     ki_Mat new_sol = boundary_integral_solve(kernel, *(boundary.get()), &fresh,
//                      id_tol, fact_threads, domain_points);

//     ASSERT_LE((new_sol - solution).vec_two_norm() / new_sol.vec_two_norm(),
//               20 * id_tol);
//   }
// }


// TEST(IeSolverTest, TreeCopyGivesSameAnswer) {
//   double id_tol = 1e-6;
//   Kernel::Pde pde = Kernel::Pde::STOKES;
//   int num_boundary_points = pow(2, 12);
//   int domain_size = 20;
//   int domain_dimension = 2;
//   int solution_dimension = 2;
//   int fact_threads = 4;

//   std::unique_ptr<Boundary> boundary =
//     std::unique_ptr<Boundary>(new Ex1Boundary());
//   boundary->initialize(num_boundary_points,
//                        BoundaryCondition::DEFAULT);
//   QuadTree quadtree;
//   quadtree.initialize_tree(boundary.get(), solution_dimension,
//                            domain_dimension);
//   std::vector<double> domain_points;
//   get_domain_points(domain_size, &domain_points, quadtree.min,
//                     quadtree.max, quadtree.min,
//                     quadtree.max);
//   Kernel kernel(solution_dimension, domain_dimension,
//                 pde, boundary.get(), domain_points);
//   ki_Mat solution = boundary_integral_solve(kernel, *(boundary.get()),
//                     &quadtree, id_tol, fact_threads, domain_points);

//   QuadTree fresh;
//   quadtree.copy_into(&fresh);
//   ki_Mat new_sol = boundary_integral_solve(kernel, *(boundary.get()), &fresh,
//                    id_tol,
//                    fact_threads, domain_points);
//   ASSERT_LE((new_sol - solution).vec_two_norm(), 1e-15);
// }

}  // namespace kern_interp
