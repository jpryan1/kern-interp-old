// Copyright 2019 John Paul Ryan
#ifndef KERN_INTERP_LINEAR_SOLVE_H_
#define KERN_INTERP_LINEAR_SOLVE_H_

#include <vector>
#include <string>
#include "kern_interp/quadtree/quadtree.h"
#include "kern_interp/boundaries/boundary.h"
#include "kern_interp/kernel/kernel.h"
#include "kern_interp/skel_factorization/skel_factorization.h"


#define TIMING_ITERATIONS 5

namespace kern_interp {

ki_Mat boundary_integral_solve(const Kernel& kernel, const Boundary& boundary,
                               QuadTree* quadtree, double id_tol, int fact_threads,
                               const std::vector<double>& domain_points);
ki_Mat initialize_U_mat(const Kernel::Pde pde,
                        const std::vector<Hole>& holes,
                        const std::vector<double>& tgt_points);
ki_Mat initialize_Psi_mat(const Kernel::Pde pde,
                          const std::vector<Hole>& holes, const Boundary& boundary);
void get_domain_points(int domain_size, std::vector<double>* points,
                       double x_min, double x_max, double y_min, double y_max);
void linear_solve(const SkelFactorization& skel_factorization,
                  const QuadTree& quadtree, const ki_Mat& f, ki_Mat* mu,
                  ki_Mat* alpha = nullptr);

void schur_solve(const SkelFactorization & skel_factorization,
                 const QuadTree & quadtree, const ki_Mat & U,
                 const ki_Mat & Psi,
                 const ki_Mat & f, const ki_Mat & K_domain,
                 const ki_Mat & U_forward,  ki_Mat * solution);


}  // namespace kern_interp

#endif  // KERN_INTERP_LINEAR_SOLVE_H_
