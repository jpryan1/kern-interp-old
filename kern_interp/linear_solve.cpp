// Copyright 2019 John Paul Ryan
#include <string.h>
#include <omp.h>
#include <cassert>
#include <cmath>
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include "kern_interp/ki_mat.h"
#include "kern_interp/quadtree/quadtree.h"
#include "kern_interp/linear_solve.h"


namespace kern_interp {


ki_Mat initialize_U_mat(const Kernel::Pde pde,
                        const std::vector<Hole>& holes,
                        const std::vector<double>& tgt_points) {
  ki_Mat U;
  if (holes.size() == 0) return ki_Mat(0, 0);
  switch (pde) {
    case Kernel::Pde::LAPLACE: {
      U = ki_Mat(tgt_points.size() / 2, holes.size());
      for (int i = 0; i < tgt_points.size(); i += 2) {
        for (int hole_idx = 0; hole_idx < holes.size(); hole_idx++) {
          Hole hole = holes[hole_idx];
          Vec2 r = Vec2(tgt_points[i], tgt_points[i + 1]) - hole.center;
          U.set(i / 2, hole_idx, log(r.norm()));
        }
      }
      break;
    }
    case Kernel::Pde::STOKES: {
      U = ki_Mat(tgt_points.size(), 3 * holes.size());
      for (int i = 0; i < tgt_points.size(); i += 2) {
        for (int hole_idx = 0; hole_idx < holes.size(); hole_idx++) {
          Hole hole = holes[hole_idx];
          Vec2 r = Vec2(tgt_points[i], tgt_points[i + 1]) - hole.center;
          double scale = 1.0 / (4 * M_PI);
          U.set(i, 3 * hole_idx, scale *
                (log(1 / r.norm()) +
                 (1.0 / pow(r.norm(), 2)) * r.a[0] * r.a[0]));
          U.set(i + 1, 3 * hole_idx, scale *
                ((1.0 / pow(r.norm(), 2)) * r.a[1] * r.a[0]));
          U.set(i, 3 * hole_idx + 1, scale *
                ((1.0 / pow(r.norm(), 2)) * r.a[0] * r.a[1]));
          U.set(i + 1, 3 * hole_idx + 1, scale *
                (log(1 / r.norm()) +
                 (1.0 / pow(r.norm(), 2)) * r.a[1] * r.a[1]));
          U.set(i, 3 * hole_idx + 2, r.a[1] * (scale / pow(r.norm(), 2)));
          U.set(i + 1, 3 * hole_idx + 2, -r.a[0] * (scale / pow(r.norm(), 2)));
        }
      }
      break;
    }
    case Kernel::Pde::LAPLACE_NEUMANN: {
      U = ki_Mat(0, 0);
      break;
    }
  }
  return U;
}


ki_Mat initialize_Psi_mat(const Kernel::Pde pde,
                          const std::vector<Hole>& holes,
                          const Boundary& boundary) {
  if (holes.size() == 0) return ki_Mat(0, 0);
  ki_Mat Psi;
  switch (pde) {
    case Kernel::Pde::LAPLACE: {
      Psi = ki_Mat(holes.size(), boundary.points.size() / 2);
      for (int i = 0; i < boundary.points.size(); i += 2) {
        Vec2 x = Vec2(boundary.points[i], boundary.points[i + 1]);
        for (int hole_idx = 0; hole_idx < holes.size(); hole_idx++) {
          Hole hole = holes[hole_idx];
          if ((x - hole.center).norm() < hole.radius + 1e-8) {
            Psi.set(hole_idx, i / 2, boundary.weights[i / 2]);
            break;
          }
        }
      }
      break;
    }
    case Kernel::Pde::STOKES: {
      Psi = ki_Mat(3 * holes.size(), boundary.points.size());
      for (int i = 0; i < boundary.points.size(); i += 2) {
        Vec2 x = Vec2(boundary.points[i], boundary.points[i + 1]);
        for (int hole_idx = 0; hole_idx < holes.size(); hole_idx++) {
          Hole hole = holes[hole_idx];
          if ((x - hole.center).norm() < hole.radius + 1e-8) {
            Psi.set(3 * hole_idx, i, boundary.weights[i / 2]);
            Psi.set(3 * hole_idx + 1, i + 1, boundary.weights[i / 2]);
            Psi.set(3 * hole_idx + 2, i, boundary.weights[i / 2]*x.a[1]);
            Psi.set(3 * hole_idx + 2, i + 1, -boundary.weights[i / 2]*x.a[0]);
            break;
          }
        }
      }
      break;
    }
    case Kernel::Pde::LAPLACE_NEUMANN: {
      Psi = ki_Mat(0, 0);
      break;
    }
  }
  return Psi;
}


void linear_solve(const SkelFactorization& skel_factorization,
                  const QuadTree& quadtree, const ki_Mat& f, ki_Mat* mu,
                  ki_Mat* alpha) {
  *mu = ki_Mat(f.height(), 1);
  if (alpha == nullptr) {
    skel_factorization.solve(quadtree, mu, f);
  } else {
    *alpha = ki_Mat(quadtree.U.width(), 1);
    skel_factorization.multiply_connected_solve(quadtree, mu, alpha, f);
  }
}


void schur_solve(const SkelFactorization & skel_factorization,
                 const QuadTree & quadtree, const ki_Mat & U,
                 const ki_Mat & Psi,
                 const ki_Mat & f, const ki_Mat & K_domain,
                 const ki_Mat & U_forward,  ki_Mat * solution) {
  ki_Mat mu;
  if (U.width() == 0) {
    linear_solve(skel_factorization, quadtree, f, &mu);
    *solution = K_domain * mu;
  } else {
    ki_Mat alpha;
    linear_solve(skel_factorization, quadtree, f, &mu, &alpha);
    *solution = (K_domain * mu) + (U_forward * alpha);
  }
}


ki_Mat boundary_integral_solve(const Kernel& kernel, const Boundary& boundary,
                               QuadTree * quadtree, double id_tol,
                               int fact_threads,
                               const std::vector<double>& domain_points) {
  // Consider making init instead of constructor for readability
  SkelFactorization skel_factorization(id_tol, fact_threads);

  ki_Mat K_domain = kernel.forward();

  ki_Mat f = boundary.boundary_values;

  ki_Mat U = initialize_U_mat(kernel.pde, boundary.holes, boundary.points);
  ki_Mat Psi = initialize_Psi_mat(kernel.pde, boundary.holes,
                                  boundary);
  ki_Mat U_forward = initialize_U_mat(kernel.pde, boundary.holes,
                                      kernel.domain_points);

  ki_Mat domain_solution((kernel.domain_points.size() / 2)*
                         kernel.solution_dimension, 1);
  quadtree->U = U;
  quadtree->Psi = Psi;
  skel_factorization.skeletonize(kernel, quadtree);

  schur_solve(skel_factorization, *quadtree, U, Psi, f, K_domain,
              U_forward, &domain_solution);
  for (int i = 0; i < kernel.domain_points.size(); i += 2) {
    Vec2 point = Vec2(kernel.domain_points[i], kernel.domain_points[i + 1]);
    if (!boundary.is_in_domain(point)) {
      if (kernel.solution_dimension == 2) {
        domain_solution.set(i, 0, 0.);
        domain_solution.set(i + 1, 0, 0.);
      } else {
        domain_solution.set(i / 2, 0, 0.);
      }
    }
  }
  return domain_solution;
}


void get_domain_points(int domain_size, std::vector<double>* points,
                       double x_min, double x_max, double y_min, double y_max) {
  for (int i = 0; i < domain_size; i++) {
    double x = x_min + ((i + 0.0) / (domain_size - 1)) * (x_max - x_min);
    for (int j = 0; j < domain_size; j++) {
      double y = y_min + ((j + 0.0) / (domain_size - 1)) * (y_max - y_min);
      points->push_back(x);
      points->push_back(y);
    }
  }
}



}  // namespace kern_interp
