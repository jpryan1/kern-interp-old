// Copyright 2019 John Paul Ryan
#ifndef KERN_INTERP_KERNEL_KERNEL_H_
#define KERN_INTERP_KERNEL_KERNEL_H_

#include <vector>
#include "kern_interp/boundaries/boundary.h"
#include "kern_interp/ki_mat.h"
#include "kern_interp/quadtree/quadtree.h"

#define NUM_PROXY_POINTS 128
#define RADIUS_RATIO 1.5

namespace kern_interp {

struct Kernel {
  enum Pde {
    LAPLACE,
    LAPLACE_NEUMANN,
    GAUSS,
    STOKES
  };
  Kernel() {}
  Kernel(int solution_dimension_, int domain_dimension_,
         Kernel::Pde pde_, Boundary* boundary,
         std::vector<double> domain_points_) :
    solution_dimension(solution_dimension_),
    domain_dimension(domain_dimension_),
    pde(pde_),
    boundary_points_(boundary->points),
    boundary_normals_(boundary->normals),
    boundary_weights_(boundary->weights),
    boundary_curvatures_(boundary->curvatures),
    domain_points(domain_points_) {}

  int solution_dimension, domain_dimension;
  Kernel::Pde pde;
  std::vector<double> boundary_points_, boundary_normals_,
      boundary_weights_, boundary_curvatures_, domain_points;

  void update_data(Boundary* boundary);

  void one_d_kern(int mat_idx, ki_Mat* ret, double r1,
                  double r2, double tn1, double tn2, double sn1, double sn2,
                  double sw, double sc, bool forward = false) const;
  void two_d_kern(int mat_idx, int tgt_parity, int src_parity,
                  ki_Mat* ret, double r1, double r2, double tn1, double tn2,
                  double sn1, double sn2, double sw, double sc,
                  bool forward = false) const;
  ki_Mat operator()(const std::vector<int> & tgt_inds,
                    const std::vector<int> & src_inds,
                    bool forward = false) const;

  ki_Mat get_id_mat(const QuadTree * tree,
                    const QuadTreeNode * node) const;
  ki_Mat get_proxy_mat(std::vector<double> center,
                       double r, const QuadTree * tree,
                       const std::vector<int> & box_inds) const;

  ki_Mat get_proxy_mat3d(std::vector<double> center,
                         double r, const QuadTree * tree,
                         const std::vector<int> & box_inds) const;

  ki_Mat forward() const;

  inline double boundary_normals(int i) const;
  inline double boundary_weights(int i) const;
  inline double boundary_curvatures(int i) const;
};  // struct


}  // namespace kern_interp

#endif  // KERN_INTERP_KERNEL_KERNEL_H_
