// Copyright 2019 John Paul Ryan
#include <omp.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include "kern_interp/kernel/kernel.h"

namespace kern_interp {


inline double Kernel::boundary_normals(int i) const {
  return (boundary_normals_.empty() ? 0. : boundary_normals_[i]);
}

inline double Kernel::boundary_weights(int i) const {
  return (boundary_weights_.empty() ? 0. : boundary_weights_[i]);
}

inline double Kernel::boundary_curvatures(int i) const {
  return (boundary_curvatures_.empty() ? 0. : boundary_curvatures_[i]);
}


void Kernel::update_data(Boundary* boundary) {
  boundary_points_ = boundary->points;
  boundary_normals_ = boundary->normals;
  boundary_weights_ = boundary->weights;
  boundary_curvatures_ = boundary->curvatures;
}


void Kernel::one_d_kern(int mat_idx, ki_Mat* ret, double r1, double r2,
                        double tn1, double tn2, double sn1, double sn2,
                        double sw, double sc, bool forward) const {
  if (pde == Pde::LAPLACE) {
    if (r1 == 0. && r2 == 0.) {
      ret->mat[mat_idx] = 0.5 + 0.5 * sc * sw * (1.0 / (2 * M_PI));
    } else {
      ret->mat[mat_idx] =  -sw * (1.0 / (2 * M_PI)) *
                           (r1 * sn1 + r2 * sn2) /
                           (r1 * r1 + r2 * r2);
    }
  } else if (pde == Pde::LAPLACE_NEUMANN) {
    if (forward) {
      ret->mat[mat_idx] =  sw * (1.0 / (2 * M_PI)) * log(sqrt(
                             r1 * r1 + r2 * r2));
    } else if (r1 == 0. && r2 == 0.) {
      ret->mat[mat_idx] = sw - 0.5 + 0.5 * sc * sw * (1.0 / (2 * M_PI));
    } else {
      ret->mat[mat_idx] =  sw + sw * (1.0 / (2 * M_PI)) *
                           (r1 * tn1 + r2 * tn2) /
                           (r1 * r1 + r2 * r2);
    }
  } else if (pde == Pde::GAUSS) {
    if (r1 == 0. && r2 == 0.) {
      ret->mat[mat_idx] = 2.;
    } else {
      ret->mat[mat_idx] = exp(-(pow(r1, 2) + pow(r2, 2)));
    }
  }
}


void Kernel::two_d_kern(int mat_idx, int tgt_parity, int src_parity,
                        ki_Mat* ret, double r1, double r2,
                        double tn1, double tn2, double sn1, double sn2,
                        double sw, double sc, bool forward) const {
  double sn[2] = {sn1, sn2};
  double tn[2] = {tn1, tn2};
  double r[2] = {r1, r2};
  double scale = 1 / M_PI;
  int is_diag = (tgt_parity + src_parity + 1) % 2;
  if (pde == Pde::STOKES) {
    if (r1 == 0. && r2 == 0.) {
      ret->mat[mat_idx] = - 0.5 * is_diag  + (1 - 2 * is_diag) * 0.5 * sc
                          * sw * scale * sn[(tgt_parity + 1) % 2] *
                          sn[(src_parity + 1) % 2 ] + sw * tn[tgt_parity]
                          * sn[src_parity];
    } else {
      ret->mat[mat_idx] = sw * scale * (r1 * sn[0] + r2 * sn[1]) /
                          (pow(r1 * r1 + r2 * r2, 2))
                          * r[tgt_parity] * r[src_parity]
                          + sw * tn[tgt_parity] * sn[src_parity ];
    }
  }
}


ki_Mat Kernel::operator()(const std::vector<int>& tgt_inds,
                          const std::vector<int>& src_inds, bool forward) const {
  ki_Mat ret(tgt_inds.size(), src_inds.size());
  int olda_ = tgt_inds.size();
  for (int j = 0; j < src_inds.size(); j++) {
    int src_ind = src_inds[j];
    int j_point_index = src_ind / solution_dimension;
    int j_points_vec_index = j_point_index * domain_dimension;

    double sp[2] = {boundary_points_[j_points_vec_index],
                    boundary_points_[j_points_vec_index + 1]
                   };
    double sn[2] =  {boundary_normals(j_points_vec_index),
                     boundary_normals(j_points_vec_index + 1)
                    };
    double sw =  boundary_weights(j_point_index);
    double sc = boundary_curvatures(j_point_index);
    for (int i = 0; i < tgt_inds.size(); i++) {
      int tgt_ind = tgt_inds[i];
      int i_point_index = tgt_ind / solution_dimension;
      int i_points_vec_index = i_point_index * domain_dimension;

      double tp[2], tn[2];
      if (forward) {
        tp[0] = domain_points[i_points_vec_index];
        tp[1] = domain_points[i_points_vec_index + 1];
        tn[0] = 0;
        tn[1] = 0;
      } else {
        tp[0] = boundary_points_[i_points_vec_index];
        tp[1] = boundary_points_[i_points_vec_index + 1];
        tn[0] = boundary_normals(i_points_vec_index);
        tn[1] = boundary_normals(i_points_vec_index + 1);
      }
      double r[2] = {tp[0] - sp[0], tp[1] - sp[1]};
      switch (solution_dimension) {
        case 1:
          one_d_kern(i + olda_ * j, &ret, r[0], r[1], tn[0], tn[1],
                     sn[0], sn[1], sw, sc, forward);
          break;
        case 2:
          two_d_kern(i + olda_ * j, tgt_ind % 2, src_ind % 2, &ret, r[0], r[1],
                     tn[0], tn[1], sn[0], sn[1],  sw, sc);
          break;
      }
    }
  }
  return ret;
}


ki_Mat Kernel::get_id_mat(const QuadTree* tree,
                          const QuadTreeNode* node) const {

  std::vector<int> active_box = node->dof_lists.active_box;
  // Grab all points inside the proxy circle which are outside the box
  std::vector<int> inner_circle, outside_box;

  // So if we're at level 2 or 1, we don't use the proxy trick
  // If at level 1, just grab active from neighbors
  if (node->level == 1) {
    for (QuadTreeNode* level_node : tree->levels[node->level]->nodes) {
      if (level_node != node) {
        for (int matrix_index :
             level_node->dof_lists.active_box) {
          outside_box.push_back(matrix_index);
        }
      }
    }
    ki_Mat mat(2 * outside_box.size(), active_box.size());
    mat.set_submatrix(0, outside_box.size(), 0, active_box.size(),
                      (*this)(outside_box, active_box), false, true);
    mat.set_submatrix(outside_box.size(), 2 * outside_box.size(),
                      0, active_box.size(),
                      (*this)(active_box, outside_box), true, true);
    return mat;
  }
  // If at level 2, grab active from all on level, plus from leaves of level 1
  if (node->level == 2) {
    for (QuadTreeNode* level_node : tree->levels[node->level]->nodes) {
      if (level_node != node) {
        for (int matrix_index :
             level_node->dof_lists.active_box) {
          outside_box.push_back(matrix_index);
        }
      }
    }
    for (int lvl = node->level - 1; lvl >= 0; lvl--) {
      for (QuadTreeNode* level_node : tree->levels[lvl]->nodes) {
        if (level_node->is_leaf) {
          for (int matrix_index :
               level_node->dof_lists.original_box) {
            outside_box.push_back(matrix_index);
          }
        }
      }
    }

    ki_Mat mat(2 * outside_box.size(), active_box.size());
    mat.set_submatrix(0, outside_box.size(), 0, active_box.size(),
                      (*this)(outside_box, active_box), false, true);
    mat.set_submatrix(outside_box.size(), 2 * outside_box.size(),
                      0, active_box.size(),
                      (*this)(active_box, outside_box), true, true);
    return mat;
  }

  for (int matrix_index : node->dof_lists.near) {
    int point_index = matrix_index / solution_dimension;
    int points_vec_index = point_index * domain_dimension;
    double x = boundary_points_[points_vec_index];
    double y = boundary_points_[points_vec_index + 1];
    double dist = 0.;
    for (int d = 0; d < domain_dimension; d++) {
      dist += pow(node->center[d] - boundary_points_[points_vec_index + d], 2);
    }
    dist = sqrt(dist);
    if (dist < RADIUS_RATIO * node->side_length) {
      inner_circle.push_back(matrix_index);
    }
  }

  ki_Mat pxy = get_proxy_mat(node->center, node->side_length
                             * RADIUS_RATIO, tree, active_box);
  // Now all the matrices are gathered, put them into mat.
  ki_Mat mat(2 * inner_circle.size() + pxy.height(), active_box.size());

  mat.set_submatrix(0, inner_circle.size(),
                    0, active_box.size(), (*this)(inner_circle, active_box),
                    false, true);


  mat.set_submatrix(inner_circle.size(), 2 * inner_circle.size(),
                    0, active_box.size(), (*this)(active_box, inner_circle),
                    true, true);
  mat.set_submatrix(2 * inner_circle.size(),  pxy.height()
                    + 2 * inner_circle.size(), 0,
                    active_box.size(),  pxy, false, true);

  return mat;
}


ki_Mat Kernel::get_proxy_mat(std::vector<double> center,
                             double r, const QuadTree * tree,
                             const std::vector<int>& box_inds) const {
  if (domain_dimension == 3) {
    return get_proxy_mat3d(center, r, tree, box_inds);
  }
  // each row is a pxy point, cols are box dofs
  double pxy_w = 2.0 * M_PI * r / NUM_PROXY_POINTS;
  std::vector<double>  pxy_p, pxy_n;

  for (int i = 0; i < NUM_PROXY_POINTS; i++) {
    double ang = 2 * M_PI * i * (1.0 / NUM_PROXY_POINTS);
    for (int k = 1; k < 2; k++) {   // modify this for annulus proxy
      double eps = (k - 1) * 1;
      pxy_p.push_back(center[0] + (r + eps) * cos(ang));
      pxy_p.push_back(center[1] + (r + eps) * sin(ang));
      pxy_n.push_back(cos(ang));
      pxy_n.push_back(sin(ang));
    }
  }
  int lda = solution_dimension * pxy_p.size();
  ki_Mat ret(lda, box_inds.size());
  for (int j = 0; j < box_inds.size(); j++) {
    int box_ind = box_inds[j];
    int j_point_index = box_ind / solution_dimension;
    int j_points_vec_index = j_point_index * domain_dimension;
    double bp[2] = {boundary_points_[j_points_vec_index],
                    boundary_points_[j_points_vec_index + 1]
                   };
    double bn[2] =  {boundary_normals(j_points_vec_index),
                     boundary_normals(j_points_vec_index + 1)
                    };
    double bw =  boundary_weights(j_point_index);
    for (int i = 0; i < pxy_p.size(); i += 2) {
      double pp[2] = {pxy_p[i], pxy_p[i + 1]};
      double pn[2] = {pxy_n[i], pxy_n[i + 1]};
      double r[2] = {pp[0] - bp[0], pp[1] - bp[1]};
      switch (solution_dimension) {
        case 1:
          // box to pxy
          one_d_kern((i / 2) + lda * j, &ret, r[0], r[1],  pn[0], pn[1],
                     bn[0], bn[1], bw,  0);
          // pxy to box
          one_d_kern((pxy_p.size() / 2) + (i / 2) + lda * j, &ret,
                     -r[0], -r[1],  bn[0], bn[1], pn[0], pn[1], pxy_w, 0);
          break;
        case 2:
          for (int pxy_parity = 0; pxy_parity < 2; pxy_parity++) {
            // box to pxy
            two_d_kern(i + pxy_parity + lda * j, pxy_parity, box_ind % 2,
                       &ret, r[0], r[1],  pn[0], pn[1], bn[0], bn[1], bw, 0);
            // pxy to box
            two_d_kern(pxy_p.size() + i + pxy_parity + lda * j,
                       box_ind % 2, pxy_parity, &ret, -r[0], -r[1],
                       bn[0], bn[1], pn[0], pn[1], pxy_w, 0);
          }
          break;
      }
    }
  }
  return ret;
}

ki_Mat Kernel::get_proxy_mat3d(std::vector<double> center,
                               double r, const QuadTree * tree,
                               const std::vector<int>& box_inds) const {

  // each row is a pxy point, cols are box dofs
  std::vector<double> pxy_p, pxy_n, pxy_w;

  for (int i = 0; i < pxy_thetas.size(); i++) {
    double theta = pxy_thetas[i];
    for (int j = 0; j < pxy_phis.size(); j++) {   // modify this for annulus proxy
      double phi = pxy_phis[j];
      pxy_p.push_back(center[0] + r * sin(phi) * cos(theta));
      pxy_p.push_back(center[1] + r * sin(phi) * sin(theta));
      pxy_p.push_back(center[2] + r * cos(phi));
      pxy_n.push_back(sin(phi) * cos(theta));
      pxy_n.push_back(sin(phi) * sin(theta));
      pxy_n.push_back(cos(phi));
      pxy_w.push_back(pow(r, 2) *pxy_phi_weights[j]*pxy_theta_weights[i]);
    }
  }

  int lda = solution_dimension * domain_dimension * pxy_w.size();

  ki_Mat ret(lda, box_inds.size());
  for (int j = 0; j < box_inds.size(); j++) {
    int box_ind = box_inds[j];
    int j_point_index = box_ind / solution_dimension;
    int j_points_vec_index = j_point_index * domain_dimension;
    std::vector<double> bp, bn;
    for (int d = 0; d < domain_dimension; d++) {
      bp.push_back(boundary_points_[j_points_vec_index + d]);
      bn.push_back(boundary_normals(j_points_vec_index + d));
    }
    double bw =  boundary_weights(j_point_index);
    for (int i = 0; i < pxy_p.size(); i += domain_dimension) {
      std::vector<double> pp, pn;
      for (int d = 0; d < domain_dimension; d++) {
        pp.push_back(pxy_p[i + d]);
        pn.push_back(pxy_n[i + d]);
      }
      double r[2] = {pp[0] - bp[0], pp[1] - bp[1]};

      
      // box to pxy
      one_d_kern((i / 2) + lda * j, &ret, r[0], r[1],  pn[0], pn[1],
                 bn[0], bn[1], bw,  0);
      // pxy to box
      one_d_kern((pxy_p.size() / 2) + (i / 2) + lda * j, &ret,
                 -r[0], -r[1],  bn[0], bn[1], pn[0], pn[1], pxy_w, 0);
      // get kernel (both ways) based on the above.
    }
  }
  return ret;
}


ki_Mat Kernel::forward() const {
  switch (solution_dimension) {
    case 1: {
      std::vector<int> tgt_inds(domain_points.size() / 2),
          src_inds(boundary_points_.size() / 2);
      for (int i = 0; i < domain_points.size() / 2; i++) tgt_inds[i] = i;
      for (int j = 0; j < boundary_points_.size() / 2; j++) src_inds[j] = j;
      return (*this)(tgt_inds, src_inds, true);
      break;
    }
    case 2:
    default: {
      std::vector<int> tgt_inds(domain_points.size()),
          src_inds(boundary_points_.size());
      for (int i = 0; i < domain_points.size(); i++) tgt_inds[i] = i;
      for (int j = 0; j < boundary_points_.size(); j++) src_inds[j] = j;
      return (*this)(tgt_inds, src_inds, true);
      break;
    }
  }
}

}  // namespace kern_interp
