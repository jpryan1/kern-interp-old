// Copyright 2019 John Paul Ryan
#include <omp.h>
#include <cmath>
#include <iostream>
#include <cassert>
#include "kern_interp/kernel/kernel.h"

namespace kern_interp {


void Kernel::laplace(int mat_idx, ki_Mat* ret, double r1,
                     double r2, double sw, double sc,
                     double sn1, double sn2, double tn1, double tn2,
                     bool forward) const {
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
  } else {
  }
}


void Kernel::update_boundary(Boundary* boundary) {
  boundary_points = boundary->points;
  boundary_normals = boundary->normals;
  boundary_weights = boundary->weights;
  boundary_curvatures = boundary->curvatures;
}


ki_Mat Kernel::stokes_get(const std::vector<int>& tgt_inds,
                          const std::vector<int>& src_inds, bool forward) const {
  double scale = 1.0 / (M_PI);
  ki_Mat ret(tgt_inds.size(), src_inds.size());
  int olda_ = tgt_inds.size();
  for (int j = 0; j < src_inds.size(); j++) {
    int src_ind = src_inds[j];
    int j_point_index = src_ind / 2;
    int j_points_vec_index = j_point_index * 2;
    double sp[2] = {boundary_points[j_points_vec_index],
                    boundary_points[j_points_vec_index + 1]
                   };
    double sn[2] =  {boundary_normals[j_points_vec_index],
                     boundary_normals[j_points_vec_index + 1]
                    };
    double sw =  boundary_weights[j_point_index];
    double sc = boundary_curvatures[j_point_index];
    for (int i = 0; i < tgt_inds.size(); i++) {
      int tgt_ind = tgt_inds[i];
      int i_point_index = tgt_ind / 2;
      int i_points_vec_index = i_point_index * 2;
      double tp[2], tn[2];
      if (forward) {
        tp[0] = domain_points[i_points_vec_index];
        tp[1] = domain_points[i_points_vec_index + 1];
        tn[0] = 0;
        tn[1] = 0;
      } else {
        tp[0] = boundary_points[i_points_vec_index];
        tp[1] = boundary_points[i_points_vec_index + 1];
        tn[0] = boundary_normals[i_points_vec_index];
        tn[1] = boundary_normals[i_points_vec_index + 1];
      }
      if (tp[0] == sp[0] && tp[1] == sp[1]) {
        int parity = (tgt_ind + src_ind + 1) % 2;
        ret.mat[i + olda_ * j] = - 0.5 * parity  + (1 - 2 * parity) * 0.5 * sc
                                 * sw * scale * sn[(tgt_ind + 1) % 2] *
                                 sn[(src_ind + 1) % 2 ] + sw * tn[tgt_ind % 2]
                                 * sn[src_ind % 2];
      } else {
        double r[2] = {tp[0] - sp[0], tp[1] - sp[1]};
        ret.mat[i + olda_ * j] = sw * scale * (r[0] * sn[0] + r[1] * sn[1]) /
                                 (pow(r[0] * r[0] + r[1] * r[1], 2))
                                 * r[tgt_ind % 2] * r[src_ind % 2 ]
                                 + sw * tn[tgt_ind % 2] * sn[src_ind % 2 ];
      }
    }
  }
  return ret;
}


ki_Mat Kernel::operator()(const std::vector<int>& tgt_inds,
                          const std::vector<int>& src_inds, bool forward) const {
  switch (pde) {
    case Kernel::Pde::LAPLACE:
    case Kernel::Pde::LAPLACE_NEUMANN:
      return laplace_get(tgt_inds, src_inds, forward);
      break;
    case Kernel::Pde::STOKES:
    default:
      return stokes_get(tgt_inds, src_inds, forward);
      break;
  }
}


ki_Mat Kernel::laplace_get(const std::vector<int>& tgt_inds,
                           const std::vector<int>& src_inds, bool forward) const {
  ki_Mat ret(tgt_inds.size(), src_inds.size());
  int olda_ = tgt_inds.size();
  for (int j = 0; j < src_inds.size(); j++) {
    int src_ind = src_inds[j];

    double sp[2] = {boundary_points[2 * src_ind],
                    boundary_points[2 * src_ind + 1]
                   };
    double sn[2] =  {boundary_normals[2 * src_ind],
                     boundary_normals[2 * src_ind + 1]
                    };
    double sw =  boundary_weights[src_ind];
    double sc = boundary_curvatures[src_ind];

    for (int i = 0; i < tgt_inds.size(); i++) {
      int tgt_ind = tgt_inds[i];
      double tp[2], tn[2];
      if (forward) {
        tp[0] = domain_points[2 * tgt_ind];
        tp[1] = domain_points[2 * tgt_ind + 1];
        tn[0] = 0;
        tn[1] = 0;
      } else {
        tp[0] = boundary_points[2 * tgt_ind];
        tp[1] = boundary_points[2 * tgt_ind + 1];
        tn[0] = boundary_normals[2 * tgt_ind];
        tn[1] = boundary_normals[2 * tgt_ind + 1];
      }
      double r[2] = {tp[0] - sp[0], tp[1] - sp[1]};
      laplace(i + olda_ * j, &ret, r[0], r[1], sw, sc, sn[0], sn[1], tn[0], tn[1],
              forward);
    }
  }
  return ret;
}


ki_Mat Kernel::get_id_mat(const QuadTree* tree,
                          const QuadTreeNode* node) const {
  double cntr_x = node->corners[0] + node->side_length / 2.0;
  double cntr_y = node->corners[1] + node->side_length / 2.0;

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
    for (QuadTreeNode* level_node : tree->levels[node->level - 1]->nodes) {
      if (level_node->is_leaf) {
        for (int matrix_index :
             level_node->dof_lists.original_box) {
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

  for (int matrix_index : node->dof_lists.near) {
    int point_index = matrix_index / solution_dimension;
    int points_vec_index = point_index * domain_dimension;
    double x = boundary_points[points_vec_index];
    double y = boundary_points[points_vec_index + 1];
    double dist = sqrt(pow(cntr_x - x, 2) + pow(cntr_y - y, 2));
    if (dist < RADIUS_RATIO * node->side_length) {
      inner_circle.push_back(matrix_index);
    }
  }

  ki_Mat pxy = get_proxy_mat(cntr_x, cntr_y, node->side_length
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


ki_Mat Kernel::get_proxy_mat(double cntr_x, double cntr_y,
                             double r, const QuadTree * tree,
                             const std::vector<int>& box_inds) const {
  // each row is a pxy point, cols are box dofs
  double proxy_weight = 2.0 * M_PI * r / NUM_PROXY_POINTS;
  std::vector<double>  pxy_p, pxy_n;
  for (int i = 0; i < NUM_PROXY_POINTS; i++) {
    double ang = 2 * M_PI * i * (1.0 / NUM_PROXY_POINTS);
    for (int k = 1; k < 2; k++) {   // modify this for annulus proxy
      double eps = (k - 1) * 0.001;
      pxy_p.push_back(cntr_x + (r + eps) * cos(ang));
      pxy_p.push_back(cntr_y + (r + eps) * sin(ang));
      pxy_n.push_back(cos(ang));
      pxy_n.push_back(sin(ang));
    }
  }
  switch (pde) {
    case Kernel::Pde::LAPLACE:
    case Kernel::Pde::LAPLACE_NEUMANN:
      return laplace_proxy_get(pxy_p, pxy_n ,
                               proxy_weight, box_inds);
      break;
    case Kernel::Pde::STOKES:
    default:
      return stokes_proxy_get(pxy_p, pxy_n , proxy_weight,
                              box_inds);
  }
}


ki_Mat Kernel::stokes_proxy_get(const std::vector<double> & pxy_p,
                                const std::vector<double> & pxy_n,
                                double pxy_w,
                                const std::vector<int> & box_inds) const {
  double scale = 1.0 / (M_PI);
  ki_Mat ret(2 * pxy_p.size(), box_inds.size());
  int lda = 2 * pxy_p.size();
  for (int j = 0; j < box_inds.size(); j++) {
    int src_ind = box_inds[j];
    int j_point_index = src_ind / 2;
    int j_points_vec_index = j_point_index * 2;
    double sp[2] = {boundary_points[j_points_vec_index],
                    boundary_points[j_points_vec_index + 1]
                   };
    double sn[2] =  {boundary_normals[j_points_vec_index],
                     boundary_normals[j_points_vec_index + 1]
                    };
    double sw =  boundary_weights[j_point_index];
    for (int i = 0; i < pxy_p.size(); i += 2) {
      double tp[2] = {pxy_p[i], pxy_p[i + 1]};
      double tn[2] = {pxy_n[i], pxy_n[i + 1]};
      double r[2] = {tp[0] - sp[0], tp[1] - sp[1]};
      double potential = sw * scale * (r[0] * sn[0] + r[1] * sn[1]) /
                         (pow(r[0] * r[0] + r[1] * r[1], 2));
      ret.mat[i + lda * j] = potential * r[0] * r[src_ind % 2 ]
                             + sw * tn[0] * sn[src_ind % 2 ];
      ret.mat[i + 1 + lda * j] = potential * r[1] * r[src_ind % 2 ]
                                 + sw * tn[1] * sn[src_ind % 2 ];


      potential = -pxy_w * scale * (r[0] * tn[0] + r[1] * tn[1]) /
                  (pow(r[0] * r[0] + r[1] * r[1], 2));
      ret.mat[pxy_p.size() + i + j * lda] =
        potential * r[0] * r[src_ind % 2] + pxy_w * sn[src_ind % 2] * tn[0];
      ret.mat[pxy_p.size() + i + 1 + j * lda] =
        potential * r[1] * r[src_ind % 2] + pxy_w * sn[src_ind % 2] * tn[1];
    }
  }
  return ret;
}


ki_Mat Kernel::laplace_proxy_get(const std::vector<double> & pxy_p,
                                 const std::vector<double> & pxy_n,
                                 double pxy_w,
                                 const std::vector<int> & box_inds) const {
  int lda = pxy_p.size();
  ki_Mat ret(lda, box_inds.size());
  for (int j = 0; j < box_inds.size(); j++) {
    int box_ind = box_inds[j];
    double bp1 = boundary_points[2 * box_ind];
    double bp2 =  boundary_points[2 * box_ind + 1];
    double bn1 =  boundary_normals[2 * box_ind];
    double bn2 = boundary_normals[2 * box_ind + 1];
    double bw =  boundary_weights[box_ind];
    for (int i = 0; i < pxy_p.size(); i += 2) {
      double pp1 = pxy_p[i];
      double pp2 = pxy_p[i + 1];
      double r[2] = {pp1 - bp1, pp2 - bp2};
      laplace((i / 2) + lda * j, &ret, r[0], r[1], bw, 0, bn1, bn2, pxy_n[i],
              pxy_n[i + 1]);
      laplace((lda / 2) + (i / 2) + lda * j, &ret, -r[0], -r[1], pxy_w, 0,
              pxy_n[i], pxy_n[i + 1], bn1, bn2);
    }
  }
  return ret;
}


ki_Mat Kernel::forward() const {
  switch (solution_dimension) {
    case 1: {
      std::vector<int> tgt_inds(domain_points.size() / 2),
          src_inds(boundary_points.size() / 2);
      for (int i = 0; i < domain_points.size() / 2; i++) tgt_inds[i] = i;
      for (int j = 0; j < boundary_points.size() / 2; j++) src_inds[j] = j;
      return (*this)(tgt_inds, src_inds, true);
      break;
    }
    case 2:
    default: {
      std::vector<int> tgt_inds(domain_points.size()),
          src_inds(boundary_points.size());
      for (int i = 0; i < domain_points.size(); i++) tgt_inds[i] = i;
      for (int j = 0; j < boundary_points.size(); j++) src_inds[j] = j;
      return (*this)(tgt_inds, src_inds, true);
      break;
    }
  }
}

}  // namespace kern_interp
