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

void Kernel::three_d_stokes(int mat_idx, int tgt_parity, int src_parity,
                        ki_Mat* ret, double r1, double r2, double r3, const ki_Mat& diag,
                        double tn1, double tn2, double tn3, double sn1, double sn2, double sn3,
                        double sw) const {
  double sn[3] = {sn1, sn2, sn3};
  double tn[3] = {tn1, tn2, tn3};
  double r[3] = {r1, r2, r3};

  double scale = -3 / (4.0*M_PI);
  // int is_diag = (tgt_parity + src_parity + 1) % 2;//todo

  if (r1 == 0. && r2 == 0. && r3==0.) {
    ret->mat[mat_idx] = diag.get(tgt_parity, src_parity);//TODO
  } else {
    ret->mat[mat_idx] = sw * scale * (r1 * sn1 + r2 * sn2 + r3 * sn3) /
                        (pow(r1 * r1 + r2 * r2 + r3 * r3, 2.5))
                        * r[tgt_parity] * r[src_parity];
    if(mat_idx== 10 || mat_idx== 91 ||mat_idx== 172|| mat_idx== 253){
std::cout<<"weight dot tpar spar "<< sw << " "<< (r1 * sn1 + r2 * sn2 + r3 * sn3)
                     << " "<< r[tgt_parity] << " "<<r[src_parity]  << std::endl;
    }
                        // + sw * tn[tgt_parity] * sn[src_parity ];
  }

}

void Kernel::three_d_laplace(int mat_idx, ki_Mat* ret, double r1, double r2, double r3,
                        double diag,  
                       double sn1, double sn2, double sn3, 
                        double sw ) const {
  if (r1 == 0. && r2 == 0.&& r3 == 0.) {
    ret->mat[mat_idx] = diag;
  } else {
    ret->mat[mat_idx] =  -sw * (1.0 / (4 * M_PI)) *
                          (r1 * sn1 + r2 * sn2 + r3 * sn3) /
                          pow(r1 * r1 + r2 * r2 + r3 * r3, 1.5);
  }
}


double Kernel::kern_integral_3d(double phi_left, double phi_right, double theta_left, double theta_right, double phi, double theta) const{
  
  int quad_points = 5;
  
  double phis[quad_points];
  double phi_weights[quad_points];
  cgqf(quad_points, 1, 0.0, 0.0, phi_left, phi_right, phis, phi_weights);

 double thetas[quad_points];
  double theta_weights[quad_points];
  cgqf(quad_points, 1, 0.0, 0.0, theta_left, theta_right, thetas, theta_weights);

  double pole_x = 0.5 + 0.5 * sin(phi) * cos(theta);
  double pole_y = 0.5 + 0.5 * sin(phi) * sin(theta);
  double pole_z = 0.5 + 0.5 * cos(phi);
    
      // double sn1 = sin(phi) * cos(theta);
      // double sn2 =  sin(phi) * sin(theta);
      // double sn3 = cos(phi);
  
  double integral = 0.;

  for(int i=0; i<quad_points; i++){
    for(int j=0; j<quad_points; j++){
      double node_phi = phis[i];
      double node_theta = thetas[j];

      double node_x = 0.5 + 0.5 * sin(node_phi) * cos(node_theta);
      double node_y = 0.5 + 0.5 * sin(node_phi) * sin(node_theta);
      double node_z = 0.5 + 0.5 * cos(node_phi);

      double r1 = node_x - pole_x;
      double r2 = node_y - pole_y;
      double r3 = node_z - pole_z;

      double sn1 = sin(node_phi) * cos(node_theta);
      double sn2 =  sin(node_phi) * sin(node_theta);
      double sn3 = cos(node_phi);

      integral += -phi_weights[i]*theta_weights[j]*sin(node_phi)* pow(0.5,2)
                          *(1.0 / (4 * M_PI)) *
                          (r1 * sn1 + r2 * sn2 + r3 * sn3) /
                          pow(r1 * r1 + r2 * r2 + r3 * r3, 1.5);

    }
  }

  return integral;
}


void Kernel::compute_diag_entries_3dlaplace(Boundary* boundary){
  boundary_diags = std::vector<double>(boundary->weights.size());
  #pragma omp parallel for num_threads(8)
  for(int pt_idx =0; pt_idx < boundary->weights.size(); pt_idx++){
    double tp1 = boundary->points[3*pt_idx];
    double tp2 = boundary->points[3*pt_idx+1];
    double tp3 = boundary->points[3*pt_idx+2];    

    double rowsum = 0.0;
    for(int other_pt_idx = 0; other_pt_idx < boundary->weights.size(); other_pt_idx++){
      if(pt_idx == other_pt_idx) continue;

      double sp1 = boundary->points[3*other_pt_idx];
      double sp2 = boundary->points[3*other_pt_idx+1];
      double sp3 = boundary->points[3*other_pt_idx+2];
      double sn1 = boundary->normals[3*other_pt_idx];
      double sn2 = boundary->normals[3*other_pt_idx+1];
      double sn3 = boundary->normals[3*other_pt_idx+2];
      double sw = boundary->weights[other_pt_idx];

      double r1 = tp1-sp1;
      double r2 = tp2-sp2;
      double r3 = tp3-sp3;

      rowsum += (-sw * (1.0 / (4 * M_PI)) *
                                (r1 * sn1 + r2 * sn2 + r3 * sn3) /
                                pow(r1 * r1 + r2 * r2 + r3 * r3, 1.5));
    }
    boundary_diags[pt_idx] = 1-rowsum;
  }
  // double r = 0.5;
  // int num_circumf_points = (int) sqrt(boundary->weights.size()*2);
  // int num_phi_points = num_circumf_points/2;

  // double phis[num_phi_points];
  // double phi_weights[num_phi_points];
  // double phi_start = 0.;
  // double phi_end = M_PI;

  // cgqf(num_phi_points, 1, 0.0, 0.0, phi_start, phi_end, phis, phi_weights);

  // double phi_bounds[num_phi_points+1];
  // double sum = 0.;
  // phi_bounds[0] = 0.;
  // for(int i=0; i<num_phi_points; i++){
  //   sum += phi_weights[i];
  //   phi_bounds[i+1] = sum;
  // }

  // // Weight at north and south pole = 0?
  // for (int i = 0; i < num_circumf_points; i++) {
  //   double theta = 2 * M_PI * i * (1.0 / num_circumf_points);
  //   double theta_left =   2 * M_PI * (i-0.5)* (1.0 / num_circumf_points);
  //   double theta_right =  2 * M_PI * (i+0.5)* (1.0 / num_circumf_points);
  //   for (int j = 0; j < num_phi_points; j++) {   // modify this for annulus proxy
  //     double phi = phis[j]; 
  //     // diag entry will be sum of four integrals from theta bounds and phi bounds
  //     double phi_left = phi_bounds[j];
  //     double phi_right = phi_bounds[j+1];
  //     if(phi_left>phi || phi_right<phi){
  //       std::cout<<"riemann assumption violated"<<std::endl;
  //       exit(0);
  //     }

  //     double approx = 0.;
  //     approx += kern_integral_3d(phi_left, phi, theta_left, theta, phi, theta);
  //     approx += kern_integral_3d(phi_left, phi, theta, theta_right, phi, theta);
  //     approx += kern_integral_3d(phi, phi_right, theta_left, theta, phi, theta);
  //     approx += kern_integral_3d(phi, phi_right, theta, theta_right, phi, theta);

  //     // std::cout<<"approx is "<<approx<<std::endl;
  //     std::vector<double> pts;
  //     pts.push_back(0.5 + r * sin(phi) * cos(theta));
  //     pts.push_back(0.5 + r * sin(phi) * sin(theta));
  //     pts.push_back(0.5 + r * cos(phi));
  //     pt_to_diag_entry[pts] = approx;
  //   }
  // }
}


void Kernel::compute_diag_entries_3dstokes(Boundary* boundary){

  boundary_diag_tensors = std::vector<ki_Mat>(boundary->weights.size());
  #pragma omp parallel for num_threads(8)
  for(int pt_idx =0; pt_idx < boundary->weights.size(); pt_idx++){
    boundary_diag_tensors[pt_idx] = ki_Mat(3,3);
    for(int i=0; i<3; i++){
      for(int j=0;j<3;j++){
        boundary_diag_tensors[pt_idx].set(i,j, 1.0);
      }
    }
  }
}


ki_Mat Kernel::operator()(const std::vector<int>& tgt_inds,
                          const std::vector<int>& src_inds, bool forward) const {
  if(domain_dimension == 3){
    return op3d(tgt_inds, src_inds, forward);
  }
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


ki_Mat Kernel::op3d(const std::vector<int>& tgt_inds,
                          const std::vector<int>& src_inds, bool forward) const {

  ki_Mat ret(tgt_inds.size(), src_inds.size());
  int olda_ = tgt_inds.size();
  for (int j = 0; j < src_inds.size(); j++) {
    int src_ind = src_inds[j];
    int j_point_index = src_ind / solution_dimension;
    int j_points_vec_index = j_point_index * domain_dimension;

    double sp[3] = {boundary_points_[j_points_vec_index],
                    boundary_points_[j_points_vec_index + 1],
                    boundary_points_[j_points_vec_index + 2]
                   };
    double sn[3] =  {boundary_normals(j_points_vec_index),
                     boundary_normals(j_points_vec_index + 1),
                     boundary_normals(j_points_vec_index + 2)
                    };
    double sw =  boundary_weights(j_point_index);

    for (int i = 0; i < tgt_inds.size(); i++) {


      int tgt_ind = tgt_inds[i];

      int i_point_index = tgt_ind / solution_dimension;
      int i_points_vec_index = i_point_index * domain_dimension;
      double tp[3], tn[3];
      if(forward){
        tp[0] = domain_points[i_points_vec_index];
        tp[1] = domain_points[i_points_vec_index + 1];
        tp[2] = domain_points[i_points_vec_index + 2];

        tn[0] = 0;
        tn[1] = 0;
        tn[2] = 0;
      }else{
        tp[0] = boundary_points_[i_points_vec_index];
        tp[1] = boundary_points_[i_points_vec_index + 1];
        tp[2] = boundary_points_[i_points_vec_index + 2];

        tn[0] = boundary_normals(i_point_index);
        tn[1] = boundary_normals(i_point_index+1);
        tn[2] = boundary_normals(i_point_index+2);
      }
      double r[3] = {tp[0] - sp[0], tp[1] - sp[1], tp[2] - sp[2]};
 
      if(i==10&& j<9){
        std::cout<<"Expecting 0s here, r is "<<r[0]<<" "<<r[1]<<" "<<r[2]<<std::endl;
      }
      if(solution_dimension==1){
        three_d_laplace(i + olda_ * j, &ret, r[0], r[1],r[2], boundary_diags[j_point_index], 
          sn[0], sn[1], sn[2], sw);
      }      
      else{
        three_d_stokes(i + olda_ * j, tgt_ind % 3, src_ind % 3, &ret, r[0], r[1],r[2], boundary_diag_tensors[j_point_index],
          tn[0], tn[1], tn[2], sn[0], sn[1], sn[2], sw);
          if(i==10&& j<9){
            std::cout<<"put into mat "<<ret.mat[i+olda_*j]<<std::endl;
            std::cout<<i+olda_*j<<std::endl;
          }
       
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
  // if (node->level == 2) {
  //   for (QuadTreeNode* level_node : tree->levels[node->level]->nodes) {
  //     if (level_node != node) {
  //       for (int matrix_index :
  //            level_node->dof_lists.active_box) {
  //         outside_box.push_back(matrix_index);
  //       }
  //     }
  //   }
  //   for (int lvl = node->level - 1; lvl >= 0; lvl--) {
  //     for (QuadTreeNode* level_node : tree->levels[lvl]->nodes) {
  //       if (level_node->is_leaf) {
  //         for (int matrix_index :
  //              level_node->dof_lists.original_box) {
  //           outside_box.push_back(matrix_index);
  //         }
  //       }
  //     }
  //   }

  //   ki_Mat mat(2 * outside_box.size(), active_box.size());
  //   mat.set_submatrix(0, outside_box.size(), 0, active_box.size(),
  //                     (*this)(outside_box, active_box), false, true);
  //   mat.set_submatrix(outside_box.size(), 2 * outside_box.size(),
  //                     0, active_box.size(),
  //                     (*this)(active_box, outside_box), true, true);
  //   return mat;
  // }
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

// TODO(John) the general proxy stuff is just unreadable, replace with switch
// statements for kernels. It's not that bad. Or at least clean up the generality stuff

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
  r*=3;
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

  int lda = 2 * solution_dimension * pxy_w.size();

  ki_Mat ret(lda, box_inds.size());
  for (int j = 0; j < box_inds.size(); j++) {
    int box_ind = box_inds[j];
    int j_point_index = box_ind / solution_dimension;
    int j_points_vec_index = j_point_index * domain_dimension;

    double bp[3] = {boundary_points_[j_points_vec_index],
     boundary_points_[j_points_vec_index + 1],
     boundary_points_[j_points_vec_index + 2]};
    double bn[3] = {boundary_normals(j_points_vec_index),
     boundary_normals(j_points_vec_index + 1),
     boundary_normals(j_points_vec_index + 2)};

    double bw =  boundary_weights(j_point_index);
    for (int i = 0; i < pxy_p.size(); i += domain_dimension) {
      double pp[3] = {pxy_p[i],
        pxy_p[i + 1],
        pxy_p[i + 2]};
      double pn[3] = {pxy_n[i],
        pxy_n[i + 1],
        pxy_n[i + 2]};
    
      double r[3] = {pp[0] - bp[0], pp[1] - bp[1], pp[2]-bp[2]};
      if(solution_dimension == 1){
        three_d_laplace((i/domain_dimension)+lda*j, &ret, 
            r[0], r[1], r[2], 0.,
            bn[0], bn[1], bn[2],
            bw);
        three_d_laplace((pxy_p.size()/domain_dimension)+(i/domain_dimension)+lda*j, &ret, 
            -r[0], -r[1], -r[2], 0.,
            pn[0], pn[1], pn[2],
            pxy_w[i/domain_dimension]);
      }else{
       for (int pxy_parity = 0; pxy_parity < 3; pxy_parity++) {
          // box to pxy
          three_d_stokes(i + pxy_parity + lda * j, pxy_parity, box_ind % 3,
                     &ret, r[0], r[1], r[2], ki_Mat(3,3), pn[0], pn[1], pn[2],
                      bn[0], bn[1], bn[2], bw);

          // pxy to box
          three_d_stokes(pxy_p.size() + i + pxy_parity + lda * j,
                     box_ind % 3, pxy_parity, &ret, -r[0], -r[1],-r[2], ki_Mat(3,3),
                     bn[0], bn[1], bn[2], pn[0], pn[1], pn[2], pxy_w[i/domain_dimension]);
        }
      }
     
    }
  }
  return ret;
}


ki_Mat Kernel::forward() const {
  switch (solution_dimension) {
    case 1: {
      std::vector<int> tgt_inds(domain_points.size() / domain_dimension),
          src_inds(boundary_points_.size() / domain_dimension);
      for (int i = 0; i < domain_points.size() / domain_dimension; i++) tgt_inds[i] = i;
      for (int j = 0; j < boundary_points_.size() / domain_dimension; j++) src_inds[j] = j;
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
