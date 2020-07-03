// Copyright 2019 John Paul Ryan
#ifndef KERN_INTERP_QUADTREE_QUADTREE_H_
#define KERN_INTERP_QUADTREE_QUADTREE_H_

#include <cassert>
#include <vector>
#include "kern_interp/ki_mat.h"
#include "kern_interp/boundaries/boundary.h"

#define MAX_LEAF_DOFS 128

namespace kern_interp {

struct InteractionLists {
  std::vector<int> original_box,
      active_box,
      redundant,
      skel,
      near,
      skelnear,
      permutation;

  void set_rs_ranges(const std::vector<int>& prm, int sk,
                     int rd) {
    assert(prm.size() == sk + rd);

    for (int i = 0; i < sk; i++) {
      skel.push_back(active_box[prm[i]]);
      permutation.push_back(prm[i]);
    }
    for (int i = sk; i < sk + rd; i++) {
      redundant.push_back(active_box[prm[i]]);
      permutation.push_back(prm[i]);
    }
  }

  void set_skelnear_range() {
    for (int i = 0; i < skel.size(); i++) {
      skelnear.push_back(skel[i]);
    }
  }
};

struct QuadTreeNode {
  int level, dofs_below;
  bool is_leaf, X_rr_is_LU_factored = false, compressed = false;
  double side_length;
  QuadTreeNode* parent;
  std::vector<QuadTreeNode*> children;
  // QuadTreeNode* children[4];
  std::vector<QuadTreeNode*> neighbors;
  InteractionLists dof_lists;
  // For inverse operator
  ki_Mat T, L, U, X_rr, schur_update, X_rr_lu;
  std::vector<lapack_int> X_rr_piv;
  // format is {BL, TL, TR, BR}
  // double corners[8];

  std::vector<double> center;
  QuadTreeNode() {
    is_leaf = true;
  }
};


struct QuadTreeLevel {
  std::vector<QuadTreeNode*> nodes;
  ~QuadTreeLevel() {
    for (QuadTreeNode* node : nodes) {
      delete node;
    }
  }
};


class QuadTree {
 public:
  int solution_dimension, domain_dimension;
  int no_proxy_level = 0;
  double min, max;
  std::vector<double> boundary_points;
  QuadTreeNode* root;
  ki_Mat allskel_mat, allskel_mat_lu, U, Psi, S_LU;
  std::vector<lapack_int> allskel_mat_piv, S_piv;
  std::vector<QuadTreeLevel*> levels;
  ~QuadTree();
  void initialize_tree(Boundary* boundary, int solution_dimension_,
                       int domain_dimension_);
  void compute_neighbor_lists();
  void recursive_add(QuadTreeNode* node, std::vector<double> pt,
                     int mat_ind);
  void get_descendent_neighbors(QuadTreeNode* big, QuadTreeNode* small);
  void node_subdivide(QuadTreeNode* node);
  void consolidate_node(QuadTreeNode* node);
  void reset();
  void reset(Boundary* boundary_);
  void copy_into(QuadTree* new_tree) const;
  void mark_neighbors_and_parents(QuadTreeNode* node);
  void perturb(const Boundary& new_boundary);

  void remove_inactive_dofs_at_level(int level);
  void remove_inactive_dofs_at_all_boxes();
  void remove_inactive_dofs_at_box(QuadTreeNode* node);
};

}  // namespace kern_interp

#endif  // KERN_INTERP_QUADTREE_QUADTREE_H_
