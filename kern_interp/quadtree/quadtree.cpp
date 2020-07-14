// Copyright 2019 John Paul Ryan
#include <omp.h>
#include <fstream>
#include <cassert>
#include <algorithm>
#include <cmath>
#include <unordered_map>
#include <iostream>
#include <utility>
#include <boost/functional/hash.hpp>
#include "kern_interp/quadtree/quadtree.h"

namespace kern_interp {

// typedef std::pair<double, double> pair;

QuadTree::~QuadTree() {
  for (QuadTreeLevel* level : levels) {
    if (level) {
      delete level;
    }
  }
  levels.clear();
}


void QuadTree::compute_neighbor_lists() {
  for (int level = 0; level < levels.size(); level++) {
    QuadTreeLevel* current_level = levels[level];
    for (int k = 0; k < current_level->nodes.size(); k++) {
      QuadTreeNode* node_a = current_level->nodes[k];
      // Each node is neighbors with all its siblings
      if (node_a->parent != nullptr) {
        for (QuadTreeNode* sibling : node_a->parent->children) {
          if (sibling != node_a) node_a->neighbors.push_back(sibling);
        }
        // Now check all parents' neighbors' children
        for (QuadTreeNode* parents_neighbor : node_a->parent->neighbors) {
          for (QuadTreeNode* cousin : parents_neighbor->children) {
            if (cousin == nullptr) continue;
            if (cousin->level != node_a->level) continue;
            double dist = 0.;
            for (int d = 0; d < domain_dimension; d++) {
              dist += pow(node_a->center[d] - cousin->center[d], 2);
            }
            dist = sqrt(dist);
            // just need to check if the distance of the bl corners
            // is <=s*sqrt(2)
            if (dist < node_a->side_length * sqrt(domain_dimension) + 1e-5) {
              node_a->neighbors.push_back(cousin);
            }
          }
        }
      }
      // now if it is a leaf, check against nodes in all subsequent levels
      if (node_a->is_leaf) {
        for (int n = 0; n < node_a->neighbors.size(); n++) {
          QuadTreeNode* neighbor =  node_a->neighbors[n];
          // make sure this isn't a neighbor from a higher level
          if (neighbor->level != node_a->level) {
            continue;
          }
          for (QuadTreeNode* child : neighbor->children) {
            if (child != nullptr) {
              get_descendent_neighbors(node_a, child);
            }
          }
        }
      }
    }
  }
}


void QuadTree::initialize_tree(Boundary* boundary,
                               int solution_dimension_,
                               int domain_dimension_) {
  assert(boundary->points.size() > 0
         && "number of boundary_points to init tree cannot be 0.");
  this->boundary_points = boundary->points;
  this->solution_dimension = solution_dimension_;
  this->domain_dimension = domain_dimension_;
  min = boundary_points[0];
  max = boundary_points[0];

  for (double point : boundary_points) {
    if (point < min) min = point;
    if (point > max) max = point;
  }

  double tree_min = min - 0.01;
  double tree_max = max + 0.01;

  root = new QuadTreeNode();

  root->level = 0;
  root->parent = nullptr;

  for (int i = 0; i < domain_dimension; i++) {
    root->center.push_back((tree_min + tree_max) / 2.0);
  }
  root->side_length = tree_max - tree_min;
  QuadTreeLevel* level_one = new QuadTreeLevel();
  level_one->nodes.push_back(root);
  levels.push_back(level_one);

  for (int i = 0; i < boundary_points.size(); i += domain_dimension) {
    std::vector<double> pt;
    for (int j = i; j < i + domain_dimension; j++) {
      pt.push_back(boundary_points[j]);
    }
    recursive_add(this->root, pt, i / domain_dimension);
  }
  compute_neighbor_lists();
  sort_leaves();

}


// adds neighbors to leaf which are on lower levels, by recursing and checking
// if the corners are along the wall.
void QuadTree::get_descendent_neighbors(QuadTreeNode* big,
                                        QuadTreeNode* small) {
  assert(big->level < small->level);
  bool are_neighbors = true;
  for (int d = 0; d < domain_dimension; d++) {
    double dist =  abs(big->center[d] - small->center[d]);
    double pred_dist = (big->side_length + small->side_length) / 2.;
    if (dist - pred_dist > 1e-14) {
      are_neighbors = false;
      break;
    }
  }
  if (are_neighbors) {
    big->neighbors.push_back(small);
    small->neighbors.push_back(big);
  }
  for (QuadTreeNode* child : small->children) {
    if (child != nullptr) {
      get_descendent_neighbors(big, child);
    }
  }
}


void QuadTree::recursive_add(QuadTreeNode* node, std::vector<double> pt,
                             int point_ind) {
  assert(node != nullptr && "recursive_add fails on null node.");
  for (int i = 0; i < solution_dimension; i++) {
    node->dof_lists.original_box.push_back(solution_dimension * point_ind + i);
  }
  // // figure out which child
  if (!node->is_leaf) {
    int child_idx = 0;
    for (int i = 0; i < domain_dimension; i++) {
      if (pt[i] >= node->center[i]) {
        child_idx += pow(2, domain_dimension - i - 1);
      }
    }
    recursive_add(node->children[child_idx], pt, point_ind);
  } else {
    // do we need one?
    // if this node is exploding and needs children

    if (node->is_leaf
        && node->dof_lists.original_box.size() > MAX_LEAF_DOFS) {
      node_subdivide(node);
    }
  }
}


// 1) gives node its four children
// 2) puts these children in their proper level
// 3) gives these children their corners
void QuadTree::node_subdivide(QuadTreeNode* node) {
  assert(node != nullptr && "node_subdivide fails on null node.");
  node->is_leaf = false;
  for (int child_idx = 0; child_idx < pow(2, domain_dimension); child_idx++) {
    std::vector<double> child_center;
    int tmp = child_idx;
    for (int d = 0; d < domain_dimension; d++) {
      if (tmp >= pow(2, domain_dimension - d - 1)) {
        child_center.push_back(node->center[d] + (node->side_length / 4.));
        tmp -= pow(2, domain_dimension - d - 1);
      } else {
        child_center.push_back(node->center[d] - (node->side_length / 4.));
      }
    }
    assert(tmp == 0);
    QuadTreeNode* child = new QuadTreeNode();
    child->center = child_center;
    node->children.push_back(child);
  }


  for (QuadTreeNode* child : node->children) {
    child->level = node->level + 1;
    child->side_length = node->side_length / 2.0;
    child->parent = node;
  }
  if (levels.size() < node->level + 2) {
    QuadTreeLevel* new_level = new QuadTreeLevel();
    levels.push_back(new_level);
    for (QuadTreeNode* child : node->children) {
      new_level->nodes.push_back(child);
    }
  } else {
    for (QuadTreeNode* child : node->children) {
      levels[node->level + 1]->nodes.push_back(child);
    }
  }
  // Now we bring the indices from the parent's box down into its childrens
  // boxes
  for (int index = 0; index < node->dof_lists.original_box.size();
       index += solution_dimension) {
    int matrix_index = node->dof_lists.original_box[index];
    int points_vec_index = (matrix_index / solution_dimension) *
                           domain_dimension;

    // double x = boundary_points[points_vec_index];
    // double y = boundary_points[points_vec_index + 1];

    int child_idx = 0;
    for (int i = 0; i < domain_dimension; i++) {
      if (boundary_points[points_vec_index + i] >= node->center[i]) {
        child_idx += pow(2, domain_dimension - i - 1);
      }
    }
    for (int i = 0; i < solution_dimension; i++) {
      node->children[child_idx]->dof_lists.original_box.push_back(matrix_index + i);
    }
  }
  for (QuadTreeNode* child : node->children) {
    if (child->dof_lists.original_box.size()  > MAX_LEAF_DOFS) {
      node_subdivide(child);
    }
  }
}


void QuadTree::mark_neighbors_and_parents(QuadTreeNode * node) {
  if (node == nullptr) return;
  if (node->p_marked) return;
  node->p_marked = true;
  node->compressed = false;
  node->X_rr_is_LU_factored = false;
  node->dof_lists.active_box.clear();
  node->dof_lists.skel.clear();
  node->dof_lists.skelnear.clear();
  node->dof_lists.redundant.clear();
  node->dof_lists.permutation.clear();
  node->T = ki_Mat(0, 0);
  node->U = ki_Mat(0, 0);
  node->L = ki_Mat(0, 0);

  for (QuadTreeNode* neighbor : node->neighbors) {
    neighbor->compressed = false;
    neighbor->X_rr_is_LU_factored = false;
    neighbor->dof_lists.active_box.clear();
    neighbor->dof_lists.skel.clear();
    neighbor->dof_lists.skelnear.clear();
    neighbor->dof_lists.redundant.clear();
    neighbor->dof_lists.permutation.clear();
    neighbor->T = ki_Mat(0, 0);
    neighbor->U = ki_Mat(0, 0);
    neighbor->L = ki_Mat(0, 0);
    mark_neighbors_and_parents(neighbor->parent);
  }
  mark_neighbors_and_parents(node->parent);
}


void QuadTree::consolidate_node(QuadTreeNode* node) {
  // Need to
  //  Move leaf child dofs into my original box
  //  erase all descendents from levels
  //  delete immediate descentdents
  node->dof_lists.original_box.clear();
  std::vector<QuadTreeNode*> remove_from_lvl;
  std::vector<QuadTreeNode*> queue;
  queue.push_back(node);
  for (int i = 0; i < queue.size(); i++) {
    QuadTreeNode* current = queue[i];
    if (current->is_leaf) {
      node->dof_lists.original_box.insert(
        node->dof_lists.original_box.end(),
        current->dof_lists.original_box.begin(),
        current->dof_lists.original_box.end());
    } else {
      for (QuadTreeNode* child : current->children) {
        queue.push_back(child);
      }
    }
    if (current != node) {
      remove_from_lvl.push_back(current);
    }
  }
  for (QuadTreeNode* erase : remove_from_lvl) {
    QuadTreeLevel* erase_level = levels[erase->level];
    for (int i = 0; i < erase_level->nodes.size(); i++) {
      if (erase_level->nodes[i] == erase) {
        QuadTreeNode* del = erase_level->nodes[i];
        erase_level->nodes.erase(erase_level->nodes.begin() + i);
        delete del;
        break;
      }
    }
  }
  node->children.clear();
  node->is_leaf = true;
}


void QuadTree::sort_leaves() {
  return; //seemingly unnecessary?
  for (int level = 0; level < levels.size(); level++) {
    QuadTreeLevel* current_level = levels[level];
    for (int k = 0; k < current_level->nodes.size(); k++) {
      QuadTreeNode* node = current_level->nodes[k];
      if ((!node->is_leaf) || node->compressed) continue;
      std::vector<std::pair<double, int>> origboxtmp;
      for (int i = 0; i < node->dof_lists.original_box.size(); i++) {
        int points_vec_index = domain_dimension * ( node->dof_lists.original_box[i] /
                               solution_dimension);
        double x = boundary_points[points_vec_index];
        double y = boundary_points[points_vec_index + 1];
        double dist = 0.;
        for (int d = 0; d < domain_dimension; d++) {
          dist += pow(node->center[d] - boundary_points[points_vec_index + d], 2);
        }
        dist = sqrt(dist);
        origboxtmp.push_back(std::pair<double, int>(dist,
                                node->dof_lists.original_box[i]));
      }
      std::sort(origboxtmp.begin(),
                origboxtmp.end());
       node->dof_lists.original_box.clear();
      for (int i = 0; i < origboxtmp.size(); i++) {
         node->dof_lists.original_box.push_back(origboxtmp[i].second);
      }
    }
  }
}



void QuadTree::perturb(const Boundary & perturbed_boundary) {
  // 1) create mapping, storing vectors of additions/deletions
  // 2) go to every node, marking those with additions and deletions
  // these are vectors of point indices (p_0, p_1, etc)
  std::vector<double> additions;
  std::vector<double> deletions;

  std::vector<double> old_points = boundary_points;
  std::vector<double> new_points = perturbed_boundary.points;
  // now create mapping of new_points to their point index in the new vec
  std::unordered_map<std::vector<double>, int, boost::hash<std::vector<double>>>
  point_to_new_index;
  for (int i = 0; i < new_points.size(); i += domain_dimension) {
    std::vector<double> new_point;
    for (int j = 0; j < domain_dimension; j++) {
      new_point.push_back(new_points[i + j]);
    }
    point_to_new_index[new_point] = i / domain_dimension;
  }

  std::vector<bool> found_in_old(new_points.size() / domain_dimension);
  for (int i = 0; i < found_in_old.size(); i++) {
    found_in_old[i] = false;
  }
  // Mapping from point index in old points vec to point index in new points vec
  std::unordered_map<int, int> old_index_to_new_index;
  for (int i = 0; i < old_points.size(); i += domain_dimension) {
    std::vector<double> old_point;
    for (int j = 0; j < domain_dimension; j++) {
      old_point.push_back(old_points[i + j]);
    }
    // Is this point also in the new points vec?
    std::unordered_map<std::vector<double>, int, boost::hash<std::vector<double>>>::const_iterator
    element =
      point_to_new_index.find(old_point);
    if (element != point_to_new_index.end()) {
      old_index_to_new_index[i / domain_dimension] = element->second;
      found_in_old[element->second] = true;
    } else {
      // If it's in the old vec but not the new vec, it was deleted
      deletions.push_back(i / domain_dimension);
    }
  }
  for (int i = 0; i < found_in_old.size(); i++) {
    if (!found_in_old[i]) {
      additions.push_back(i);
    }
  }

  // go through all leaf original box vectors and apply mapping.
  // (if there is a deletion it will be processed later)
  // each node will be one of three things
  //   1) unmarked, in which case the below is a perfectly good mapping
  //   2) marked non-leaf, the below is irrelevant, everything will be dumped
  //   3) marked leaf, only the leaf portion of the below is relevant.
  for (QuadTreeLevel* level : levels) {
    for (QuadTreeNode* node : level->nodes) {
      std::vector<int> ob, ab, s, r, sn, n;
      if (node->is_leaf) {
        for (int idx : node->dof_lists.original_box) {
          int point_index = idx / solution_dimension;
          std::unordered_map<int, int>::const_iterator element =
            old_index_to_new_index.find(point_index);
          if (element != old_index_to_new_index.end()) {
            ob.push_back(solution_dimension * element->second
                         + idx % solution_dimension);
          }
        }
        node->dof_lists.original_box = ob;
      }
      for (int idx : node->dof_lists.active_box) {
        int point_index = idx / solution_dimension;
        std::unordered_map<int, int>::const_iterator element =
          old_index_to_new_index.find(point_index);
        if (element != old_index_to_new_index.end()) {
          ab.push_back(solution_dimension * element->second
                       + idx % solution_dimension);
        }
      }
      for (int idx : node->dof_lists.skel) {
        int point_index = idx / solution_dimension;
        std::unordered_map<int, int>::const_iterator element =
          old_index_to_new_index.find(point_index);
        if (element != old_index_to_new_index.end()) {
          s.push_back(solution_dimension * element->second
                      + idx % solution_dimension);
        }
      }
      for (int idx : node->dof_lists.redundant) {
        int point_index = idx / solution_dimension;
        std::unordered_map<int, int>::const_iterator element =
          old_index_to_new_index.find(point_index);
        if (element != old_index_to_new_index.end()) {
          r.push_back(solution_dimension * element->second
                      + idx % solution_dimension);
        }
      }
      for (int idx : node->dof_lists.skelnear) {
        int point_index = idx / solution_dimension;
        std::unordered_map<int, int>::const_iterator element =
          old_index_to_new_index.find(point_index);
        if (element != old_index_to_new_index.end()) {
          sn.push_back(solution_dimension * element->second
                       + idx % solution_dimension);
        }
      }
      node->dof_lists.active_box = ab;
      node->dof_lists.skel = s;
      node->dof_lists.skelnear = sn;
      node->dof_lists.redundant = r;
    }
  }

  // go through all additions, find their leaves, make addition and call mark
  // function
  std::vector<QuadTreeNode*> maybe_bursting;
  for (int i = 0; i < additions.size(); i++) {
    // double newx = new_points[2 * additions[i]];
    // double newy = new_points[2 * additions[i] + 1];
    std::vector<double> newpt;
    for (int j = 0; j < domain_dimension; j++) {
      newpt.push_back(new_points[domain_dimension * additions[i] + j]);
    }
    QuadTreeNode* current = root;
    while (!current->is_leaf) {
      int child_idx = 0;
      for (int j = 0; j < domain_dimension; j++) {
        if (newpt[j] >= current->center[j]) {
          child_idx += pow(2, domain_dimension - j - 1);
        }
      }
      current = current->children[child_idx];

    }
    for (int j = 0; j < solution_dimension; j++) {
      current->dof_lists.original_box.push_back(solution_dimension
          * additions[i] + j);
    }
    maybe_bursting.push_back(current);
    mark_neighbors_and_parents(current);
  }

  for (QuadTreeLevel* level : levels) {
    for (QuadTreeNode* node : level->nodes) {
      if (node->is_leaf) {
        node->dofs_below = node->dof_lists.original_box.size();
      } else {
        node->dofs_below = 0;
      }
    }
  }
  for (int l = levels.size() - 1; l >= 1; l--) {
    QuadTreeLevel* level = levels[l];
    for (QuadTreeNode* node : level->nodes) {
      node->parent->dofs_below += node->dofs_below;
    }
  }

  // go through all deletions, find their leaves, make deletion and call mark
  // function
  std::unordered_map<QuadTreeNode*, bool> sparse;

  for (int i = 0; i < deletions.size(); i++) {
    QuadTreeNode* current = root;
    bool path_marked = false;

    std::vector<double> oldpt;
    for (int j = 0; j < domain_dimension; j++) {
      oldpt.push_back(old_points[domain_dimension * deletions[i] + j]);
    }

    while (!current->is_leaf) {
      if (current->dofs_below < MAX_LEAF_DOFS && !path_marked) {
        path_marked = true;
        sparse[current] = true;
      }

      int child_idx = 0;
      for (int j = 0; j < domain_dimension; j++) {
        if (oldpt[j] >= current->center[j]) {
          child_idx += pow(2, domain_dimension - j - 1);
        }
      }
      current = current->children[child_idx];
    }
    mark_neighbors_and_parents(current);
  }

  this->boundary_points = perturbed_boundary.points;
  // If any nodes are bursting now, subdivide them.
  for (QuadTreeNode* node : maybe_bursting) {
    if (node->is_leaf
        && node->dof_lists.original_box.size() > MAX_LEAF_DOFS) {
      node_subdivide(node);
    //TODO Is this necessary?
      for (QuadTreeNode* child : node->children) mark_neighbors_and_parents(child);
    }
  }
  // If we can consolidate nodes into their parent, do that.
  for (auto it = sparse.begin(); it != sparse.end(); ++it) {
    consolidate_node(it->first);
  }
  for (QuadTreeLevel* level : levels) {
    for (QuadTreeNode* node : level->nodes) {
      node->p_marked = false;
      node->neighbors.clear();
    }
  }
  compute_neighbor_lists();

  sort_leaves();
}


void copy_info(QuadTreeNode* old_node, QuadTreeNode* new_node) {
  new_node->level = old_node->level;
  new_node->dofs_below = old_node->dofs_below;
  new_node->is_leaf = old_node->is_leaf;
  new_node->X_rr_is_LU_factored = old_node->X_rr_is_LU_factored;
  new_node->compressed = old_node->compressed;
  new_node->side_length = old_node->side_length;
  new_node->dof_lists = old_node->dof_lists;
  new_node->T = old_node->T;
  new_node->L = old_node->L;
  new_node->U = old_node->U;
  new_node->X_rr = old_node->X_rr;
  new_node->schur_update = old_node->schur_update;
  new_node->X_rr_lu = old_node->X_rr_lu;
  new_node->X_rr_piv = old_node->X_rr_piv;
  new_node->center = old_node->center;
  // for (int i = 0; i < 8; i++) {
  //   new_node->corners[i] = old_node->corners[i];
  // }
}


void QuadTree::copy_into(QuadTree* new_tree) const {
  // The strategy here is going to be to create a new node for every old node,
  // then keep a mapping from new to old. With that, we'll copy all the data
  // over, including connections, levels, and matrices.
  *new_tree = QuadTree();
  std::vector < QuadTreeNode*> new_nodes;
  std::unordered_map<QuadTreeNode*, QuadTreeNode*> old_to_new;
  std::unordered_map<QuadTreeNode*, QuadTreeNode*> new_to_old;
  for (int lvl = 0; lvl < levels.size(); lvl++) {
    QuadTreeLevel* level = levels[lvl];
    for (int n = 0; n < level->nodes.size(); n++) {
      QuadTreeNode* old_node = level->nodes[n];
      QuadTreeNode* new_node = new QuadTreeNode();
      new_nodes.push_back(new_node);
      copy_info(old_node, new_node);
      old_to_new[old_node] = new_node;
      new_to_old[new_node] = old_node;
    }
  }

  for (int n = 0; n < new_nodes.size(); n++) {
    QuadTreeNode* new_node = new_nodes[n];
    new_node->parent = old_to_new[new_to_old[new_node]->parent];
    if (!new_to_old[new_node]->is_leaf) {
      for (int c = 0; c < pow(2, domain_dimension); c++) {
        new_node->children.push_back(old_to_new[new_to_old[new_node]->children[c]]);
      }
    }
    for (int nbr = 0; nbr < new_to_old[new_node]->neighbors.size(); nbr++) {
      QuadTreeNode* neighbor = new_to_old[new_node]->neighbors[nbr];
      new_node->neighbors.push_back(old_to_new[neighbor]);
    }
  }

  new_tree->root = old_to_new[root];
  new_tree->solution_dimension = solution_dimension;
  new_tree->domain_dimension = domain_dimension;
  new_tree->no_proxy_level = no_proxy_level;
  new_tree->min = min;
  new_tree->max = max;
  new_tree->allskel_mat = allskel_mat;
  new_tree->allskel_mat_lu = allskel_mat_lu;
  new_tree->U = U;
  new_tree->Psi = Psi;
  new_tree->S_LU = S_LU;
  new_tree->allskel_mat_piv = allskel_mat_piv;
  new_tree->S_piv = S_piv;

  for (int lvl = 0; lvl < levels.size(); lvl++) {
    QuadTreeLevel* old_level = levels[lvl];
    QuadTreeLevel* new_level = new QuadTreeLevel();
    for (int n = 0; n < old_level->nodes.size(); n++) {
      new_level->nodes.push_back(old_to_new[old_level->nodes[n]]);
    }
    new_tree->levels.push_back(new_level);
  }
}


void QuadTree::reset(Boundary * boundary_) {
  for (QuadTreeLevel* level : levels) {
    if (level) {
      delete level;
    }
  }
  levels.clear();
  initialize_tree(boundary_, solution_dimension,
                  domain_dimension);
}


void QuadTree::remove_inactive_dofs_at_all_boxes() {
  int lvls = levels.size();
  for (int level = lvls - 1; level >= 0; level--) {
    remove_inactive_dofs_at_level(level);
  }
}


void QuadTree::remove_inactive_dofs_at_level(int level) {
  QuadTreeLevel* current_level = levels[level];
  // First, get all active dofs from children
  for (QuadTreeNode* node : current_level->nodes) {
    if (node->compressed) continue;
    remove_inactive_dofs_at_box(node);
  }
  // Next, get all active near dofs from neighbors
  for (QuadTreeNode* node_a : current_level->nodes) {
    node_a->dof_lists.near.clear();
    for (QuadTreeNode* neighbor : node_a->neighbors) {
      // Some neighbors are smaller boxes from higher levels, we don't
      // care about those, their parents have the updated information.
      if (neighbor->level > node_a->level) {
        continue;
      }
      if (neighbor->is_leaf) {
        for (int idx : neighbor->dof_lists.original_box) {
          node_a->dof_lists.near.push_back(idx);
        }
      } else {
        for (int idx : neighbor->dof_lists.active_box) {
          node_a->dof_lists.near.push_back(idx);
        }
      }
    }
  }
}


void QuadTree::remove_inactive_dofs_at_box(QuadTreeNode* node) {
  // this function removes from the box any DoFs which have already been made
  // redundant. It involves a bunch of annoying C++ functions and probably
  // would look nicer in matlab.

  // populate active_box
  node->dof_lists.skel.clear();
  node->dof_lists.skelnear.clear();
  node->dof_lists.redundant.clear();
  node->dof_lists.active_box.clear();

  if (!node->is_leaf) {
    for (QuadTreeNode* child : node->children) {
      if (child->compressed) {
        for (int i : child->dof_lists.skel) {
          node->dof_lists.active_box.push_back(i);
        }
      } else {
        for (int i : child->dof_lists.active_box) {
          node->dof_lists.active_box.push_back(i);
        }
      }
    }
  } else {
    node->dof_lists.active_box = node->dof_lists.original_box;
  }
}

}  // namespace kern_interp
