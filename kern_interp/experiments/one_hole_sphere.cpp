// Copyright 2019 John Paul Ryan
#include <omp.h>
#include <string.h>
#include <fstream>
#include <memory>
#include <iostream>
#include <cmath>
#include <cassert>
#include "kern_interp/ki_mat.h"
#include "kern_interp/skel_factorization/skel_factorization.h"
#include "kern_interp/quadtree/quadtree.h"
#include "kern_interp/kernel/kernel.h"
#include "kern_interp/linear_solve.h"
#include "kern_interp/boundaries/sphere.h"
#include "kern_interp/boundaries/donut.h"

namespace kern_interp {


void run_one_hole_sphere() {
  // double start = omp_get_wtime();
  // srand(0);
  // std::unique_ptr<Boundary> boundary =
  //   std::unique_ptr<Boundary>(new Sphere());

  // Hole hole;
  // hole.center = PointVec(0.5,0.5,0.5);
  // hole.radius = 0.1;
  // boundary->holes.push_back(hole);  
  // boundary->initialize(pow(2,7),  BoundaryCondition::STOKES_3D_MIX);

  // QuadTree quadtree;
  // quadtree.initialize_tree(boundary.get(), 3, 3);
  // std::vector<double> old_domain_points, domain_points;
  // get_domain_points3d(20, &old_domain_points, quadtree.min,
  //                   quadtree.max);

  // for(int i=0; i<old_domain_points.size(); i+=3){
  //   if(boundary->is_in_domain(PointVec(old_domain_points[i],
  //                                     old_domain_points[i+1],
  //                                     old_domain_points[i+2] ))){
  //     domain_points.push_back(old_domain_points[i]);
  //     domain_points.push_back(old_domain_points[i+1]);
  //     domain_points.push_back(old_domain_points[i+2]);
  //   }
  // }

  // Kernel kernel(3, 3, Kernel::Pde::STOKES, boundary.get(), domain_points);
  // double cstart=omp_get_wtime();
  // kernel.compute_diag_entries_3dstokes(boundary.get());
  // double cend=omp_get_wtime();
  // std::cout<<"computer diag "<<(cend-cstart)<<std::endl;
  // ki_Mat sol = boundary_integral_solve(kernel, *(boundary.get()), &quadtree,
  //                                      1e-6, 8, domain_points);



  // std::ofstream sol_out;
  // sol_out.open("output/data/ie_solver_solution.txt");
  // int points_index = 0;
  // for (int i = 0; i < sol.height(); i += 3) {
  //   sol_out << domain_points[points_index] << "," <<
  //           domain_points[points_index + 1] << ","<<domain_points[points_index+2]<<",";
  //   points_index += 3;
  //   sol_out << sol.get(i, 0) << "," << sol.get(i + 1, 0)<<","<<sol.get(i+2,0)
  //           << std::endl;
  // }
  // sol_out.close();


  int num_threads = 8;
  double id_tol=1e-6;

  std::unique_ptr<Boundary> boundary3d =
    std::unique_ptr<Boundary>(new Sphere());
  // Hole hole3d;
  // hole3d.center = PointVec(0.5, 0.5, 0.5);
  // hole3d.radius = 0.1;
  // boundary3d->holes.push_back(hole3d);
  boundary3d->initialize(pow(2,5),  BoundaryCondition::STOKES_3D_MIX);

  QuadTree quadtree3d;
  quadtree3d.initialize_tree(boundary3d.get(), 3, 3);

  std::unique_ptr<Boundary> boundary2d =
    std::unique_ptr<Boundary>(new Donut());
      Hole hole2d;
  hole2d.center = PointVec(0.5, 0.5);
  hole2d.radius = 0.1;
  boundary2d->holes.push_back(hole2d);
  boundary2d->initialize(pow(2,11),  BoundaryCondition::STOKES_2D_MIX);
  QuadTree quadtree2d;
  quadtree2d.initialize_tree(boundary2d.get(), 3, 3);

  std::vector<double> domain_points2d, domain_points3d;
  int domain_size = 10;

  for (int i = 0; i < domain_size; i++) {
    double r = 0.12 + (0.86*(i/(domain_size + 0.)));
    for (int j = 0; j < domain_size; j++) {
      double theta = 2 * M_PI * (j / (domain_size+ 0.));
      // for (int k = 1; k < domain_size-1; k++) {
      //   double phi = M_PI * (i / (domain_size+0.));
      double phi = M_PI/2.0;
        double x = 0.5 + r*sin(phi)*cos(theta);
        double y = 0.5 + r*sin(phi)*sin(theta);
        double z = 0.5 + r*cos(phi);
        domain_points2d.push_back(x);
        domain_points2d.push_back(y);

        domain_points3d.push_back(x);
        domain_points3d.push_back(y);
        domain_points3d.push_back(z);
      // }
    }
  }

  // std::vector<double> old_domain_points, domain_points;
  // get_domain_points3d(10, &old_domain_points, quadtree.min,
  //                   quadtree.max);
  // TODO(John) get_domain_points needs to deal with this
  // for(int i=0; i<old_domain_points.size(); i+=3){
  //   if(boundary->is_in_domain(PointVec(old_domain_points[i],
  //                                     old_domain_points[i+1],
  //                                     old_domain_points[i+2] ))){
  //     domain_points.push_back(old_domain_points[i]);
  //     domain_points.push_back(old_domain_points[i+1]);
  //     domain_points.push_back(old_domain_points[i+2]);
  //   }
  // }

  Kernel kernel3d(3, 3, Kernel::Pde::STOKES, boundary3d.get(), domain_points3d);
  Kernel kernel2d(2, 2, Kernel::Pde::STOKES, boundary2d.get(), domain_points2d);
  // TODO(John) this should be part of kernel init
  kernel3d.compute_diag_entries_3dstokes(boundary3d.get());

  
  ki_Mat sol3d = boundary_integral_solve(kernel3d, *(boundary3d.get()), &quadtree3d,
                                       id_tol, num_threads, domain_points3d);

  ki_Mat sol2d = boundary_integral_solve(kernel2d, *(boundary2d.get()), &quadtree2d,
                                     id_tol, num_threads, domain_points2d);

  double err = 0.;
  for(int i=0; i<sol2d.height()/2; i++){
    err += pow(sol2d.get(2*i,0) - sol3d.get(3*i,0), 2) + pow(sol2d.get(2*i+1, 0) - sol3d.get(3*i+1,0),2);
    if(i<10){
      std::cout<<"2 : 3 "<<sol2d.get(2*i,0)<<" "<<sol2d.get(2*i+1,0)<<" : "<<sol3d.get(3*i,0)<<" "<<sol3d.get(3*i+1,0)<<std::endl;
    }
  }
  err=sqrt(err);
  std::cout<<"err "<<err<<std::endl;


  // std::ofstream sol_out;
  // sol_out.open("output/data/ie_solver_solution.txt");
  // int points_index = 0;
  // for (int i = 0; i < sol3d.height(); i += 3) {
  //   sol_out << domain_points2d[points_index] << "," <<
  //           domain_points2d[points_index + 1] << ",";
  //   points_index += 2;
  //   sol_out << sol3d.get(i, 0) << "," << sol3d.get(i + 1, 0)
  //           << std::endl;
  // }
  // sol_out.close();
  // std::ofstream bound_out;
  // bound_out.open("output/data/ie_solver_boundary.txt");
  // for (int i = 0; i < boundary2d->points.size(); i += 2) {
  //   bound_out << boundary2d->points[i] << "," << boundary2d->points[i + 1]
  //             << std::endl;
  // }
  // bound_out.close();

 
 // conclusion of last timing test - lvl 2 has some boxes that take forever
}

}  // namespace kern_interp


int main(int argc, char** argv) {
  srand(0);
  openblas_set_num_threads(1);
  kern_interp::run_one_hole_sphere();
  return 0;
}

