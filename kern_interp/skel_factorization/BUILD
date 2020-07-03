package(default_visibility = ["//visibility:public"])
load("//kern_interp:buildvars.bzl", "COPTS")

cc_library(
	name = "skel_factorization",
	hdrs = ["skel_factorization.h"],
	srcs = ["skel_factorization.cpp"],
  copts= COPTS,
	deps = ["//kern_interp:ki_mat",
			"//kern_interp/quadtree",
			"//kern_interp/kernel"]
)