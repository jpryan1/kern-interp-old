package(default_visibility = ["//visibility:public"])
load("//kern_interp:buildvars.bzl", "COPTS")


cc_library(
	name = "kernel",
	hdrs = ["kernel.h"],
	srcs = ["kernel.cpp"],
  copts= COPTS,

	deps = ["//kern_interp:ki_mat",
          "//kern_interp/boundaries",
					"//kern_interp/quadtree",
          "//kern_interp:legendre"]
)
