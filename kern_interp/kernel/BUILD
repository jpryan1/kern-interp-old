package(default_visibility = ["//visibility:public"])


cc_library(
	name = "kernel",
	hdrs = ["kernel.h"],
	srcs = ["kernel.cpp"],
	copts = ["-fopenmp"],

	deps = ["//kern_interp:ki_mat",
          "//kern_interp/boundaries",
					"//kern_interp/quadtree",
          "//kern_interp:legendre"]
)
