package(default_visibility = ["//visibility:public"])

cc_library(
	name = "boundaries",
	hdrs = [
			"circle.h",
			"annulus.h",
			"uniformdist.h",
			"donut.h",
			"cubic_spline.h",
			"ex1boundary.h",
			"ex2boundary.h",
			"ex3boundary.h",
			"boundary.h"],
	copts= ["-std=c++14"],
	srcs = [
			"circle.cpp",
			"annulus.cpp",
			"uniformdist.cpp",
			"donut.cpp",
			"cubic_spline.cpp",
			"boundary.cpp",
			"cubic_boundary.cpp",
			"ex1boundary.cpp",
			"ex2boundary.cpp",
			"ex3boundary.cpp",
			],
	deps = ["//kern_interp:ki_mat",
			"//kern_interp:vec2"]
)
