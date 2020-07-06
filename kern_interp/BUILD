package(default_visibility = ["//visibility:public"])

load("//kern_interp:buildvars.bzl", "COPTS")

cc_library(
	name = "linear_solve",
	hdrs = ["linear_solve.h"],
  copts= COPTS,
	srcs = ["linear_solve.cpp"],
	deps = [":ki_mat",
			"//kern_interp/skel_factorization",
			"//kern_interp/quadtree",
			"//kern_interp/boundaries"],
  linkopts = ["-pthread"],

)


cc_library(
  name = "ki_mat",
  hdrs = ["ki_mat.h"],
  copts= COPTS,
  srcs = ["ki_mat.cpp",
    ],
  deps = [":pointvec"],
)



cc_library(
  name = "legendre",
  hdrs = ["legendre.h"],
    copts= COPTS,

  srcs = ["legendre.cpp",
    ],
)


cc_library(
	name = "pointvec",
	hdrs = ["pointvec.h"],
    copts= COPTS,

	srcs = ["pointvec.cpp"],
)


cc_test(
  name = "kern_interp_test",
  srcs = ["kern_interp_test.cpp"],
  copts= COPTS,
	deps = [
			":ki_mat",
			"//kern_interp/skel_factorization",
			"//kern_interp/quadtree",
			"//kern_interp/boundaries",
			":linear_solve",
			"@gtest//:gtest",
      "@gtest//:gtest_main"
    ],
  	linkopts = ["-llapacke",
  	        "-lopenblas",
			"-llapack",
			"-lblas",
			"-lgomp"]
)