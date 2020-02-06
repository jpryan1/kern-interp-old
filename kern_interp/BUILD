package(default_visibility = ["//visibility:public"])

cc_library(
	name = "linear_solve",
	hdrs = ["linear_solve.h"],
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
	srcs = ["ki_mat.cpp",
		],
	deps = [":vec2"],
)


cc_library(
	name = "vec2",
	hdrs = ["vec2.h"],
	srcs = ["vec2.cpp"],
)


cc_test(
  name = "kern_interp_test",
  srcs = ["kern_interp_test.cpp"],
  copts= ["-std=c++14"],
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
			"-llapack",
			"-lblas",
			"-lgomp"]
)