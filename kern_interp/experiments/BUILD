package(default_visibility = ["//visibility:public"])

cc_binary(
  name = "experiment_one",
  srcs = ["experiment_one.cpp"],
  copts= ["-fopenmp",
          "-std=c++14"],
  deps = [
      "//kern_interp:ki_mat",
      "//kern_interp/skel_factorization",
      "//kern_interp/quadtree",
      "//kern_interp/boundaries",
      "//kern_interp:linear_solve",
      ],
  linkopts = ["-llapacke",
        "-llapack",
        "-lblas",
        "-lgomp"]

)


cc_binary(
  name = "experiment_two",
  srcs = ["experiment_two.cpp"],
  copts= ["-fopenmp",
          "-std=c++14"],
  deps = [
      "//kern_interp:ki_mat",
      "//kern_interp/skel_factorization",
      "//kern_interp/quadtree",
      "//kern_interp/boundaries",
      "//kern_interp:linear_solve",
      ],
  linkopts = ["-llapacke",
        "-llapack",
        "-lblas",
        "-lgomp"]

)


cc_binary(
  name = "experiment_three",
  srcs = ["experiment_three.cpp"],
  copts= ["-fopenmp",
          "-std=c++14"],
  deps = [
      "//kern_interp:ki_mat",
      "//kern_interp/skel_factorization",
      "//kern_interp/quadtree",
      "//kern_interp/boundaries",
      "//kern_interp:linear_solve",
      ],
 linkopts = ["-llapacke",
        "-llapack",
        "-lblas",
        "-lgomp"]

)

cc_binary(
  name = "covar",
  srcs = ["covar.cpp"],
  copts= ["-fopenmp",
          "-std=c++14"],
  deps = [
      "//kern_interp:ki_mat",
      "//kern_interp/skel_factorization",
      "//kern_interp/quadtree",
      "//kern_interp/boundaries",
      "//kern_interp:linear_solve",
      ],
 linkopts = ["-llapacke",
        "-llapack",
        "-lblas",
        "-lgomp"]

)
