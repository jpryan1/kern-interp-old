package(default_visibility = ["//visibility:public"])
load("//kern_interp:buildvars.bzl", "COPTS")

cc_library(
  name = "quadtree",
  hdrs = ["quadtree.h"],
  srcs = ["quadtree.cpp",
        ],
  copts= COPTS,
  deps = ["//kern_interp:ki_mat",
      "//kern_interp/boundaries"]
)