import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys


font = {'size'   : 18}

LW=3
MS=10
matplotlib.rc('font', **font)
plt.figure(figsize=(15,10))

acc_lines = open("output/linalg_acc.txt", "r").readlines()
vals = [float(line.split(" ")[3]) for line in acc_lines]

plt.plot([pow(10, -i) for i in range(3, 10)],\
        vals, ":b^", linewidth=LW, markersize=MS)

plt.title("Error vs. dense solve, N=8192")
plt.xlabel("ID Error Tolerance")
plt.ylabel(r"$|\bar{f}-f^*|/|f^*|$")
plt.yscale('log', basex=10)
plt.xscale('log', basex=10)
plt.gca().invert_xaxis()
plt.savefig("output/linalg_acc_plot.eps", format="eps")
# plt.show()
