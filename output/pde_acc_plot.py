import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys


font = {'size'   : 18}

LW=3
MS=10
matplotlib.rc('font', **font)
fig, axs = plt.subplots(1, 2, figsize=(25,10))

acc_lines = open("output/pde_acc.txt", "r").readlines()

epslines = acc_lines[:7]
nlines = acc_lines[7:]

epsy = np.array([float(line.split(" ")[2]) for line in epslines])
epserr = np.array([[float(line.split(" ")[1]), float(line.split(" ")[3])] \
  for line in epslines])

for i in range(len(epsy)):
  epserr[i][0] = epsy[i] -  epserr[i][0] 
  epserr[i][1] = epserr[i][1]  - epsy[i] 

axs[0].errorbar([pow(10, -i) for i in range(3, 10)],\
        epsy, epserr.transpose(), fmt="--b^", linewidth=LW, markersize=MS)
# axs[1].plot([pow(10, -i) for i in range(3, 10)],\
#         vals, ":b^", linewidth=LW, markersize=MS)

axs[0].set_title("Computed vs analytic solution, N=8192")
axs[0].set_xlabel("ID Error Tolerance")
axs[0].set_ylabel(r"$|\bar{u}-u^*|/|u^*|$")
axs[0].set_yscale('log', basex=10)
axs[0].set_xscale('log', basex=10)
axs[0].invert_xaxis()


ny = np.array([float(line.split(" ")[2]) for line in nlines])
nerr = np.array([[float(line.split(" ")[1]), float(line.split(" ")[3])] \
  for line in nlines])

for i in range(len(ny)):
  nerr[i][0] = ny[i] -  nerr[i][0] 
  nerr[i][1] = nerr[i][1]  - ny[i] 

axs[1].errorbar([pow(2, i) for i in range(6, 12)],\
        ny, nerr.transpose(), fmt="--b^", linewidth=LW, markersize=MS)

axs[1].set_title("Computed vs analytic solution, ID error tol=1e-9")
axs[1].set_xlabel("Number of integration nodes")
axs[1].set_ylabel(r"$|\bar{u}-u^*|/|u^*|$")
axs[1].set_yscale('log', basex=10)
axs[1].set_xscale('log', basex=10)


# axs[0].savefig("output/pde_acc_plot.eps", format="eps")
plt.savefig("pde_acc_plot.eps", format="eps")
# plt.show()
