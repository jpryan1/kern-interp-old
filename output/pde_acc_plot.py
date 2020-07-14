import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sys


font = {'size'   : 22}

LW=3
MS=10
matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(14,14))

acc_lines = open("output/pde_acc.txt", "r").readlines()

epslines = acc_lines[:7]
nlines = acc_lines[7:]

epsy = np.array([float(line.split(" ")[3]) for line in epslines])
# epserr = np.array([[float(line.split(" ")[1]), float(line.split(" ")[3])] \
#   for line in epslines])

# for i in range(len(epsy)):
#   epserr[i][0] = epsy[i] -  epserr[i][0] 
#   epserr[i][1] = epserr[i][1]  - epsy[i] 

ax.plot([pow(10, -i) for i in range(3, 10)],\
        epsy, "-b^", linewidth=LW, markersize=MS)
# axs[1].plot([pow(10, -i) for i in range(3, 10)],\
#         vals, ":b^", linewidth=LW, markersize=MS)

ax.set_xlabel("ID Error Tolerance")
ax.set_ylabel(r"$||\hat{u}-u||_{\infty}/||u||_{\infty}$")
ax.set_yscale('log', basey=10)
ax.set_xscale('log', basex=10)
ax.invert_xaxis()


# ny = np.array([float(line.split(" ")[2]) for line in nlines])
# nerr = np.array([[float(line.split(" ")[1]), float(line.split(" ")[3])] \
#   for line in nlines])

# for i in range(len(ny)):
#   nerr[i][0] = ny[i] -  nerr[i][0] 
#   nerr[i][1] = nerr[i][1]  - ny[i] 

# axs[1].errorbar([pow(2, i) for i in range(6, 12)],\
#         ny, nerr.transpose(), fmt="--b^", linewidth=LW, markersize=MS)

# axs[1].set_title("Computed vs analytic solution, ID error tol=1e-9")
# axs[1].set_xlabel("Number of integration nodes")
# axs[1].set_ylabel(r"$|\bar{u}-u^*|/|u^*|$")
# axs[1].set_yscale('log', basex=10)
# axs[1].set_xscale('log', basex=10)


# axs[0].savefig("output/pde_acc_plot.eps", format="eps")
plt.savefig("pde_acc_plot.eps", format="eps")
# plt.show()
