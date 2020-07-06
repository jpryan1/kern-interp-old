from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 unused import

import numpy as np
import sys
from copy import copy
import matplotlib
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Ex Spiral
is_channel_plot = False
ARROW_LENGTH = 0.4
BORDER_WIDTH = 8
HEAD_WIDTH = 3
QUIVER_RES_X = 1
QUIVER_RES_Y = 1
BOUNDARY_RES = 5
ZOOM = 1
TICK_LABEL_SIZE = 40
TICKS = [0.2, 0.9, 1.6]
# OUTPUT_FILE = "ex1.eps"
# config.num_boundary_points = pow(2, 12);
# config.domain_size = 200;

print("args: {ZOOM} ")
fig = plt.figure(figsize=(15,15))
ax = fig.gca(projection='3d')
# plt.axis('off')

ax.xaxis.set_pane_color((1.0,1.0,1.0, 0.0))
ax.yaxis.set_pane_color((1.0,1.0,1.0, 0.0))
ax.zaxis.set_pane_color((1.0,1.0,1.0, 0.0))
# ax.xaxis.set_pane_color((0.9,0.9,0.9, 0.2))
# ax.yaxis.set_pane_color((0.9,0.9,0.9, 0.2))
# ax.zaxis.set_pane_color((0.9,0.9,0.9, 0.2))
# make the grid lines transparent
ax.xaxis._axinfo["grid"]['color'] = (0.6,0.6,0.6, 0.5)
ax.yaxis._axinfo["grid"]['color'] =   (0.6,0.6,0.6, 0.5)
ax.zaxis._axinfo["grid"]['color'] =  (0.6,0.6,0.6, 0.5)
for line in ax.xaxis.get_ticklines():
    line.set_visible(False)
for line in ax.yaxis.get_ticklines():
    line.set_visible(False)
for line in ax.zaxis.get_ticklines():
    line.set_visible(False)

ax.xaxis.set_ticklabels([])
ax.yaxis.set_ticklabels([])
ax.zaxis.set_ticklabels([])


MASKED_VALUE = 11111.1
if(len(sys.argv) > 1):
  ZOOM = float(sys.argv[1])


solution_lines = open("output/data/ie_solver_solution.txt", "r").readlines()

is_stokes = True

if is_stokes:
  CMAP = copy(matplotlib.cm.viridis)
else:
  CMAP = copy(matplotlib.cm.hot)
CMAP.set_bad("white",1.)
# solution_dim = 3 #int(np.sqrt(len(solution_lines)))
# solution_grid = np.array([[MASKED_VALUE for x in range(solution_dim)] for y in range(solution_dim)])

X, Y, Z, U, V, W, C = [], [], [], [], [], [], []

stats=[]
xshift=0.1
yshift=0.1
zshift=0.1

for i in range(0, len(solution_lines),1):

  linesplit = [float(n) for n in solution_lines[i].split(',')]

  mag = np.sqrt((linesplit[0]-0.5)**2 +(linesplit[1]-0.5)**2 + (linesplit[2]-0.5)**2)
  cyl_mag = np.sqrt((linesplit[0]-0.5)**2 +(linesplit[1]-0.5)**2 )
  if(cyl_mag <0.7 and mag <0.96 and linesplit[2] < 0.8 and linesplit[2] > 0.2):

    # occ_mag1 =  np.sqrt((linesplit[0]-0.3)**2 +(linesplit[2]-0.5)**2)
    # occ_mag2 =  np.sqrt((linesplit[0]-0.7)**2 +(linesplit[2]-0.5)**2)
    # if linesplit[1] < 0.5 and (occ_mag1 < 0.1 or occ_mag2 < 0.1):
    #   continue
    if abs(0.35-linesplit[1]) > 0.02 and abs(0.64-linesplit[1]) > 0.02 and abs(0.5-linesplit[1]) > 0.02:
      continue 


    if linesplit[2] ==0.5 and linesplit[1]==0.5 and linesplit[0] < 1.0:
      continue    

    if linesplit[2] ==0.5 and abs(0.64-linesplit[1]) > 0.02 and linesplit[0] < 0.3:
      continue
    u, v, w = linesplit[3], linesplit[4],linesplit[5]
    sz = np.sqrt( (u)**2 + (v)**2+(w)**2 )
    if(sz == 0):
      continue


    X.append((linesplit[0])-xshift)
    Y.append((linesplit[1])-yshift)
    Z.append((linesplit[2])-zshift)
    U.append((linesplit[3]))
    V.append((linesplit[4]))
    W.append((linesplit[5]))
    stats.append(sz)
    C.append( cm.winter((sz-0.6)*2.2))
stats.sort()
u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
# x = 0.3+0.25*np.cos(u)*np.sin(v)
# y = 0.3+0.25*np.sin(u)*np.sin(v)
# z = 0.3+0.25*np.cos(v)
# ax.plot_wireframe(x, y, z, color="pink")
# ax.plot_wireframe(x+0.4, y+0.4, z+0.4, color="pink")

# x = 0.1+(0.25*np.cos(u)*np.sin(v))
# y = 0.5+(0.25*np.sin(u)*np.sin(v))
# z = 0.5+(0.25*np.cos(v))
# ax.plot_wireframe(x, y, z, color="gray", linestyle="dashed")

# x2 = 0.9+(0.25*np.cos(u)*np.sin(v))
# y2 = 0.5+(0.25*np.sin(u)*np.sin(v))
# z2 = 0.5+(0.25*np.cos(v))
# ax.plot_wireframe(x2, y2, z2, color="gray", linestyle="dashed")



clen = len(C)

for i in range(clen):
  C.append(C[i])
  C.append(C[i])
ax.quiver(X,Y,Z, U,V,W, colors=C, length=0.08, linewidth=5, arrow_length_ratio=0.3,pivot="middle")



N=100
stride = 1
u = np.linspace(0, 2 * np.pi, N)
v = np.linspace(0, np.pi, N)
x = np.outer(np.cos(u), np.sin(v))
y = np.outer(np.sin(u), np.sin(v))
z = np.outer(np.ones(np.size(u)), np.cos(v))
ax.plot_surface(0.1*x+0.5-xshift, 0.1*y+0.5-yshift, 0.1*z+0.5-zshift, linewidth=0, antialiased=False, edgecolor="none", cstride=stride, rstride=stride, color="white")
# ax.plot_wireframe(0.1*x+0.5-xshift, 0.1*y+0.5-yshift, 0.1*z+0.5-zshift, linewidth=0.15,color="gray")
# ax.plot_surface(0.1*x+0.3, 0.1*y+0.5, 0.1*z+0.5, linewidth=0.0, cstride=stride, rstride=stride, color="white", shade=True)

maxlim = 0.85
# plt.savefig(OUTPUT_FILE, bbox_inches="tight", format="eps")
ax.set_xlim([-0.0,maxlim])   # Like so.
ax.set_ylim([-0.0,maxlim])
ax.set_zlim([-0.0,maxlim])


"""                                                                                                                                                    
Scaling is done from here...                                                                                                                           
"""
x_scale=5.0
y_scale=5.0
z_scale=5.0

scale=np.diag([x_scale, y_scale, z_scale, 1.0])
scale=scale*(1.0/scale.max())
scale[3,3]=1.0

def short_proj():
  return np.dot(Axes3D.get_proj(ax), scale)

ax.get_proj=short_proj

ax.set_aspect("equal")

ax.view_init(azim=5, elev=0)
ax.dist = 5
plt.savefig("sphere.png", dpi=300)
plt.show()


