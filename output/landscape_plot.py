
import matplotlib
# matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
font = {'size'   : 18}

matplotlib.rc('font', **font)
fig, ax = plt.subplots(figsize=(15,15))
# fig, ax = plt.subplots(figsize=(10,7))
# axs=[ax]
# data = open("output/ex3Agrads.txt", "r").readlines()

# pointsx = []
# pointsy = []
# vals = []
# im = np.array([[0. for i in range(100)] for j in range(50)])

# for line in data:
#   spl = line.split(" ")

#   linenums = [float(s) for s in spl]

#   # if(linenums[0]<50):
#   #   linenums[0] += 100
#   pointsx.append(linenums[0])
#   pointsy.append(linenums[1])
#   vals.append(linenums[2])
#   im[int(linenums[1])-25][(50+int(linenums[0]))%100] = linenums[2]

# # for i in range(len(pointsx)):
# #   if(pointsx[i]<pointsy[i]):
# #     pointsx[i] = pointsx[i] + 2*np.pi


# sc = ax.imshow(im, origin="lower",cmap=plt.cm.cool,\
#   extent=[0,100,25,75],interpolation="bilinear")

# # sc = ax.scatter(pointsx,pointsy,c=vals, cmap=plt.cm.cool, s=300)
# cbar=fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
# cbar.set_ticks([-1,1])
# # cbar.ax.tick_params(labelsize=20)

# ax.set_xticks([0,25,50,75,100])
# ax.set_xticklabels([r"$-\pi$",r"$-\pi/2$",r"$0$",r"$\pi/2$",r"$\pi$"])
# ax.set_yticks([25,37.5,50,62.5,75])
# ax.set_yticklabels([r"$\pi/2$",r"$3\pi/4$",r"$\pi$",r"$5\pi/4$",r"$3\pi/2$"])

# ax.set_xlabel(r"$\theta_1$")
# ax.set_ylabel(r"$\theta_2-\theta_1$")
# ax.set_title("Laplace")

# print("Max from ex3: ",max(vals))

# maxval=0
# maxind =0
# for i in range(len(vals)):
#   if vals[i] > maxval:
#     maxind = i
#     maxval = vals[i]

# print("Ex3a", 2.*np.pi*(pointsx[maxind])/100.,\
#   (2.*np.pi*(pointsx[maxind])/100.)+(2.*np.pi*(pointsy[maxind])/100.), maxval)




# plt.savefig("landscape.eps",  bbox_inches="tight", format="eps")
# plt.show()



data = open("output/ex3Bflows.txt", "r").readlines()

pointsx = []
pointsy = []
vals = []
im = np.array([[0. for i in range(100)] for j in range(50)])
for line in data:
  spl = line.split(" ")

  linenums = [float(s) for s in spl]

  pointsx.append(linenums[0])
  pointsy.append(linenums[1])
  vals.append(linenums[2])
  im[int(linenums[1])-25][int(linenums[0])] = linenums[2]
# for i in range(len(pointsy)):
#   if(pointsx[i]>pointsy[i]):
#     pointsy[i] = pointsy[i] + 2*np.pi

sc = ax.imshow(im, origin="lower",cmap=plt.cm.cool,\
   extent=[0,100,25,75],interpolation="bilinear")
# sc = ax.scatter(pointsx,pointsy,c=vals, cmap=plt.cm.cool, s=300)
cbar=fig.colorbar(sc, ax=ax, shrink=0.5, aspect=10)
cbar.set_ticks([-3, 0.5])

ax.set_xticks([0,25,50,75,100])
ax.set_xticklabels(["0",r"$\pi/2$",r"$\pi$",r"$3\pi/2$",r"$2\pi$"])
ax.set_yticks([25,37.5,50,62.5,75])
ax.set_yticklabels([r"$\pi/2$",r"$3\pi/4$",r"$\pi$",r"$5\pi/4$",r"$3\pi/2$"])

ax.set_xlabel(r"$\theta_1$")
ax.set_ylabel(r"$\theta_2-\theta_1$")
ax.set_title("Stokes")

print("Max from ex3: ",max(vals))

maxval=0
maxind =0
for i in range(len(vals)):
  if vals[i] > maxval:
    maxind = i
    maxval = vals[i]

print("Ex3b", 2.*np.pi*(pointsx[maxind])/100.,\
  (2.*np.pi*(pointsx[maxind])/100.)+(2.*np.pi*(pointsy[maxind])/100.), maxval)
plt.savefig("landscape.eps",  bbox_inches="tight", format="eps")
plt.show()
