import numpy as np
from matplotlib import pyplot as plt
from matplotlib.cm import ScalarMappable

x = range(50)
scales = np.linspace(0, 2, 7)
print(scales)
locs = range(4)
cmap = plt.get_cmap("Spectral")
norm = plt.Normalize(scales.min(), scales.max())

fig, axes = plt.subplots(2,2, constrained_layout=True, sharey=True)

for s_plot, ax in enumerate(axes.flat):
    for scale in scales:
        print(scale)
        y = np.random.normal(loc=locs[s_plot], scale=scale, size=50)
        print(x)
        print(y)
        print([cmap(norm(scale))])
        sc = ax.scatter(x, y, c=[cmap(norm(scale))], s=5)
        ax.set_title("Mean = {:d}".format(locs[s_plot]))

sm =  ScalarMappable(norm=norm, cmap=cmap)
sm.set_array([])
cbar = fig.colorbar(sm, ax=axes[:,1])
cbar.ax.set_title("scale")

plt.show()