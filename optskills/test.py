import cma


import numpy as np
print('Hello, cec15')
n = 51
xmax = 2.0
xmin = -xmax
ymax = xmax
ymin = -ymax
xs = np.linspace(xmin, xmax, n)
ys = np.linspace(ymin, ymax, n)
xv, yv = np.meshgrid(xs, ys)
zv = np.zeros(shape=(n, n))
for i in range(n):
    for j in range(n):
        x = np.array([xs[i], ys[j]])
        # zv[i][j] = bent_cigar_func(x, adjust=0.0)
        # zv[i][j] = weierstrass_func(x, adjust=1.5)
        zv[i][j] = 1.0

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import matplotlib.pyplot as plt

fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(xv, yv, zv, rstride=1, cstride=1,
                       cmap=cm.coolwarm,
                       linewidth=0, antialiased=False)
# ax.set_zlim(-1.01, 1.01)

ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

fig.colorbar(surf, shrink=0.5, aspect=5)

import matplotlib
matplotlib.interactive(False)
print matplotlib.is_interactive()
plt.show()
