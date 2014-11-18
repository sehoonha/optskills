import numpy as np
from math import pi, sin, cos, fmod, fabs

PI = pi


def shiftfunc(x, Os):
    return x - Os


def rotatefunc(x, Mr):
    return Mr * x


def sr_func(x, Os=None, Mr=None, sh_rate=1.0):
    if Os is not None:
        if Mr is not None:
            y = shiftfunc(x, Os)
            y = y * sh_rate
            sr_x = rotatefunc(y, Mr)
        else:
            sr_x = shiftfunc(x, Os)
            sr_x = sr_x * sh_rate
    else:
        if Mr is not None:
            y = x * sh_rate
            sr_x = rotatefunc(y, Mr)
        else:
            sr_x = x * sh_rate
    return sr_x


def bent_cigar_func(x, Os=None, Mr=None):
    nx = len(x)
    # z = sr_func(x, Os, Mr, 1.0)
    z = sr_func(x, Os, Mr, 100.0)
    f = z[0] * z[0]
    for i in range(1, nx):
        f += pow(10.0, 6.0) * z[i] * z[i]
    return f / 5e10


def weierstrass_func(x, Os=None, Mr=None):
    a = 0.5
    b = 3.0
    k_max = 20
    f = 0.0

    nx = len(x)
    # z = sr_func(x, Os, Mr, 0.5 / 100.0)
    z = sr_func(x, Os, Mr, 0.5)
    for i in range(nx):
        sum = 0.0
        sum2 = 0.0
        for j in range(k_max + 1):
            sum += pow(a, j) * cos(2.0 * PI * pow(b, j) * (z[i] + 0.5))
            sum2 += pow(a, j) * cos(2.0 * PI * pow(b, j) * 0.5)
        f += sum
    f -= nx * sum2
    return f / 10.0


def schwefel_func(x, Os=None, Mr=None):
    nx = len(x)
    # z = sr_func(x, Os, Mr, 1000.0 / 100.0)
    z = sr_func(x, Os, Mr, 100000.0 / 100.0)
    f = 0.0
    for i in range(nx):
        z[i] += 4.209687462275036e+002
        if z[i] > 500:
            f -= (500.0 - fmod(z[i], 500)) * sin(pow(500.0 - fmod(z[i], 500),
                                                     0.5))
            tmp = (z[i] - 500.0) / 100.0
            f += tmp * tmp / nx
        elif z[i] < -500:
            f -= (-500.0 + fmod(fabs(z[i]), 500)) * sin(
                pow(500.0 - fmod(fabs(z[i]), 500), 0.5))
            tmp = (z[i] + 500.0) / 100
            f += tmp * tmp / nx
        else:
            f -= z[i] * sin(pow(fabs(z[i]), 0.5))
        f += 4.189828872724338e+002 * nx
    # return f
    return f / 1000.0 - 0.80


if __name__ == '__main__':
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
            # zv[i][j] = bent_cigar_func(x)
            # zv[i][j] = weierstrass_func(x)
            zv[i][j] = schwefel_func(x)

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

    plt.show()
