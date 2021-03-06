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


def bent_cigar_func(x, Os=None, Mr=None, sr=None, adjust=1.0):
    nx = len(x)
    # z = sr_func(x, Os, Mr, 1.0)
    z = sr_func(x, Os, Mr, 100.0)
    f = z[0] * z[0]
    for i in range(1, nx):
        f += pow(10.0, 6.0 * adjust) * z[i] * z[i]
    return f / 5e10


def weierstrass_func(x, Os=None, Mr=None, sr=None, adjust=1.0):
    a = 0.5 * adjust
    b = 3.0
    k_max = 20
    f = 0.0

    sr = 1.0 if sr is None else sr

    nx = len(x)
    # z = sr_func(x, Os, Mr, 0.5 / 100.0)
    z = sr_func(x, Os, Mr, 0.5 * sr)
    for i in range(nx):
        sum = 0.0
        sum2 = 0.0
        for j in range(k_max + 1):
            sum += pow(a, j) * cos(2.0 * PI * pow(b, j) * (z[i] + 0.5))
            sum2 += pow(a, j) * cos(2.0 * PI * pow(b, j) * 0.5)
        f += sum
    f -= nx * sum2
    return f / 10.0


def schwefel_func(x, Os=None, Mr=None, sr=None, adjust=1.0):
    nx = len(x)
    # z = sr_func(x, Os, Mr, 1000.0 / 100.0)
    adjust = 0.2
    z = sr_func(x, Os, Mr, 100000.0 / 100.0 * adjust)
    f = 0.0
    dim = float(len(x))
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
    # return max(f / 1000.0 - 0.838, 0.0)
    # return max(f / 1000.0, 0.0)
    return max((f - (dim * (dim - 1)) * 4.189e+002) / 1000.0, 0.0)


def hgbat_func(x, Os=None, Mr=None, sr=None, adjust=1.0):
    nx = len(x)
    # z = sr_func(x, Os, Mr, 5.0 / 100.0)
    z = sr_func(x, Os, Mr, 1.3)

    alpha = 1.0 / 4.0
    r2 = 0.0
    sum_z = 0.0

    for i in range(nx):
        z[i] = z[i] - 1.0  # shift to orgin
        r2 += z[i] * z[i]
        sum_z += z[i]
    f = pow(fabs(pow(r2,2.0)-pow(sum_z,2.0)),2*alpha) + (0.5*r2 + sum_z)/nx + 0.5
    f *= 0.1
    return f


if __name__ == '__main__':
    print('Hello, cec15')
    n = 101
    xcen = 0.0
    ycen = 0.0
    xsize = 1.0
    ysize = 1.0
    xmax = xcen + xsize
    xmin = xcen - xsize
    ymax = ycen + ysize
    ymin = ycen - ysize
    xs = np.linspace(xmin, xmax, n)
    ys = np.linspace(ymin, ymax, n)
    xv, yv = np.meshgrid(xs, ys)
    zv = np.zeros(shape=(n, n))
    for i in range(n):
        for j in range(n):
            x = np.array([xs[i], ys[j]])
            # zv[i][j] = bent_cigar_func(x, adjust=0.0)
            # zv[i][j] = weierstrass_func(x, adjust=1.5)
            zv[i][j] = schwefel_func(x)
            # zv[i][j] = hgbat_func(x)
            # zv[i][j] = 1.0
    print('Minimum value = %.8f' % np.min(zv))
    # print('Value at orig = %.8f' % hgbat_func(np.array([0.0, 0.0])))
    # print('Value at diff = %.8f' % hgbat_func(np.array([1.0, 1.0])))
    # print('Value at orig = %.8f' % hgbat_func(np.zeros(10)))
    # print('Value at orig = %.8f' % hgbat_func(0.02 * (np.random.rand(10) - 0.5)))

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
