import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter


# hypothesis function of the form y = c1 + c2*x
def h(c1, c2, xh):
    size = xh.shape[0]

    a1 = np.ones((size, 1), dtype=np.float)
    a1 = np.concatenate((a1, xh), axis=1)

    a2 = np.array([c1, c2], dtype=np.double)

    yh = a1.dot(a2)

    return yh


def fit():
    print("fit function")


def plot_j(x_train, y_train):
    param1 = np.arange(-5, 5, 0.25)
    param1 = param1.reshape(len(param1), 1)
    param2 = np.arange(-5, 5, 0.25)
    param2 = param2.reshape(len(param2), 1)

    j_value = []

    for i in range(0, len(param1)):
        y_pred = h(param1[i], param2[i], x_train)

        j_value.append((np.average((y_train - y_pred) ** 2, axis=0, weights=None))/2)

    j_value = np.array(j_value)

    print(param1.shape)
    print(param2.shape)
    print(j_value.shape)

    fig = plt.figure()
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(param1, param2, j_value, cmap=cm.coolwarm, linewidth=10)
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()




