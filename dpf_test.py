import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm

y_trgl, y_br, y_bl = 1.75, 1, 6
x_0, y_0, a_x_max, a_y_max = 5, 3, 3, 1
a_sta = 1e5


def plot_3d(x, y, x_obs, y_obs):
    Z = np.array(
        [[(dnf(xi, yi, x_obs, y_obs)) for xi in x] for yi in y]
    )

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    X, Y = np.meshgrid(x, y)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def plot_2d(x, y, φ, v, n_obs, x_obs, y_obs, φ_obs, v_obs):
    Z = np.array(
        [[(dnf(xi, yi, φ, v, n_obs, x_obs, y_obs, φ_obs, v_obs)) for xi in x] for yi in y]
    )

    Z = np.flipud(Z)
    y = np.flipud(y)

    plt.imshow(Z, cmap='coolwarm', extent=[min(x), max(x), min(y), max(y)])
    plt.colorbar()

    # plt.gca().set_aspect(5)

    plt.show()


def u_trgl(y, y_trgl, a=0.5):
    return a * (y - y_trgl) ** 2


def u_rbd(y, y_br, y_bl, b=100):
    if y >= y_bl:
        return b * (y - y_bl) ** 2
    elif y <= y_br:
        return b * (y - y_br) ** 2
    return 0


def get_safe_distances(X_e, X_obs):
    x_e, y_e, v_x_e, v_y_e = X_e[0], X_e[1], X_e[2], X_e[3]
    x_obs, y_obs, v_x_obs, v_y_obs = X_obs[0], X_obs[1], X_obs[2], X_obs[3]

    x_s = x_0 / 2 + (v_x_e - v_x_obs) ** 2 / (2 * a_x_max)
    y_s = y_0 / 2 + (v_y_e - v_y_obs) ** 2 / (2 * a_y_max)

    return x_s, y_s


def u_opf(x, y, x_obs, y_obs):
    x_s, y_s = 20, 1.5

    exp = 1 / 2 * ((x - x_obs) ** 2 / x_s ** 2 + (y - y_obs) ** 2 / y_s ** 2)

    temp = a_sta / (2 * np.pi * np.sqrt(x_s ** 2 + y_s ** 2)) * np.exp(-exp)
    return temp


def rotate(x, y, theta):
    c, s = np.cos(theta), np.sin(theta)
    return c * x - s * y, s * x + c * y


def dnf(x, y, φ, v, n_obs, x_obs, y_obs, φ_obs, v_obs, Af=0.01, b=1, σ_x=2, σ_y=0.5):
    U_pe = 0
    for i in range(n_obs):
        theta = φ - φ_obs[i]
        # Rotate observation coordinates
        x_i, y_i = rotate(x_obs[i], y_obs[i], theta)

        Δ_ep = v - v_obs[i]
        α = Δ_ep / 5

        # Rotate point (x, y)
        x_rot, y_rot = rotate(x, y, theta)

        # Evaluate distribution at rotated coordinates
        U_pe += Af * np.exp(
            - (
                      (x_rot - x_i) ** 2 / (2 * σ_x ** 2) +
                      (y_rot - y_i) ** 2 / (2 * σ_y ** 2)
              ) ** b
        ) * np.exp(-α * (x_rot - x_i))

    return U_pe


#
# def dnf(x, y, v, n_obs, x_obs, y_obs, v_obs, Af=0.01, b=1, σ_x=2, σ_y=0.5):
#     U_pe = 0
#     for i in range(n_obs):
#         Δ_ep = v - v_obs[i]
#         α = Δ_ep / 5
#         U_pe += Af * np.exp(
#             - (
#                       (x - x_obs[i]) ** 2 / (2 * σ_x ** 2) +
#                       (y - y_obs[i]) ** 2 / (2 * σ_y ** 2)
#               ) ** b
#         ) * np.exp(-α * (x - x_obs[i]))
#
#     return U_pe


x = np.arange(-5, 20, 0.1)
y = np.arange(-5, 5, 0.1)
φ = 0
v = 2

n_obs = 2
x_obs, y_obs = [10, 0], [2, 0]
φ_obs = [np.pi / 4, 0]
v_obs = [1, 3]

plot_2d(x, y, φ, v, n_obs, x_obs, y_obs, φ_obs, v_obs)
