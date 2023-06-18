import matplotlib.pyplot as plt
import numpy as np
import math

h = 3.75
L, W = 4.2, 1.8
θ = 3.2 / 180 * np.pi
l = 3
Lf = 1

v0, v1 = 20, 10
D1 = 50


def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def bezier_curve(j, P):
    x, y = 0, 0
    for i in range(0, 6):
        c = binomial_coefficient(5, i)
        x += c * (1 - j) ** (5 - i) * j ** i * P[0, i]
        y += c * (1 - j) ** (5 - i) * j ** i * P[1, i]
    return x, y


def get_bezier_control_points(i):
    Px0 = Py0 = Py1 = Py2 = 0
    Py3 = Py4 = Py5 = h

    Li = Lf + l
    Di = Li * np.cos(np.arctan2(W, 2 * Lf) - θ)
    tc1 = D1 / (v0 - v1)
    Px2 = Px3 = v0 * tc1 - Di
    Px5 = 2 * Px2
    Px1 = (Px2 - Px0) / i
    Px4 = Px5 - (Px5 - Px3) / i

    Px = np.array([
        Px0, Px1, Px2, Px3, Px4, Px5
    ])
    Py = np.array([
        Py0, Py1, Py2, Py3, Py4, Py5
    ])

    tca = Px2 / (v0 - v1)
    return np.array([Px, Py]), tca


if __name__ == '__main__':
    # Plot the Bézier curve
    plt.figure(figsize=(10, 5))
    j = np.linspace(0, 1, num=500)

    for i in range(1, 11):
        P, tca = get_bezier_control_points(i)
        curve = np.array([bezier_curve(ji, P) for ji in j])
        plt.plot(curve[:, 0], curve[:, 1])

    plt.title('Bézier Curve for Lane-Changing Path')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend(['i=1', 'i=2', 'i=3', 'i=4', 'i=5', 'i=6', 'i=7', 'i=8', 'i=9', 'i=10'])
    plt.grid(True)
    plt.show()
