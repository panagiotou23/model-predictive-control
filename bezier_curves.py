import numpy as np
import matplotlib.pyplot as plt

# Define the control points
control_points = np.array([[0, 0], [1, 0], [2, 0], [2, 3.75], [4, 3.75], [5, 3.75]])


# Define the Bézier curve function
def bezier_curve(control_points, num_points=100):
    N = len(control_points)
    t = np.linspace(0, 1, num_points)

    # Initialize an empty array for the curve points
    curve = np.zeros((num_points, 2))

    # Calculate the curve points
    for i in range(N):
        curve += np.outer(binom(N - 1, i) * (1 - t) ** (N - 1 - i) * t ** i, control_points[i])

    return curve


# Define the binomial coefficient function
def binom(n, k):
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))


# Generate the Bézier curve
curve = bezier_curve(control_points)

# Plot the control points and the Bézier curve
plt.figure()
plt.plot(control_points[:, 0], control_points[:, 1], 'ro-', label='Control Points')
plt.plot(curve[:, 0], curve[:, 1], 'b-', label='Bézier Curve')
plt.legend()
plt.show()
