import casadi as cs
import numpy as np
import matplotlib.pyplot as plt
import scipy.special


def I_o(x):
    # return scipy.special.iv(0, x)  # Computing the modified Bessel function of 0 order
    return np.math.factorial(0) * np.power((x / 2), 0) / np.math.gamma(
        1 + 0 + 1)  # Modified Bessel function of the 0th order at x


def dpf(
        ego_state: np.ndarray,
        proceeding_state: np.ndarray
):
    α, A_f, σ_x, σ_y, b = 100, 1000, 1, 1, 1
    x_e, y_e, φ_e, v_e = ego_state[0], ego_state[1], ego_state[2], ego_state[3]
    x_p, y_p, φ_p, v_p = proceeding_state[0], proceeding_state[1], proceeding_state[2], proceeding_state[3]

    Δ_pe = v_p - v_e
    θ = φ_p - φ_e

    U_rd = A_f * np.exp(
        - ((x_e - x_p) ** 2 / (2 * σ_x ** 2) + (y_e - y_p) ** 2 / (2 * σ_y ** 2)) ** b
    )

    η = Δ_pe
    U_Δ = α * np.exp(
        η * np.cos(θ)
    ) / (2 * np.pi * I_o(η))

    print("U_Δ: ", U_Δ)
    print("U_rd: ", U_rd)
    print()
    return U_Δ * U_rd


ego_state = np.array([
    0., 0., 0., 13.
])

proceeding_state = np.array([
    25., 0., 0., 15.
])

U_pe = np.zeros(50)

for i in range(50):
    U_pe[i] = dpf(
        ego_state,
        proceeding_state
    )
    ego_state += np.array([
        .5, 0, 0, 0
    ])

print(U_pe)

plt.figure()
plt.plot(
    U_pe
)
plt.show()
