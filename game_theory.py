import matplotlib.pyplot as plt
import numpy as np


def normal_distribution(x, sigma):
    exponent = -(x ** 2) / (2 * sigma ** 2)
    return np.exp(exponent)


def normal_cdf(x, sigma):
    z = x / (sigma * np.sqrt(2))
    return (1 + np.math.erf(z)) / 2


def safety_factor(Theadway, sigma=1.2):
    # return 2 * (1 - normal_distribution(Theadway, sigma)) - 1
    if abs(Theadway) >= 3:
        return 1
    return 2 * abs(Theadway) / 3 - 1



def space_factor_same_lane(t21):
    return 2 * normal_cdf(t21, 2) - 1


def space_factor_diff_lane(t21):
    # return 2 * normal_cdf((t21 + 1.5), 0.8) - 1
    if t21 <= 3:
        return -1
    if t21 >= 0:
        return 1
    return 2 / 3 * t21 + 1


def safety_payoff(P1, P2, v1, v2, a1, a2, Tcl):
    Theadway0 = (P1 - P2) / v2

    P1l = P1 + v1 * Tcl + 0.5 * a1 * (Tcl ** 2)
    P2l = P2 + v2 * Tcl + 0.5 * a2 * (Tcl ** 2)
    v1l = v1 + a1 * Tcl
    v2l = v2 + a2 * Tcl
    if P1l >= P2l:
        TheadwayTcl = (P1l - P2l) / v2l
    else:
        TheadwayTcl = (P2l - P1l) / v1l

    SP_0 = safety_factor(Theadway0)
    SP_Tcl = safety_factor(TheadwayTcl)
    print("T0: ", Theadway0, "\tTTcl: ", TheadwayTcl)
    print("SP0: ", SP_0, "\t SPTcl: ", SP_Tcl)
    Usafety = 0.5 * (SP_Tcl - SP_0)
    print("P: ", P1l, "\t", P2l)
    print("Safety: ", Usafety, "\t", SP_0, "\t", SP_Tcl)

    return Usafety


def space_factor_t21(P1, P2, v1, v2):
    if P2 <= P1:
        return (P2 - P1) / v2
    else:
        return (P2 - P1) / v1


def space_payoff(P1, P2, v1, v2, a1, a2, Tcl, initial_same_lane):
    t21_0 = space_factor_t21(P1, P2, v1, v2)
    P1_Tcl = P1 + v1 * Tcl + 0.5 * a1 * Tcl ** 2
    P2_Tcl = P2 + v2 * Tcl + 0.5 * a2 * Tcl ** 2
    v1_Tcl = v1 + a1 * Tcl
    v2_Tcl = v2 + a2 * Tcl
    if v1_Tcl <= 0:
        v1_Tcl = 1e-5
    if v2_Tcl <= 0:
        v2_Tcl = 1e-5
    t21_Tcl = space_factor_t21(P1_Tcl, P2_Tcl, v1_Tcl, v2_Tcl)

    if initial_same_lane:
        RP0 = space_factor_same_lane(t21_0)
        RPTcl = space_factor_same_lane(t21_Tcl)
    else:
        RP0 = space_factor_diff_lane(t21_0)
        RPTcl = space_factor_diff_lane(t21_Tcl)

    U_space2 = 0.5 * (RPTcl - RP0)
    U_space1 = -U_space2

    print("Space: ", U_space1, "\t", RP0, "\t", RPTcl)
    print(t21_0, "\t", t21_Tcl)
    print()
    return U_space1, U_space2


def f_w(v, a):
    return np.exp(
        - (Tcl ** 2 * (a - a0) ** 2 / w1 + (v + a * Tcl - v_desired) ** 2 / w2)
    )


def β(q):
    return normal_cdf(q, 1)


def total_payoff(P1, P2, v1, v2, a1, a2, Tcl, initial_same_lane):
    U_safety = safety_payoff(P1, P2, v1, v2, a1, a2, Tcl)
    U_space1, U_space2 = space_payoff(P1, P2, v1, v2, a1, a2, Tcl, initial_same_lane)

    f1 = f_w(v1, a1)
    f2 = f_w(v2, a2)

    q1, q2 = 0, 0
    U_payoff1 = f1 * (
            0.5 * U_safety + 0.5 * U_space1 + 1
    ) - 1

    U_payoff2 = f2 * (
            0.5 * U_safety + 0.5 * U_space2 + 1
    ) - 1

    return U_payoff1, U_payoff2


Tc1 = 5
y1 = 3
LW = 3

a0 = 0
v_desired = 10
w1, w2 = 1e2, 1e2

Tcl = 3

P1, v1, a1 = 0, 15, 0
P2, v2, a2 = 15, 10, 0

print(
    f_w(v1, a0)
)
#
# a_vals = np.arange(-6, 4, 0.1)
# total = np.array([
#     total_payoff(P1, P2, v1, v2, a1, a2, Tcl, True) for a1 in a_vals
# ])
#
# safety = np.array([
#     safety_payoff(P1, P2, v1, v2, a1, a2, Tcl) for a1 in a_vals
# ])
# space = np.array([
#     space_payoff(P1, P2, v1, v2, a1, a2, Tcl, False) for a1 in a_vals
# ])
# plt.figure()
# plt.plot(a_vals, safety)
# plt.plot(a_vals, space)
# plt.legend(['Safety', 'Space 1', 'Space 2'])
# plt.show()
#
# plt.figure()
# plt.plot(a_vals, total[:, 0])
# plt.plot(a_vals, total[:, 1])
# plt.plot(a_vals, total[:, 0] + total[:, 1])
# plt.legend(['Payoff 1', 'Payoff 2', 'Payoff 1 + Payoff 2'])
# plt.show()
