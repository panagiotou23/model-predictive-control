import matplotlib.pyplot as plt
import numpy as np

from scipy.stats import norm


def normal_distribution(x, sigma):
    exponent = -(x ** 2) / (2 * sigma ** 2)
    return np.exp(exponent)


def safety_factor(Theadway, sigma=1.2):
    return 2 * (1 - normal_distribution(Theadway, sigma)) - 1


def safety_payoff(P1, P2, v1, v2, a1, a2, Tc1, y1, LW):
    Theadway0 = (P1 - P2) / v2

    Tcl = y1 * Tc1 / LW

    P1l = P1 + v1 * Tcl + 0.5 * a1 * (Tcl ** 2)
    P2l = P2 + v2 * Tcl + 0.5 * a2 * (Tcl ** 2)

    if P1l >= P2l:
        TheadwayTcl = (P1l - P2l) / (v2 + a2 * Tcl)
    else:
        TheadwayTcl = (P2l - P1l) / (v1 + a1 * Tcl)

    SP_0 = safety_factor(Theadway0)
    SP_Tcl = safety_factor(TheadwayTcl)

    Usafety = 0.5 * (SP_Tcl - SP_0)

    return Usafety


def space_factor_t21(P1, P2, v1, v2):
    if P2 <= P1:
        return (P2 - P1) / v2
    else:
        return (P2 - P1) / v1


def space_factor_same_lane(t21, sigma=0.5):
    return norm.cdf(t21, loc=3, scale=sigma) - norm.cdf(t21, loc=-3, scale=sigma)


def space_factor_diff_lane(t21, sigma=0.5):
    return norm.cdf(t21, loc=0, scale=sigma) - norm.cdf(t21, loc=-3, scale=sigma)


def space_payoff(P1, P2, v1, v2, a1, a2, Tc1, y1, LW, initial_same_lane, sigma=0.5):
    Tcl = y1 * Tc1 / LW

    t21_0 = space_factor_t21(P1, P2, v1, v2)
    P1_Tcl = P1 + v1 * Tcl + 0.5 * a1 * Tcl ** 2
    P2_Tcl = P2 + v2 * Tcl + 0.5 * a2 * Tcl ** 2
    t21_Tcl = space_factor_t21(P1_Tcl, P2_Tcl, v1, v2)

    if initial_same_lane:
        RP0 = space_factor_same_lane(t21_0, sigma)
    else:
        RP0 = space_factor_diff_lane(t21_0, sigma)

    # Check if the cars are in the same lane after Tcl based on their relative lane positions
    future_same_lane = (y1 * Tc1) % (2 * LW) < LW

    if future_same_lane:
        RPTcl = space_factor_same_lane(t21_Tcl, sigma)
    else:
        RPTcl = space_factor_diff_lane(t21_Tcl, sigma)

    Uspace = 0.5 * (RPTcl - RP0)

    return Uspace


Tc1 = 3
y1 = 3
LW = 3

P1, v1, a1 = 10, 10, 0
P2, v2, a2 = 15, 5, 0

print(safety_payoff(P1, P2, v1, v2, a1, a2, Tc1, y1, LW))
