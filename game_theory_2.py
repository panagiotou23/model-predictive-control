import numpy as np

L, W, l, θ_max, a_max = 4.2, 1.8, 3, 3.2 / 180 * np.pi, 7
tlc, td, τ, ti = 5.17, 1.2, 0.9, 0.15
q1, q2 = 0.65, 0.35


def get_safety_distance_headway(vm):
    return vm * td + l


def get_breaking_distance(vm, vf):
    return (vm - vf) * τ + ti / 2 + (vm - vf) ** 2 / (2 * a_max) + l


def get_fusion_distance(vm, vf, same_lane):
    if vm > vf:
        if same_lane:
            return np.max(
                vm - vf * tlc / 2 + L + W / 2 * np.sin(θ_max),
                q1 * get_safety_distance_headway(vm) + q2 * get_breaking_distance(vm, vf)
            )
        else:
            return vm - vf * tlc / 2 + L - W / 2 * np.sin(θ_max) + q1 * get_safety_distance_headway(
                vm) + q2 * get_breaking_distance(vm, vf) - l
    else:
        return q1 * get_safety_distance_headway(vm) + q2 * l


def single_safety_payoff(dk, sk):
    if dk >= sk:
        return 1
    elif l < dk < sk:
        return np.log(dk / sk + 1) / np.log(2)
    else:
        return -np.inf


def safety_payoff(x0, x1, x2, x3, v0, v1, v2, v3, lane_change):
    if lane_change == 2:
        s01 = (
                q1 * get_safety_distance_headway(v0) + q2 * get_breaking_distance(v0, v1)
        ) if v0 > v1 else q1 * get_safety_distance_headway(v0) + q2 * l
        s23 = (
                q1 * get_safety_distance_headway(v2) + q2 * get_breaking_distance(v2, v3)
        ) if v2 > v3 else q1 * get_safety_distance_headway(v2) + q2 * l

        d01 = np.abs(x0 - x1)
        d23 = np.abs(x2 - x3)

        u0 = single_safety_payoff(d01, s01)
        u3 = single_safety_payoff(d23, s23)
        return u0, u3

    s01 = v0 - v1 * tlc / 2 + L + W / 2 * np.sin(θ_max)
    s02 = (
            v0 - v2 * tlc / 2 + L - W / 2 * np.sin(θ_max) +
            q1 * get_safety_distance_headway(v0) + q2 * get_breaking_distance(v0, v1)
    ) if v0 > v2 else q1 * get_safety_distance_headway(v0) + q2 * l
    s03 = (
            v3 - v0 * 3 / 4 * tlc + L - l +
            q1 * get_safety_distance_headway(v0) + q2 * get_breaking_distance(v0, v1)
    ) if v3 > v0 else q1 * get_safety_distance_headway(v3) + q2 * l

    d01 = np.abs(x0 - x1)
    d02 = np.abs(x0 - x2)
    d03 = np.abs(x0 - x3)

    u0 = (single_safety_payoff(d01, s01) + single_safety_payoff(d02, s02) + single_safety_payoff(d03, s03)) / 3
    u3 = single_safety_payoff(d03, s03)

    return u0, u3


x0, x1, x2, x3 = 4, 14, 20, -6
v0, v1, v2, v3 = 10, 5, 15, 10

print(safety_payoff(x0, x1, x2, x3, v0, v1, v2, v3, lane_change=1))
print(safety_payoff(x0, x1, x2, x3, v0, v1, v2, v3, lane_change=2))
print()
