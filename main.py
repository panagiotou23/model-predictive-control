from ctypes import Union

import numpy as np
from alpaqa.casadi_loader import generate_and_compile_casadi_problem
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from controller import mpc_controller, straight_line_controller
from dynamics import KinematicBicyclePacejka
from road import Road
from simulation import plot_results, simulate_motion, plot_trajectory
import alpaqa as pa
import casadi as cs


n, dt, v_ref = 3, 0.1, 1.0

model = KinematicBicyclePacejka()
road = Road()

def alpaqa_test():
    x1 = cs.SX.sym("x1")

    f_expr = (1 - x1) ** 2
    g_expr = cs.vertcat(
        x1
    )

    x = cs.vertcat(x1)
    f = cs.Function("f", [x], [f_expr])
    g = cs.Function("g", [x], [g_expr])

    prob = generate_and_compile_casadi_problem(f, g)

    prob.D.lowerbound = [-np.inf]
    prob.D.upperbound = [0] 

    inner_solver = pa.StructuredPANOCLBFGSSolver(
        panoc_params={
            'max_iter': 1000,
            'stop_crit': pa.PANOCStopCrit.ApproxKKT,
        },
        lbfgs_params={
            'memory': 10,
        },
    )

    solver = pa.ALMSolver(
        alm_params={
            'ε': 1e-10,
            'δ': 1e-10,
            'Σ_0': 0,
            'σ_0': 2,
            'Δ': 20,
        },
        inner_solver=inner_solver
    )

    x0 = np.array([-1.5])
    y0 = np.zeros((prob.m,))

    x_sol, y_sol, stats = solver(prob, x0, y0)

    # Print the results
    print(stats["status"])
    print(f"Solution:      {x_sol}")
    print(f"Multipliers:   {y_sol}")
    print(f"Cost:          {prob.eval_f(x_sol):.5f}")


def pacejka_model(t_start, t_end, t_step):

    def obj_fun(t, x):
        u = mpc_controller(
            model=model,
            current_state=x,
            road=road,
            target_velocity=1,
            N=n,
            dt=dt
        )
        # u = straight_line_controller(
        #     current_state=x,
        #     road=road
        # )

        cost = model(x, u, t)
        _, closest_point = road.find_nearest_point(
            np.array(x[0], x[1])
        )
        print("(", x[0], ", ", x[1], ")")
        print("(", closest_point[0], ", ", closest_point[1], ")")
        print("v = ", np.sqrt(x[3] ** 2 + x[4] ** 2))
        print("u = (", u[0], ", ", u[1], ")")
        print()
        print(t)
        print()
        return cost

    x0 = np.array([
        0,  # x
        0,  # y
        0,  # φ
        1,  # vx
        0,  # vy
        0  # ω
    ])

    t_span = [t_start, t_end]
    t_step = np.arange(t_start, t_end, t_step)
    res = solve_ivp(
        obj_fun,
        t_span,
        x0,
        t_eval=t_step
    )

    return res


def animations_and_graphs(res):
    t = res.t
    x = res.y[0, :]
    y = res.y[1, :]
    φ = res.y[2, :]
    vx = res.y[3, :]
    vy = res.y[4, :]
    ω = res.y[5, :]

    u = np.zeros([2, t.size])
    for i in range(t.size):
        u[:, i] = mpc_controller(
            model=model,
            current_state=res.y[:, i],
            road=road,
            target_velocity=v_ref,
            N=n,
            dt=dt
        )

    # plot_results(t, x, y, φ, vx, vy, ω, u, "Pacejka")

    # plot_trajectory(x, y, φ, u, "Pacejka")

    simulate_motion(model, x, y, φ, vx, vy, u, t, "Pacejka")

if __name__ == '__main__':
    # t_start, t_end, t_step = 0, 2, 0.1
    #
    # res = pacejka_model(t_start, t_end, t_step)
    #
    # animations_and_graphs(res)

    alpaqa_test()