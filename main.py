from ctypes import Union

import numpy as np
from alpaqa.casadi_loader import generate_and_compile_casadi_problem
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from car_dynamics import KinematicBicyclePacejka
from controller import mpc_controller, straight_line_controller, MPCController
from road import Road
from simulation import plot_results, simulate_motion, plot_trajectory
import alpaqa as pa
import casadi as cs

n, dt, v_ref = 3, 0.1, 1.0

model = KinematicBicyclePacejka()
road = Road()


def alpaqa_test():
    # Test
    model = KinematicBicyclePacejka()

    N_sim = 180
    N_horiz = 2
    u_dim = 1
    f_d = model.dynamics()
    y_null, u_null = model.state_0, model.u_0
    param = np.array([
        9.7e-2,  # length
        4.7e-2,  # axis_front
        5e-2,  # axis_rear
        0.09,  # front
        0.07,  # rear
        8e-2,  # width
        5.5e-2,  # height
        0.1735,  # mass
        18.3e-5,  # inertia
        0.32,  # max_steer
        1.0,  # max_drive
        0.268,  # bf
        2.165,  # cf
        3.47,  # df
        0.242,  # br
        2.38,  # cr
        2.84,  # dr
        0.266,  # cm1
        0.1,  # cm2
        0.1025,  # cr0
        0.1629,  # cr1
        0.0011  # cr2
    ]).T

    L_cost = model.generate_cost_fun()  # stage cost
    y_init = cs.SX.sym("y_init", *y_null.shape)  # initial state
    U = cs.SX.sym("U", u_dim * N_horiz)  # control signals over horizon
    mpc_param = cs.vertcat(y_init, model.params)  # all parameters
    U_mat = model.input_to_matrix(U)  # Input as dim by N_horiz matrix

    # Cost
    mpc_sim = model.simulate(N_horiz, y_init, U_mat, model.params)
    mpc_cost = 0
    for n in range(N_horiz):  # Apply the stage cost function to each stage
        y_n = mpc_sim[:, n]
        mpc_cost += L_cost(y_n[2], v_ref)
    mpc_cost_fun = cs.Function('f_mpc', [U, mpc_param], [mpc_cost])

    # Constraints
    constr = []
    for n in range(N_horiz):  # For each stage,
        y_n = mpc_sim[:, n]
        constr += [y_n[0] ** 2 - 20]
        constr += [y_n[1] ** 2 - 0.1]
        constr += [y_n[2] ** 2 - 0.1]
        constr += [y_n[3] ** 2 - 2]
        constr += [y_n[4] ** 2 - 0.1]
        constr += [y_n[5] ** 2 - 0.1]
    mpc_constr_fun = cs.Function("g", [U, mpc_param], [cs.vertcat(*constr)])

    mpc_cost_fun.disp()
    print()
    mpc_constr_fun.disp()
    print()

    prob = generate_and_compile_casadi_problem(mpc_cost_fun, mpc_constr_fun)
    prob.C.lowerbound = -1 * np.ones((u_dim * N_horiz,))
    prob.C.upperbound = +1 * np.ones((u_dim * N_horiz,))
    prob.D.lowerbound = np.zeros((6 * N_horiz,))

    y_n = model.state_0
    y_mpc = np.empty((y_n.shape[0], N_sim))
    controller = MPCController(model, prob, N_horiz)
    for n in range(N_sim):
        # Solve the optimal control problem
        u_n = controller(y_n)
        # Apply the first optimal control signal to the system and simulate for
        # one time step, then update the state
        y_n = model.simulate(1, y_n, [u_n[0], 0], param).T
        y_mpc[:, n] = y_n
        print(u_n)

    print(controller.tot_it, controller.failures)

    plt.figure()
    plt.plot(
        y_mpc[0, :],
        y_mpc[1, :]
    )
    plt.show()


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

    plot_trajectory(x, y, φ, u, "Pacejka")

    simulate_motion(model, x, y, φ, vx, vy, u, t, "Pacejka")


if __name__ == '__main__':
    # t_start, t_end, t_step = 0, 2, 0.1
    #
    # res = pacejka_model(t_start, t_end, t_step)
    #
    # animations_and_graphs(res)

    alpaqa_test()
