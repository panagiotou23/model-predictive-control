from ctypes import Union

import numpy as np
from alpaqa.casadi_loader import generate_and_compile_casadi_problem
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from car_dynamics import KinematicBicyclePacejka
from controller import mpc_controller, MPCController
from road import Road
from simulation import simulate_motion, plot_trajectory
import alpaqa as pa
import casadi as cs

n, dt, v_ref = 3, 0.1, 1.0

model = KinematicBicyclePacejka()
road = Road()


def alpaqa_test():
    x = cs.SX.sym("x")
    y = cs.SX.sym("y")
    c = cs.SX.sym("c", 3)

    X = cs.vertcat(x, y)

    f_expr = (x + y) ** 2 + c[0] * x * y - c[1] * x + c[2] * y
    f = cs.Function("f", [X, c], [f_expr])

    g_expr = x + y
    g = cs.Function("g", [X, c], [g_expr])

    prob = generate_and_compile_casadi_problem(f, g)
    prob.C.lowerbound = - 10 * np.ones((2, 1))
    prob.C.upperbound = 10 * np.ones((2, 1))
    prob.D.lowerbound = np.zeros((1, 1))

    prob.param = np.array([1, 1, 3])
    # %% NLP solver
    from datetime import timedelta

    solver = pa.ALMSolver(
        alm_params={
            'ε': 1e-4,
            'δ': 1e-4,
            'Σ_0': 1e5,
            'max_time': timedelta(seconds=0.5),
        },
        inner_solver=pa.StructuredPANOCLBFGSSolver(
            panoc_params={
                'stop_crit': pa.ProjGradNorm2,
                'max_time': timedelta(seconds=0.2),
                'hessian_step_size_heuristic': 15,
            },
            lbfgs_params={'memory': 12},
        ),
    )

    sol = np.array([2, -1])
    λ = np.zeros((1, 1))
    sol, λ, stats = solver(prob, sol, λ)

    # Print some solver statistics
    print(stats['status'], stats['outer_iterations'],
          stats['inner']['iterations'], stats['elapsed_time'],
          stats['inner_convergence_failures'])

    print(np.linalg.norm(λ))
    print(sol)


def alpaqa_vehicle_test():
    n, dt, v_ref = 3, 0.1, 1.0
    # Test
    N_sim = 100
    N_horiz = 12
    u_dim = 2
    centerline_size = 10000
    f_d = model.dynamics()
    y_null, u_null = model.X_0, model.u_0

    max_drive, max_steer = 1.0, 0.32
    length, axis_front, axis_rear, front, rear, width, height, mass, inertia = \
        9.7e-2, 4.7e-2, 5e-2, 0.09, 0.07, 8e-2, 5.5e-2, 0.1735, 18.3e-5
    bf, cf, df, br, cr, dr = 0.268, 2.165, 3.47, 0.242, 2.38, 2.84
    cm1, cm2, cr0, cr1, cr2 = 0.266, 0.1, 0.1025, 0.1629, 0.0011

    param = np.array([
        length,  # length
        axis_front,  # axis_front
        axis_rear,  # axis_rear
        front,  # front
        rear,  # rear
        width,  # width
        height,  # height
        mass,  # mass
        inertia,  # inertia
        max_steer,  # max_steer
        max_drive,  # max_drive
        bf,  # bf
        cf,  # cf
        df,  # df
        br,  # br
        cr,  # cr
        dr,  # dr
        cm1,  # cm1
        cm2,  # cm2
        cr0,  # cr0
        cr1,  # cr1
        cr2  # cr2
    ]).T
    centerline_val = np.array([[0, i / 500 - 0.1] for i in range(centerline_size)]).ravel()

    L_cost = model.generate_cost_fun(centerline_size)  # stage cost
    y_init = cs.SX.sym("y_init", *y_null.shape)  # initial state
    centerline = cs.SX.sym("centerline", centerline_size * 2, )
    U = cs.SX.sym("U", u_dim * N_horiz)  # control signals over horizon
    mpc_param = cs.vertcat(y_init, model.params, centerline)  # all parameters
    U_mat = model.input_to_matrix(U)  # Input as dim by N_horiz matrix

    # Cost
    mpc_sim = model.simulate(N_horiz, y_init, U_mat, model.params)
    mpc_cost = 0
    for n in range(N_horiz):  # Apply the stage cost function to each stage
        y_n = mpc_sim[:, n]
        u_n = U_mat[:, n]
        mpc_cost += L_cost(y_n, u_n, v_ref, centerline)
    mpc_cost_fun = cs.Function('f_mpc', [U, mpc_param], [mpc_cost])

    # Constraints
    constr = []
    for n in range(N_horiz):  # For each stage,
        y_n = mpc_sim[:, n]
        constr += [y_n[0] ** 2 - 20]
        constr += [y_n[1] ** 2 - 1]
        constr += [y_n[2] ** 2 - 1]
        constr += [y_n[3] ** 2 - 2]
        constr += [y_n[4] ** 2 - 1]
        constr += [y_n[5] ** 2 - 0.1]
    mpc_constr_fun = cs.Function("g", [U, mpc_param], [cs.vertcat(*constr)])

    mpc_cost_fun.disp()
    print()
    mpc_constr_fun.disp()
    print()

    prob = generate_and_compile_casadi_problem(mpc_cost_fun, mpc_constr_fun)
    prob.C.lowerbound = np.tile([-max_drive, -max_steer], N_horiz)
    prob.C.upperbound = np.tile([max_drive, max_steer], N_horiz)
    # prob.D.lowerbound = np.zeros((6 * N_horiz,))

    y_n = model.X_0
    y_mpc = np.empty((y_n.shape[0], N_sim))
    prob.param = np.concatenate((y_n, param, centerline_val))
    controller = MPCController(model, prob, N_horiz)
    for n in range(N_sim):
        pos = np.array((y_n[0], y_n[1])).reshape(2, 1)

        print("Nearest point")
        print(model.find_nearest_point(
            centerline_size,
            pos,
            centerline_val.reshape((centerline.shape[0] // 2, 2))
        ))
        print("Errors")
        print(model.compute_errors(
            centerline_size,
            pos,
            y_n[2],
            centerline_val
        ))

        # Solve the optimal control problem
        U = controller(y_n)
        u_n = model.input_to_matrix(U)[:, 0]

        # Apply the first optimal control signal to the system and simulate for
        # one time step, then update the state
        y_n = model.simulate(1, y_n, u_n, param).T
        y_mpc[:, n] = y_n

        print("X")
        print(y_n)
        print("u")
        print(u_n)
        print()

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

    alpaqa_vehicle_test()
