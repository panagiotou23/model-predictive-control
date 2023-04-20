import alpaqa as pa
import casadi as cs
import numpy as np
from alpaqa.casadi_loader import generate_and_compile_casadi_problem
from matplotlib import pyplot as plt

from car_dynamics import KinematicBicyclePacejka
from controller import MPCController


def get_centerline(size, is_straight=True):
    if is_straight:
        return np.array([[i / 10 - 0.1, 0] for i in range(size)])
    else:
        theta = np.linspace(0, 2 * np.pi, size)
        radius = 5
        center = [0, 0]

        x = radius * np.cos(theta) + center[0]
        y = radius * np.sin(theta) + center[1]
        y += 5
        return np.stack((x, y), axis=1)


def create_casadi_problem(model, N_horiz, centerline_size, v_ref, max_drive, max_steer):
    L_cost = model.generate_stage_cost_fun(centerline_size, v_ref)  # stage cost
    y_init = cs.SX.sym("y_init", model.X.shape)  # initial state
    centerline = cs.SX.sym("centerline", centerline_size * 2, 1)
    U = cs.SX.sym("U", model.u.shape[0] * N_horiz)  # control signals over horizon
    mpc_param = cs.vertcat(y_init, model.params, centerline)  # all parameters
    U_mat = model.input_to_matrix(U)  # Input as dim by N_horiz matrix

    # Cost
    mpc_sim = model.simulate(N_horiz, y_init, U_mat, model.params)
    mpc_cost = 0
    for n in range(N_horiz):  # Apply the stage cost function to each stage
        y_n = mpc_sim[:, n]
        u_n = U_mat[:, n]
        mpc_cost += L_cost(y_n, u_n, centerline)
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

    prob = pa.casadi_loader.generate_and_compile_casadi_problem(mpc_cost_fun, mpc_constr_fun)
    prob.C.lowerbound = np.tile([-max_drive, -max_steer], N_horiz)
    prob.C.upperbound = np.tile([max_drive, max_steer], N_horiz)
    # prob.D.lowerbound = np.zeros((6 * N_horiz,))

    return prob


def alpaqa_vehicle_test():
    model = KinematicBicyclePacejka()

    n, dt, v_ref = 3, 0.1, 1.
    # Test
    N_sim = 400
    N_horiz = 12
    u_dim = 2
    centerline_size = 100
    f_d = model.dynamics()
    y_null, u_null = np.array([
        0,  # x
        0,  # y
        0,  # φ
        .5,  # vx
        0,  # vy
        0  # ω
    ]), \
        np.array([0, 0])

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
    centerline_val = get_centerline(centerline_size, is_straight=False)
    centerline_val = centerline_val.ravel(order='F')

    prob = create_casadi_problem(model, N_horiz, centerline_size, v_ref, max_drive, max_steer)

    y_n = y_null
    y_mpc = np.empty((y_n.shape[0], N_sim))
    prob.param = np.concatenate((y_n, param, centerline_val))
    controller = MPCController(model, prob, N_horiz)
    for n in range(N_sim):
        pos = np.array((y_n[0], y_n[1])).reshape(2, 1)
        nearest_point, previous_point, next_point = model.find_nearest_point(
            centerline_size,
            pos,
            centerline_val.reshape((centerline_val.shape[0] // 2, 2), order='C')
        )
        cte, heading_error, pos_error = model.compute_errors(
            centerline_size,
            pos,
            y_n[2],
            centerline_val
        )
        print("Pos: ", pos.reshape(2, ))
        print("Nearest point: ", nearest_point, "\tNext point: ", next_point, "\tPrevious point: ", previous_point)
        print("CTE: ", cte, "\tHeading Error: ", heading_error, "\tPos Error: ", pos_error)
        print()

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

    centerline_val = centerline_val.reshape((centerline_val.shape[0] // 2, 2), order='F')

    plt.figure()
    plt.plot(
        centerline_val[:, 0],
        centerline_val[:, 1]
    )
    plt.plot(
        y_mpc[0, :],
        y_mpc[1, :]
    )
    plt.show()


if __name__ == '__main__':
    # t_start, t_end, t_step = 0, 2, 0.1
    #
    # res = pacejka_model(t_start, t_end, t_step)
    #
    # animations_and_graphs(res)

    alpaqa_vehicle_test()
