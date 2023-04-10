from ctypes import Union

import numpy as np
from alpaqa.casadi_loader import generate_and_compile_casadi_problem
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from car_dynamics import KinematicBicyclePacejka
from controller import mpc_controller, straight_line_controller
from road import Road
from simulation import plot_results, simulate_motion, plot_trajectory
import alpaqa as pa
import casadi as cs


n, dt, v_ref = 3, 0.1, 1.0

model = KinematicBicyclePacejka()
road = Road()

def alpaqa_test():
    # Test
    vehicle_model = KinematicBicyclePacejka()

    vehicle_f_d = vehicle_model.dynamics()
    vehicle_y_null, vehicle_u_null = vehicle_model.state_0, vehicle_model.u_0
    vehicle_param = np.array([
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

    N_sim = 180
    u_test = np.array([[1, i/N_sim] for i in range(0, N_sim)]).T
    y_test = vehicle_model.simulate(N_sim, vehicle_y_null, u_test, vehicle_param)

    y_res = np.array(y_test)

    x = np.array(y_res[0:N_sim * 6:6]).T
    y = np.array(y_res[1:N_sim * 6:6]).T

    plt.figure()
    plt.plot(x, y, 'k')
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