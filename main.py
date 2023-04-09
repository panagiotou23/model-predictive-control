import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import solve_ivp

from controller import mpc_controller, straight_line_controller
from dynamics import KinematicBicyclePacejka
from road import Road
from simulation import plot_results, simulate_motion, plot_trajectory

n, dt, v_ref = 3, 0.1, 1.0

model = KinematicBicyclePacejka()
road = Road()


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
    t_start, t_end, t_step = 0, 2, 0.1

    res = pacejka_model(t_start, t_end, t_step)
    #
    # t = res.t
    # x = res.y[0, :]
    # y = res.y[1, :]
    # φ = res.y[2, :]
    # vx = res.y[3, :]
    # vy = res.y[4, :]
    # ω = res.y[5, :]
    # plt.figure()
    # plt.title("Test")
    # plt.quiver(x, y, np.cos(φ), np.sin(φ), scale=100, color='r', width=0.002)
    # plt.quiver(x, y, np.cos(φ), np.sin(φ), scale=100, color='y', width=0.002)
    # plt.plot(x, y, 'r')
    #
    #
    # theta = np.linspace(0, 2 * np.pi, 1000)
    # radius = 5
    # center = [0, 0]
    #
    # xc = radius * np.cos(theta) + center[0]
    # yc = radius * np.sin(theta) + center[1]
    # yc += 5
    # plt.plot(xc, yc, 'k')
    #
    # manager = plt.get_current_fig_manager()
    # manager.full_screen_toggle()
    # plt.show()

    animations_and_graphs(res)
