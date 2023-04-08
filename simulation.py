import matplotlib.pyplot as plt
import numpy as np


def plot_results(t, x, y, φ, vx, vy, ω, u, title):
    plt.figure()
    plt.title(title)

    plt.subplot(321)
    plt.plot(t, x)
    plt.plot(t, y)
    plt.plot(t, φ)
    plt.legend(["$x$", "$y$", "$\phi$"])

    plt.subplot(322)
    plt.plot(t, vx)
    plt.plot(t, vy)
    plt.plot(t, ω)
    plt.legend(["$u_x$", "$u_y$", "$\omega$"])

    plt.subplot(323)
    plt.plot(t, u[0, :])
    plt.plot(t, u[1, :])
    plt.legend(["$d$", "$\delta$"])

    plt.subplot(324)
    plt.plot(t, np.sqrt(np.multiply(vx, vx) + np.multiply(vy, vy)))
    plt.legend(["$|u|$"])

    δ = u[1, :]
    plt.subplot(313)
    plt.quiver(x, y, np.cos(φ), np.sin(φ), scale=100, color='r', width=0.002)
    # plt.quiver(x, y, vx, vy, scale=100, color='c', width=0.002)
    plt.quiver(x, y, np.cos(φ + δ), np.sin(φ + δ), scale=100, color='y', width=0.002)
    plt.plot(x, y, 'r')
    plt.legend([
        "$\phi$",
        # "$u$",
        "$\delta$",
        "position"
    ])

    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


def plot_trajectory(x, y, φ, u, title):
    δ = u[1, :]
    plt.figure()
    plt.title(title)
    plt.quiver(x, y, np.cos(φ), np.sin(φ), scale=100, color='r', width=0.002)
    plt.quiver(x, y, np.cos(φ + δ), np.sin(φ + δ), scale=100, color='y', width=0.002)
    plt.plot(x, y, 'r')
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    plt.show()


def rotate_point(point_x, point_y, origin_x, origin_y, angle):
    return [
        np.cos(angle) * (point_x - origin_x) - np.sin(angle) * (point_y - origin_y) + origin_x,
        np.sin(angle) * (point_x - origin_x) + np.cos(angle) * (point_y - origin_y) + origin_y
    ]


def plot_car(model, x, y, φ):
    lf, lr = model.params.axis_front, model.params.axis_rear
    w = model.params.width

    x_left = x - lr
    x_right = x + lf
    y_lower = y - w / 2
    y_upper = y + w / 2

    car_corners = np.array([
        rotate_point(x_left, y_lower, x, y, φ),
        rotate_point(x_left, y_upper, x, y, φ),
        rotate_point(x_right, y_upper, x, y, φ),
        rotate_point(x_right, y_lower, x, y, φ),
        rotate_point(x_left, y_lower, x, y, φ)
    ])
    plt.plot(car_corners[:, 0], car_corners[:, 1])


def simulate_motion(model, x, y, φ, vx, vy, u, t, title):
    offset = 0.5
    x_lim = [
        np.min(x) - offset,
        np.max(x) + offset
    ]
    y_lim = [
        np.min(y) - offset,
        np.max(y) + offset
    ]
    δ = u[1, :]

    plt.figure()
    plt.title(title)
    manager = plt.get_current_fig_manager()
    manager.full_screen_toggle()
    for i in range(x.size):
        plot_car(model, x[i], y[i], φ[i])
        plt.plot(x[:i], y[:i], 'k')
        plt.scatter(x[i], y[i])
        plt.quiver(x[i], y[i], np.cos(φ[i]), np.sin(φ[i]), scale=100, color='r', width=0.002)
        front_x, front_y = rotate_point(x[i] + model.params.axis_front, y[i], x[i], y[i], φ[i])
        plt.quiver(front_x, front_y, np.cos(φ[i] + δ[i]), np.sin(φ[i] + δ[i]), scale=100, color='y', width=0.002)
        plt.xlim(x_lim)
        plt.ylim(y_lim)
        plt.figtext(0.15, 0.9, '|| Speed %.2f \n|| t = %.2f' % (vx[i], t[i]), fontsize=10)
        plt.pause(0.00001)
        plt.clf()
    plt.show()
