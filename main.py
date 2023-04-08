import alpaqa as pa
import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import minimize

from dynamics import KinematicBicyclePacejka
from simulation import simulate_motion


def mpc_controller(model, current_state, target_state, N=2, dt=0.1):
    # Define the objective function
    def cost_fn(u_flat, *args):
        states, target_state, N, dt = args
        cost = 0
        u = u_flat.reshape(2, N)
        for i in range(N):
            if i == 0:
                x = np.copy(current_state)
            else:
                x += model(states[i - 1], u[:, i - 1], dt * (i - 1)) * dt
            states[i] = x
            cost += np.sum((x - target_state) ** 2)
        return cost

    # Initialize states and control inputs
    states = [None] * N
    u_flat = np.zeros((N * 2,))

    # Solve the optimization problem
    result = minimize(cost_fn, u_flat, args=(states, target_state, N, dt))
    # result = pa.PANOCSolver(cost_fn )
    u = result.x.reshape(2, N)

    # Return the first optimal control inputs
    return u[:, 0]


def pacejka_model(t_start, t_end, t_step):
    model = KinematicBicyclePacejka()

    def obj_fun(t, x):
        # u = calc_input(t, x)
        n, dt, v_ref = 2, 0.1, 1.0
        u = mpc_controller(
            model=model,
            current_state=x,
            target_state=np.array([x[0] + v_ref * dt * n, 0, 0, v_ref, 0, 0]),
            N=n,
            dt=dt
        )
        cost = model(x, u, t)
        return cost

    x0 = np.array([
        0,  # x
        0,  # y
        0,  # φ
        0.1,  # vx
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

    return model, res


def animations_and_graphs(model, res):
    t = res.t
    x = res.y[0, :]
    y = res.y[1, :]
    φ = res.y[2, :]
    vx = res.y[3, :]
    vy = res.y[4, :]
    ω = res.y[5, :]

    u = np.zeros([2, t.size])
    n, dt, v_ref = 2, 0.1, 1.0
    for i in range(t.size):
        u[:, i] = mpc_controller(
            model=model,
            current_state=res.y[:, i],
            target_state=np.array([res.y[0, i] + v_ref * dt * n, 0, 0, v_ref, 0, 0]),
            N=n,
            dt=dt
        )

    # plot_results(t, x, y, φ, vx, vy, ω, u, "Pacejka")
    #
    # plot_trajectory(x, y, φ, u, "Pacejka")
    #
    simulate_motion(model, x, y, φ, vx, vy, u, t, "Pacejka")


class Road:

    def __init__(
            self,
            left: np.array = None,
            right: np.array = None,
            center: np.array = None
    ) -> None:
        self.left = [[i / 100, -5] for i in range(1000)] if left is None else left
        self.right = [[i / 100, 5] for i in range(1000)] if right is None else right
        self.center = [[i / 100, 0] for i in range(1000)] if center is None else center


if __name__ == '__main__':
    t_start, t_end, t_step = 0, 10, 0.01

    [model, res] = pacejka_model(t_start, t_end, t_step)

    animations_and_graphs(model, res)
