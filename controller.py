import numpy as np
from scipy.optimize import minimize
import alpaqa as pa

from road import Road


def mpc_controller_alpaqa(model, current_state, road, target_velocity, N=2, dt=0.1):
    # Define the objective function
    def cost_fn(u_flat, *args):
        states, N, dt, current_state, road, target_velocity = args
        cost = 0
        u = u_flat.reshape(2, N)
        cte_weight = 100  # tuning parameter for CTE penalty
        heading_weight = 10  # tuning parameter for heading penalty
        velocity_weight = 10  # tuning parameter for velocity penalty
        for i in range(N):
            if i == 0:
                x = np.copy(current_state)
            else:
                x += model(states[i - 1], u[:, i - 1], dt * (i - 1)) * dt
            states[i] = x

            _, heading_error, cte = road.compute_errors(
                np.array(x[0], x[1]),
                x[2]
            )

            velocity = np.sqrt(x[3] ** 2 + x[4] ** 2)
            # Add penalty terms to the cost function
            cost += cte_weight * cte ** 2 + heading_weight * heading_error ** 2 \
                + velocity_weight * velocity
        return cost

    # Initialize states and control inputs
    states = [None] * N
    u_flat = np.zeros((N * 2,))

    # Solve the optimization problem


    # Return the first optimal control inputs
    return u[:, 0]


def mpc_controller(model, current_state, road, target_velocity, N=2, dt=0.1):
    # Define the objective function
    def cost_fn(u_flat, *args):
        states, N, dt, current_state, road, target_velocity = args
        cost = 0
        u = u_flat.reshape(2, N)
        cte_weight = 100  # tuning parameter for CTE penalty
        heading_weight = 10  # tuning parameter for heading penalty
        velocity_weight = 10  # tuning parameter for velocity penalty
        for i in range(N):
            if i == 0:
                x = np.copy(current_state)
            else:
                x += model(states[i - 1], u[:, i - 1], dt * (i - 1)) * dt
            states[i] = x

            _, heading_error, cte = road.compute_errors(
                np.array(x[0], x[1]),
                x[2]
            )

            velocity = np.sqrt(x[3] ** 2 + x[4] ** 2)
            # Add penalty terms to the cost function
            cost += cte_weight * cte ** 2 + heading_weight * heading_error ** 2 \
                + velocity_weight * velocity
        return cost

    # Initialize states and control inputs
    states = [None] * N
    u_flat = np.zeros((N * 2,))

    # Solve the optimization problem
    result = minimize(cost_fn, u_flat, args=(states, N, dt, current_state, road, target_velocity))
    u = result.x.reshape(2, N)

    # Return the first optimal control inputs
    return u[:, 0]


def mpc_controller_initial(model, current_state, target_state, N=2, dt=0.1):
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
    u = result.x.reshape(2, N)

    # Return the first optimal control inputs
    return u[:, 0]


def straight_line_controller(current_state: np.ndarray, road: Road):
    pos = np.array([
        current_state[0], current_state[1]
    ])
    idx, closest_point = road.find_nearest_point(
        vehicle_position=pos
    )
    cte, heading_error, pos_error = road.compute_errors(
        pos,
        current_state[2]
    )
    print("Current point \t(", current_state[0], ", ", current_state[1], "), \tHeading ", current_state[2])
    print("Closest point \t(", closest_point[0], ", ", closest_point[1], ")")
    print("CTE = ", cte, "\tPositional error = ", pos_error, "\tHeading error = ", heading_error)
    print()
    return np.array([
        1,
        0
    ])
