import numpy as np
from scipy.optimize import minimize
import alpaqa as pa
from datetime import timedelta

from road import Road


class MPCController:

    def __init__(self, model, problem, N_horiz):
        self.model = model
        self.problem = problem

        self.N_horiz = 12 if N_horiz is None else N_horiz
        self.u_dim = 2
        self.tot_it = 0
        self.failures = 0
        self.U = np.tile([1, 0], N_horiz)
        self.λ = np.zeros((6 * N_horiz,))

        self.solver = pa.ALMSolver(
            alm_params={
                'ε': 1e-4,
                'δ': 1e-4,
                'Σ_0': 1e5,
                'max_time': timedelta(seconds=0.5),
                'max_iter': 1000,
            },
            inner_solver=pa.StructuredPANOCLBFGSSolver(
                panoc_params={
                    'stop_crit': pa.ProjGradNorm2,
                    'max_time': timedelta(seconds=0.2),
                    'max_iter': 1000,
                    'hessian_step_size_heuristic': 15,
                },
                lbfgs_params={'memory': N_horiz},
            ),
        )

    def __call__(self, y_n):
        y_n = np.array(y_n).ravel()
        # Set the current state as the initial state
        self.problem.param[:y_n.shape[0]] = y_n
        # Solve the optimal control problem
        # (warm start using the previous solution and Lagrange multipliers)
        self.U, self.λ, stats = self.solver(self.problem, self.U, self.λ)
        # Print some solver statistics
        print(stats['status'], stats['outer_iterations'],
              stats['inner']['iterations'], stats['elapsed_time'],
              stats['inner_convergence_failures'])
        # print(self.U)
        self.tot_it += stats['inner']['iterations']
        self.failures += stats['status'] != pa.SolverStatus.Converged
        # Print the Lagrange multipliers, shows that constraints are active
        # print(np.linalg.norm(self.λ))
        # Return the optimal control signal for the first time step
        # return self.model.input_to_matrix(self.U)[:, 0]
        return self.U


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
