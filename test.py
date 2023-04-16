import alpaqa as pa
import numpy as np

import casadi as cs
from alpaqa.casadi_loader import generate_and_compile_casadi_problem

if __name__ == '__main__':

    import casadi as ca

    # create a symbolic variable of type SX
    x = ca.SX.sym('x', 3)

    # round the first element of x to the nearest integer
    x0_rounded = ca.ceil(x[0])

    # cast the rounded expression to an integer
    x0_int = int(x0_rounded)

    # create another symbolic variable of type SX
    y = ca.SX.sym('y', 3)

    # access the second element of y using the integer value of x[0]
    y1 = y[x0_int]


    size = 10
    x = cs.SX.sym("x", 2)

    line_flat = cs.SX.sym("line", 2 * size, )

    # Calculate the distance by summing the squared differences of each coordinate separately
    line = line_flat.reshape((line_flat.shape[0] // 2, 2))
    diff = line - cs.repmat(x.T, size, 1)
    squared_diff = diff**2
    sum_squared_diff = cs.sum2(squared_diff)
    dist = cs.sqrt(sum_squared_diff)

    # Find the index of the minimum value in the dist vector
    min_val = dist[0]
    idx = 0
    for i in range(1, size):
        is_min = cs.if_else(dist[i] < min_val, 1, 0)
        min_val = cs.if_else(is_min, dist[i], min_val)
        idx = cs.if_else(is_min, i, idx)

    f = cs.Function("f", [x, line_flat], [idx])
    g = cs.Function("x", [x, line_flat], [cs.sum1(x)])

    prob = generate_and_compile_casadi_problem(f, g)

    prob.param = np.array([
        [0, 0],
        [1, 1],
        [2, 2],
        [3, 3],
        [4, 4],
        [5, 5],
        [6, 6],
        [7, 7],
        [8, 8],
        [9, 9],
    ]).ravel()

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

    λ = np.array([1])
    x = np.array([1, 1.5])
    x, λ, stats = solver(prob, x, λ)

    print(x)
    print(stats)
