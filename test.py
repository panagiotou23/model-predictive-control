import casadi as cs
import numpy as np
from alpaqa.casadi_loader import generate_and_compile_casadi_problem
import alpaqa as pa

if __name__ == '__main__':
    size = 10
    x = cs.SX.sym("x")

    line = cs.SX.sym("line", size)
    dist = cs.reshape(cs.sqrt((line - x) ** 2), (size, 1))

    # Find the index of the minimum value in the dist vector
    min_val = dist[0]
    idx = 0
    for i in range(1, size):
        is_min = cs.if_else(dist[i] < min_val, 1, 0)
        min_val = cs.if_else(is_min, dist[i], min_val)
        idx = cs.if_else(is_min, i, idx)

    f = cs.Function("f", [x, line], [idx])
    g = cs.Function("x", [x, line], [x ** 2])

    prob = generate_and_compile_casadi_problem(f, g)
    prob.param = np.array([
        0,
        1,
        2,
        3,
        4,
        5,
        6,
        7,
        8,
        9
    ])

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
    x = np.array([1])
    x, λ, stats = solver(prob, x, λ)

    print(x)
    print(stats)
