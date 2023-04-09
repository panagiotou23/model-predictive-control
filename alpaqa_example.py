## @example mpc/python/hanging-chain/hanging-chain-mpc.py
# This example shows how to call the alpaqa solver from Python code, applied
# to a challenging model predictive control problem.

# %% Hanging chain MPC example

import casadi as cs
import matplotlib
import numpy as np
import os
from os.path import join, dirname
import sys

from typing import Union
import casadi as cs
import numpy as np
from alpaqa.casadi_loader import generate_and_compile_casadi_problem
from casadi import vertcat as vc

import matplotlib.animation as animation


class HangingChain:
    def __init__(self, N: int, dim: int):
        self.N = N
        self.dim = dim

        self.y1 = cs.SX.sym("y1", dim * N, 1)  # state: balls 1→N positions
        self.y2 = cs.SX.sym("y2", dim * N, 1)  # state: balls 1→N velocities
        self.y3 = cs.SX.sym("y3", dim, 1)  # state: ball  1+N position
        self.u = cs.SX.sym("u", dim, 1)  # input: ball  1+N velocity
        self.y = vc(self.y1, self.y2, self.y3)  # full state vector

        self.m = cs.SX.sym("m")  # mass
        self.D = cs.SX.sym("D")  # spring constant
        self.L = cs.SX.sym("L")  # spring length

        self.params = vc(self.m, self.D, self.L)

        self.g = np.array([0, 0, -9.81] if dim == 3 else [0, -9.81])  # gravity
        self.x0 = np.zeros((dim,))  # ball 0 position
        self.x_end = np.eye(1, dim, 0).ravel()  # ball N+1 reference position

    def dynamics(self, Ts=0.05):
        y, y1, y2, y3, u = self.y, self.y1, self.y2, self.y3, self.u
        dist = lambda xa, xb: cs.norm_2(xa - xb)
        N, d = self.N, self.dim
        p = self.params

        # Continuous-time dynamics y' = f(y, u; p)

        f1 = [y2]
        f2 = []
        for i in range(N):
            xi = y1[d * i:d * i + d]
            xip1 = y1[d * i + d:d * i + d * 2] if i < N - 1 else y3
            Fiip1 = self.D * (1 - self.L / dist(xip1, xi)) * (xip1 - xi)
            xim1 = y1[d * i - d:d * i] if i > 0 else self.x0
            Fim1i = self.D * (1 - self.L / dist(xi, xim1)) * (xi - xim1)
            fi = (Fiip1 - Fim1i) / self.m + self.g
            f2 += [fi]
        f3 = [u]

        f_expr = vc(*f1, *f2, *f3)
        self.f = cs.Function("f", [y, u, p], [f_expr], ["y", "u", "p"], ["y'"])

        # Discretize dynamics y[k+1] = f_d(y[k], u[k]; p)

        opt = {"tf": Ts, "simplify": True, "number_of_finite_elements": 4}
        intg = cs.integrator("intg", "rk", {
            "x": y,
            "p": vc(u, p),
            "ode": f_expr
        }, opt)

        f_d_expr = intg(x0=y, p=vc(u, p))["xf"]
        self.f_d = cs.Function("f_d", [y, u, p], [f_d_expr], ["y", "u", "p"],
                               ["y+"])

        return self.f_d

    def state_to_pos(self, y):
        N, d = self.N, self.dim
        rav = lambda x: np.array(x).ravel()
        xdim = lambda y, i: np.concatenate(
            ([0], rav(y[i:d * N:d]), rav(y[-d + i])))
        if d == 3:
            return (xdim(y, 0), xdim(y, 1), xdim(y, 2))
        else:
            return (xdim(y, 0), xdim(y, 1), np.zeros((N + 1,)))

    def input_to_matrix(self, u):
        """
        Reshape the input signal from a vector into a dim × N_horiz matrix (note
        that CasADi matrices are stored column-wise and NumPy arrays row-wise)
        """
        if isinstance(u, np.ndarray):
            return u.reshape((self.dim, u.shape[0] // self.dim), order='F')
        else:
            return u.reshape((self.dim, u.shape[0] // self.dim))

    def simulate(self, N_sim: int, y_0: np.ndarray, u: Union[np.ndarray, list,
    cs.SX.sym],
                 p: Union[np.ndarray, list, cs.SX.sym]):
        if isinstance(u, list):
            u = np.array(u)
        if isinstance(u, np.ndarray):
            if u.ndim == 1 or (u.ndim == 2 and u.shape[1] == 1):
                if u.shape[0] == self.dim:
                    u = np.tile(u, (N_sim, 1)).T
        return self.f_d.mapaccum(N_sim)(y_0, u, p)

    def initial_state(self):
        N, d = self.N, self.dim
        y1_0 = np.zeros((d * N))
        y1_0[0::d] = np.arange(1, N + 1) / (N + 1)
        y2_0 = np.zeros((d * N))
        y3_0 = np.zeros((d,))
        y3_0[0] = 1

        y_null = np.concatenate((y1_0, y2_0, y3_0))
        u_null = np.zeros((d,))

        return y_null, u_null

    def generate_cost_fun(self, α=25, β=1, γ=0.01):
        N, d = self.N, self.dim
        y1t = cs.SX.sym("y1t", d * N, 1)
        y2t = cs.SX.sym("y2t", d * N, 1)
        y3t = cs.SX.sym("y3t", d, 1)
        ut = cs.SX.sym("ut", d, 1)
        yt = cs.vertcat(y1t, y2t, y3t)

        L_cost = α * cs.sumsqr(y3t - self.x_end) + γ * cs.sumsqr(ut)
        for i in range(N):
            xdi = y2t[d * i:d * i + d]
            L_cost += β * cs.sumsqr(xdi)
        return cs.Function("L_cost", [yt, ut], [L_cost])


# %% Build the model

Ts = 0.05
N = 6  # number of balls
dim = 2  # dimension

model = HangingChain(N, dim)
f_d = model.dynamics(Ts)
y_null, u_null = model.initial_state()

param = [0.03, 1.6, 0.033 / N]  # Concrete parameters m, D, L

# %% Apply an initial input to disturb the system

N_dist = 3
u_dist = [-0.5, 0.5, 0.5] if dim == 3 else [-0.5, 0.5]
y_dist = model.simulate(N_dist, y_null, u_dist, param)
y_dist = np.hstack((np.array([y_null]).T, y_dist))

# %% Simulate the system without a controller

N_sim = 180
y_sim = model.simulate(N_sim, y_dist[:, -1], u_null, param)
y_sim = np.hstack((y_dist, y_sim))

# %% Define MPC cost and constraints

N_horiz = 12

L_cost = model.generate_cost_fun()  # stage cost
y_init = cs.SX.sym("y_init", *y_null.shape)  # initial state
U = cs.SX.sym("U", dim * N_horiz)  # control signals over horizon
constr_param = cs.SX.sym("c", 3)  # Coefficients of cubic constraint function
mpc_param = cs.vertcat(y_init, model.params, constr_param)  # all parameters
U_mat = model.input_to_matrix(U)  # Input as dim by N_horiz matrix

# Cost
mpc_sim = model.simulate(N_horiz, y_init, U_mat, model.params)
mpc_cost = 0
for n in range(N_horiz):  # Apply the stage cost function to each stage
    y_n = mpc_sim[:, n]
    u_n = U_mat[:, n]
    mpc_cost += L_cost(y_n, u_n)
mpc_cost_fun = cs.Function('f_mpc', [U, mpc_param], [mpc_cost])

# Constraints
g_constr = lambda c, x: c[0] * x ** 3 + c[1] * x ** 2 + c[2] * x  # Cubic constr
constr = []
for n in range(N_horiz):  # For each stage,
    y_n = mpc_sim[:, n]
    for i in range(N):  # for each ball in the stage,
        yx_n = y_n[dim * i]  # constrain the x, y position of the ball
        yy_n = y_n[dim * i + dim - 1]
        constr += [yy_n - g_constr(constr_param, yx_n)]
    constr += [y_n[-1] - g_constr(constr_param, y_n[-dim])]  # Ball N+1
mpc_constr_fun = cs.Function("g", [U, mpc_param], [cs.vertcat(*constr)])

# Fill in the constraint coefficients c(x-a)³ + d(x - a) + b
a, b, c, d = 0.6, -1.4, 5, 2.2
constr_coeff = [c, -3 * a * c, 3 * a * a * c + d]
constr_lb = b - c * a ** 3 - d * a

# %% NLP formulation
import alpaqa as pa

prob = generate_and_compile_casadi_problem(mpc_cost_fun, mpc_constr_fun)
prob.C.lowerbound = -1 * np.ones((dim * N_horiz,))
prob.C.upperbound = +1 * np.ones((dim * N_horiz,))
prob.D.lowerbound = constr_lb * np.ones((len(constr),))

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
        lbfgs_params={'memory': N_horiz},
    ),
)


# %% MPC controller


class MPCController:
    tot_it = 0
    failures = 0
    U = np.zeros((N_horiz * dim,))
    λ = np.zeros(((N + 1) * N_horiz,))

    def __init__(self, model, problem):
        self.model = model
        self.problem = problem

    def __call__(self, y_n):
        y_n = np.array(y_n).ravel()
        # Set the current state as the initial state
        self.problem.param[:y_n.shape[0]] = y_n
        # Solve the optimal control problem
        # (warm start using the previous solution and Lagrange multipliers)
        self.U, self.λ, stats = solver(self.problem, self.U, self.λ)
        # Print some solver statistics
        print(stats['status'], stats['outer_iterations'],
              stats['inner']['iterations'], stats['elapsed_time'],
              stats['inner_convergence_failures'])
        self.tot_it += stats['inner']['iterations']
        self.failures += stats['status'] != pa.SolverStatus.Converged
        # Print the Lagrange multipliers, shows that constraints are active
        print(np.linalg.norm(self.λ))
        # Return the optimal control signal for the first time step
        return self.model.input_to_matrix(self.U)[:, 0]


# %% Simulate the system using the MPC controller

y_n = np.array(y_dist[:, -1]).ravel()  # initial state for controller
n_state = y_n.shape[0]
prob.param = np.concatenate((y_n, param, constr_coeff))

y_mpc = np.empty((n_state, N_sim))
controller = MPCController(model, prob)
for n in range(N_sim):
    # Solve the optimal control problem
    u_n = controller(y_n)
    # Apply the first optimal control signal to the system and simulate for
    # one time step, then update the state
    y_n = model.simulate(1, y_n, u_n, param).T
    y_mpc[:, n] = y_n
y_mpc = np.hstack((y_dist, y_mpc))

print(controller.tot_it, controller.failures)

# %% Visualize

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import patheffects

mpl.rcParams['animation.frame_format'] = 'svg'

fig, ax = plt.subplots()
x, y, z = model.state_to_pos(y_null)
line, = ax.plot(x, y, '-o', label='Without MPC')
line_ctrl, = ax.plot(x, y, '-o', label='With MPC')
plt.legend()
plt.ylim([-2.5, 1])
plt.xlim([-0.25, 1.25])

x = np.linspace(-0.25, 1.25, 256)
y = np.linspace(-2.5, 1, 256)
X, Y = np.meshgrid(x, y)
Z = g_constr(constr_coeff, X) + constr_lb - Y

fx = [patheffects.withTickedStroke(spacing=7, linewidth=0.8)]
cgc = plt.contour(X, Y, Z, [0], colors='tab:green', linewidths=0.8)
plt.setp(cgc.collections, path_effects=fx)


class Animation:
    points = []

    def __call__(self, i):
        x, y, z = model.state_to_pos(y_sim[:, i])
        for p in self.points:
            p.remove()
        self.points = []
        line.set_xdata(x)
        line.set_ydata(y)
        viol = y - g_constr(constr_coeff, x) + 1e-5 < constr_lb
        if np.sum(viol):
            self.points += ax.plot(x[viol], y[viol], 'rx', markersize=12)
        x, y, z = model.state_to_pos(y_mpc[:, i])
        line_ctrl.set_xdata(x)
        line_ctrl.set_ydata(y)
        viol = y - g_constr(constr_coeff, x) + 1e-5 < constr_lb
        if np.sum(viol):
            self.points += ax.plot(x[viol], y[viol], 'rx', markersize=12)
        return [line, line_ctrl] + self.points


ani = mpl.animation.FuncAnimation(fig,
                                  Animation(),
                                  interval=1000 * Ts,
                                  blit=True,
                                  repeat=True,
                                  frames=1 + N_dist + N_sim)

# Export the animation
out = join(dirname(__file__), 'hanging-chain.html')
os.makedirs(dirname(out), exist_ok=True)
with open(out, "w") as f:
    f.write('<center>')
    f.write(ani.to_jshtml())
    f.write('</center>')

# Show the animation
plt.show()
