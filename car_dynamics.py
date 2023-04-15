from ctypes import Union

import casadi as cs
import numpy as np
from casadi import vertcat as vc

from road_casadi import Road


class KinematicBicyclePacejka:
    def __init__(self) -> None:
        """Initialize kinematic bicycle model.
        """

        self.x = cs.SX.sym("x")
        self.y = cs.SX.sym("y")
        self.φ = cs.SX.sym("φ")
        self.vx = cs.SX.sym("vx")
        self.vy = cs.SX.sym("vy")
        self.ω = cs.SX.sym("ω")

        self.d = cs.SX.sym("d")
        self.δ = cs.SX.sym("δ")

        self.X = vc(
            self.x,
            self.y,
            self.φ,
            self.vx,
            self.vy,
            self.ω
        )

        self.u = vc(
            self.d,
            self.δ
        )

        self.length = cs.SX.sym("length")
        self.axis_front = cs.SX.sym("axis_front")
        self.axis_rear = cs.SX.sym("axis_rear")
        self.front = cs.SX.sym("front")
        self.rear = cs.SX.sym("rear")
        self.width = cs.SX.sym("width")
        self.height = cs.SX.sym("height")
        self.mass = cs.SX.sym("mass")
        self.inertia = cs.SX.sym("inertia")

        self.max_steer = cs.SX.sym("max_steer")
        self.max_drive = cs.SX.sym("max_drive")

        self.bf = cs.SX.sym("bf")
        self.cf = cs.SX.sym("cf")
        self.df = cs.SX.sym("df")

        self.br = cs.SX.sym("br")
        self.cr = cs.SX.sym("cr")
        self.dr = cs.SX.sym("dr")

        self.cm1 = cs.SX.sym("cm1")
        self.cm2 = cs.SX.sym("cm2")
        self.cr0 = cs.SX.sym("cr0")
        self.cr1 = cs.SX.sym("cr1")
        self.cr2 = cs.SX.sym("cr2")

        self.params = vc(
            self.length,
            self.axis_front,
            self.axis_rear,
            self.front,
            self.rear,
            self.width,
            self.height,
            self.mass,
            self.inertia,
            self.max_steer,
            self.max_drive,
            self.bf,
            self.cf,
            self.df,
            self.br,
            self.cr,
            self.dr,
            self.cm1,
            self.cm2,
            self.cr0,
            self.cr1,
            self.cr2
        )

        self.X_0 = np.array([
            0,  # x
            0,  # y
            0,  # φ
            0.3,  # vx
            0,  # vy
            0  # ω
        ])
        self.u_0 = np.array([0, 0])

        self.f = None
        self.f_d = None

    def dynamics(self, Ts=0.05):
        d, δ = self.u[0], self.u[1]
        φ, vx, vy, ω = self.φ, self.vx, self.vy, self.ω

        lf, lr = self.axis_front, self.axis_rear
        m = self.mass
        iz = self.inertia

        cm1 = self.cm1
        cm2 = self.cm2
        cr0 = self.cr0 * cs.sign(vx)
        cr2 = self.cr2

        bf = self.bf
        cf = self.cf
        df = self.df
        br = self.br
        cr = self.cr
        dr = self.dr

        # Continuous-time dynamics y' = f(y, u; p)

        af = - cs.arctan2(ω * lf + vy, vx) + δ
        ar = cs.arctan2(ω * lr - vy, vx)

        frx = (cm1 - cm2 * vx) * d - cr0 - cr2 * vx * vx
        ffy = df * cs.sin(cf * cs.arctan(bf * af))
        fry = dr * cs.sin(cr * cs.arctan(br * ar))

        f_expr = vc(
            vx * cs.cos(φ) - vy * cs.sin(φ),
            vx * cs.sin(φ) + vy * cs.cos(φ),
            ω,
            (frx - ffy * cs.sin(δ) + m * vy * ω) / m,
            (fry + ffy * cs.cos(δ) - m * vx * ω) / m,
            (ffy * lf * cs.cos(δ) - fry * lr) / iz
        )

        y, u, p = self.X, self.u, self.params
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

    def input_to_matrix(self, u):
        """
        Reshape the input signal from a vector into a dim × N_horiz matrix (note
        that CasADi matrices are stored column-wise and NumPy arrays row-wise)
        """
        if isinstance(u, np.ndarray):
            return u.reshape((self.u.shape[0], u.shape[0] // self.u.shape[0]), order='F')
        else:
            return u.reshape((self.u.shape[0], u.shape[0] // self.u.shape[0]))

    def simulate(
            self,
            N_sim: int,
            y_0: np.ndarray,
            u,
            p
    ):
        return self.f_d.mapaccum(N_sim)(y_0, u, p)

    def wrap_to_pi(
            self,
            angle: cs.SX
    ):
        return np.mod(angle + np.pi, 2 * np.pi) - np.pi

    def find_nearest_point(
            self,
            size: int,
            vehicle_position,
            centerline
    ):

        min_dist = cs.inf
        idx = -1
        for i in range(size):
            dist = (centerline[i, 0] - vehicle_position[0]) ** 2 + \
                   (centerline[i, 1] - vehicle_position[1]) ** 2
            min_dist = cs.fmin(dist, min_dist)
            if cs.is_equal(dist, min_dist):
                idx = i
        return idx, centerline[idx, :]


    def compute_errors(
            self,
            size: int,
            vehicle_position,
            vehicle_heading,
            centerline_flat
    ):
        if isinstance(centerline_flat, np.ndarray):
            centerline = centerline_flat.reshape((centerline_flat.shape[0] // 2, 2), order='C')
        else:
            centerline = centerline_flat.reshape((centerline_flat.shape[0] // 2, 2))

        # find the nearest point on centerline
        idx, nearest_point = self.find_nearest_point(size, vehicle_position, centerline)
        nearest_point = nearest_point.T
        # calculate cross-track error
        v_vec = vehicle_position - centerline[idx - 1]
        w_vec = (nearest_point - centerline[idx - 1])
        cte = (v_vec[0] * w_vec[1] - v_vec[1] * w_vec[0]) / cs.sqrt(
            w_vec[0] ** 2 + w_vec[1 ** 2])

        # calculate heading error
        if idx < size - 1:
            if cs.is_equal(centerline[idx + 1][0], centerline[idx][0]):
                desired_heading = 0
            else:
                desired_heading = cs.arctan2(centerline[idx + 1, 1] - centerline[idx, 1],
                                             centerline[idx + 1, 0] - centerline[idx, 0])
        else:
            desired_heading = 0
        heading_error = self.wrap_to_pi(desired_heading - vehicle_heading)

        # calculate positional error
        next_point = centerline[min(idx + 1, size - 1)]
        v_vec_next = vehicle_position - nearest_point
        w_vec_next = next_point - nearest_point
        pos_error = (v_vec_next[0] * w_vec_next[1] - v_vec_next[1] * w_vec_next[0]) / cs.sqrt(
            w_vec_next[0] ** 2 + w_vec_next[1 ** 2])

        return cte, heading_error, pos_error

    def generate_cost_fun(self, centerline_size, centerline, c=np.array([50, 25, 50, 25, 0.01])):
        x = cs.SX.sym("x")
        y = cs.SX.sym("y")
        φ = cs.SX.sym("φ")
        vx = cs.SX.sym("vx")
        vy = cs.SX.sym("vy")
        ω = cs.SX.sym("ω")
        X = vc(x, y, φ, vx, vy, ω)

        d = cs.SX.sym("d")
        δ = cs.SX.sym("d")
        u = vc(d, δ)

        target_v = cs.SX.sym("target_v")

        pos = vc(x, y)
        cte, heading_error, pos_error = self.compute_errors(
            centerline_size,
            pos,
            φ,
            centerline
        )

        L_cost = c[0] * (cs.sqrt(vx ** 2 + vy ** 2) - target_v) ** 2 + \
                 c[1] * y ** 2 + \
                 c[2] * heading_error ** 2 + \
                 c[3] * ω ** 2
        return cs.Function("L_cost", [X, u, target_v], [L_cost])
