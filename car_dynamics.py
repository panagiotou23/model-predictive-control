from ctypes import Union

import casadi as cs
import numpy as np
from casadi import vertcat as vc

from road import Road


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

        self.state = vc(
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

        self.state_0 = np.array([
            0,  # x
            0,  # y
            0,  # φ
            1,  # vx
            0,  # vy
            0  # ω
        ])
        self.u_0 = np.array([0, 0])

        self.road = Road()

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

        y, u, p = self.state, self.u, self.params
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

    def simulate(
            self,
            N_sim: int,
            y_0: np.ndarray,
            u,
            p
    ):
        return self.f_d.mapaccum(N_sim)(y_0, u, p)

    def generate_cost_fun(self, α=25, β=1, γ=0.01):
        xt = cs.SX.sym("yt")
        yt = cs.SX.sym("yt")
        φt = cs.SX.sym("yt")
        ut = cs.SX.sym("ut")

        _, heading_error, pos_error = self.road.compute_errors(
            np.array([xt, yt]), φt
        )

        L_cost = α * pos_error ** 2 + β * heading_error ** 2 + γ * ut ** 2
        return cs.Function("L_cost", [xt, yt, φt, ut], [L_cost])