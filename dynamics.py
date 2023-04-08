import numpy as np


class VehicleParameters:
    length: float = 9.7e-2  # length of the car (meters)
    axis_front: float = 4.7e-2  # distance cog and front axis (meters)
    axis_rear: float = 5e-2  # distance cog and rear axis (meters)
    front: float = 0.09  # distance cog and front (meters)
    rear: float = 0.07  # distance cog and rear (meters)
    width: float = 8e-2  # width of the car (meters)
    height: float = 5.5e-2  # height of the car (meters)
    mass: float = 0.1735  # mass of the car (kg)
    inertia: float = 18.3e-5  # moment of inertia around vertical (kg*m^2)

    # input limits
    max_steer: float = 0.32  # max steering angle (radians)
    max_drive: float = 1.0  # maximum drive

    """Pacejka 'Magic Formula' parameters.
    Used for magic formula: `peak * sin(shape * arctan(stiffness * alpha))`
    as in Pacejka (2005) 'Tyre and Vehicle Dynamics', p. 161, Eq. (4.6)
    """
    # front
    bf: float = 0.268   # front stiffness factor
    cf: float = 2.165  # front shape factor
    df: float = 3.47  # front peak factor

    # rear
    br: float = 0.242  # rear stiffness factor
    cr: float = 2.38  # rear shape factor
    dr: float = 2.84  # rear peak factor

    # kinematic approximation
    friction: float = 1  # friction parameter
    acceleration: float = 2  # maximum acceleration

    # motor parameters
    cm1: float = 0.266
    cm2: float = 0.1
    cr0: float = 0.1025
    cr1: float = 0.1629
    cr2: float = 0.0011


class KinematicBicyclePacejka:
    def __init__(
            self,
            params: VehicleParameters = None,
    ) -> None:
        """Initialize kinematic bicycle model.

        Args:
            params (VehicleParameters, optional): vehicle parameters. Defaults to VehicleParameters().
        """
        self.params = VehicleParameters() if params is None else params

    def clip_inputs(self, d=None, δ=None):
        res = []
        if d is not None:
            res.append(np.clip(d, -self.params.max_drive, self.params.max_drive))
        if δ is not None:
            res.append(np.clip(δ, -self.params.max_steer, self.params.max_steer))
        if len(res) == 1:
            return res[0]
        return tuple(res)

    def __call__(self, x: np.ndarray, u: np.ndarray, t: int = None) -> np.ndarray:
        """Evaluate kinematic bicycle model.

        Args:
            x (np.ndarray): state [x, y, ɸ, vx, vy, ω].
            u (np.ndarray): input [d, δ].

        Returns:
            np.ndarray: derivative of the state.
        """

        # unpack arguments
        d, δ = u[0], u[1]
        φ, vx, vy, ω = x[2], x[3], x[4], x[5]

        # unpack parameters
        lf, lr = self.params.axis_front, self.params.axis_rear
        m = self.params.mass
        iz = self.params.inertia

        cm1 = self.params.cm1
        cm2 = self.params.cm2
        cr0 = self.params.cr0 * np.sign(vx)
        cr2 = self.params.cr2

        bf = self.params.bf
        cf = self.params.cf
        df = self.params.df
        br = self.params.br
        cr = self.params.cr
        dr = self.params.dr

        d, δ = self.clip_inputs(d, δ)

        # evaluate

        af = - np.arctan2(ω * lf + vy, vx) + δ
        ar = np.arctan2(ω * lr - vy, vx)

        frx = (cm1 - cm2 * vx) * d - cr0 - cr2 * vx * vx
        ffy = df * np.sin(cf * np.arctan(bf * af))
        fry = dr * np.sin(cr * np.arctan(br * ar))

        res = np.array([
            vx * np.cos(φ) - vy * np.sin(φ),
            vx * np.sin(φ) + vy * np.cos(φ),
            ω,
            (frx - ffy * np.sin(δ) + m * vy * ω) / m,
            (fry + ffy * np.cos(δ) - m * vx * ω) / m,
            (ffy * lf * np.cos(δ) - fry * lr) / iz
        ])

        return res


class KinematicBicycleSimplified:
    def __init__(
            self,
            params: VehicleParameters = None,
    ) -> None:
        """Initialize kinematic bicycle model.

        Args:
            params (VehicleParameters, optional): vehicle parameters. Defaults to VehicleParameters().
        """
        self.params = VehicleParameters() if params is None else params

    def clip_inputs(self, d=None, δ=None):
        res = []
        if d is not None:
            res.append(np.clip(d, -self.params.max_drive, self.params.max_drive))
        if δ is not None:
            res.append(np.clip(δ, -self.params.max_steer, self.params.max_steer))
        if len(res) == 1:
            return res[0]
        return tuple(res)

    def __call__(self, x: np.ndarray, u: np.ndarray, t: int = None) -> np.ndarray:
        """Evaluate kinematic bicycle model.

        Args:
            x (np.ndarray): state [x, y, ɸ, v].
            u (np.ndarray): input [d, δ].

        Returns:
            np.ndarray: derivative of the state.
        """

        # unpack arguments
        d, δ = u[0], u[1]
        φ, v = x[2], x[3]

        # unpack parameters
        lf, lr = self.params.axis_front, self.params.axis_rear
        a, μ = self.params.acceleration, self.params.friction

        d, δ = self.clip_inputs(d, δ)

        # evaluate
        β = np.arctan2(lf * np.tan(δ), lf + lr)
        res = np.array([
            v * np.cos(φ + β),  # x dot
            v * np.sin(φ + β),  # y dot
            v * np.sin(β) / lr,  # φ dot
            a * d - μ * v  # v dot
        ])
        return res
