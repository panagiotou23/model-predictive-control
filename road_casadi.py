import casadi as cs
import numpy as np


class Road:

    def __init__(self, size):
        self.size = size
        self.centerline = cs.SX.sym("centerline", size, 2)
