import numpy as np
import matplotlib.pyplot as plt


class Car:
    def __init__(self, x, y, u_x, u_y, a, lane):
        self.x = x
        self.y = y
        self.u_x = u_x
        self.u_y = u_y
        self.a = a
        self.lane = lane

    def choose_lane(self):
        pass


def get_states(
        t0: float,
        t1: float,
        ego: Car,
        fe: Car,
        lane
):

    SE_t0_t1 = ego.x + ego.u_x * (t0 + t1) + 1 / 2 * ego.a * (t0 + t1) ** 2

    return SE_t0_t1


t0 = 1
t1 = 2
ego = Car(0, 0, 10, 0, 0, 0)
fe = Car(0, 0, 20, 0, 0, 0)
test = get_states(t0, t1, ego, fe)

print(test)
