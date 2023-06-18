import math

import numpy as np
import bezier_curves
import matplotlib.pyplot as plt


def binomial_coefficient(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k))


def bezier_curve(j, P):
    x, y = 0, 0
    for i in range(0, 6):
        c = binomial_coefficient(5, i)
        x += c * (1 - j) ** (5 - i) * j ** i * P[0, i]
        y += c * (1 - j) ** (5 - i) * j ** i * P[1, i]
    return x, y


class Car:

    def __init__(
            self,
            name="Car0",
            x=0,
            v=10,
            lane=1,
            L=4.2,
            W=1.8,
            l=3,
            θ_max=3.2 / 180 * np.pi,
            tlc=5.17,
            td=1.2,
            ti=0.15,
            τ=0.9,
            a_max=7,
            h=3.75,
            Lf=1,

    ) -> None:
        self.name = name
        self.x = x
        self.lane = lane
        self.v = v
        self.L = L
        self.W = W
        self.l = l
        self.θ_max = θ_max
        self.tlc = tlc
        self.td = td
        self.ti = ti
        self.τ = τ
        self.a_max = a_max
        self.h = h
        self.Lf = Lf

    def move(self, dt):
        self.x += self.v * dt

    def get_car_in_front(self, cars, target_lane):
        car_in_front = None

        for car in cars:
            if car.lane != target_lane:
                continue

            if car.x > self.x:
                if car_in_front is None:
                    car_in_front = car

                if car_in_front.x > car.x:
                    car_in_front = car

        return car_in_front

    def get_bezier_control_points(self, car, i=5):
        Px0 = Py0 = Py1 = Py2 = 0
        Py3 = Py4 = Py5 = self.h

        Li = self.Lf + self.l
        Di = Li * np.cos(np.arctan2(self.W, 2 * self.Lf) - self.θ_max)
        D1 = car.x - self.x
        tc1 = D1 / (self.v - car.v)
        Px2 = Px3 = self.v * tc1 - Di
        Px5 = 2 * Px2
        Px1 = (Px2 - Px0) / i
        Px4 = Px5 - (Px5 - Px3) / i

        Px = np.array([
            Px0, Px1, Px2, Px3, Px4, Px5
        ])
        Py = np.array([
            Py0, Py1, Py2, Py3, Py4, Py5
        ])

        tca = Px2 / (self.v - car.v)
        return np.array([Px, Py]), tca

    def get_safety_distance(self, car, target_lane, q1=0.65, q2=0.35):
        if self.lane == car.lane:
            if self.x > car.x:
                return np.abs(self.x - car.x)

            if target_lane == self.lane:
                return (
                        q1 * self.v + self.td +
                        q2 * ((self.v - car.v) * self.τ + self.ti / 2 +
                              (self.v - car.v) ** 2 / (2 * car.a_max)) + self.l
                )
            # S01
            if self.v > car.v:
                return self.v - car.v * self.tlc / 2 + self.L + self.W / 2 * np.sin(self.θ_max)

            return q1 * self.v * self.td + self.l

        # S02
        if self.x < car.x:
            if self.v > car.v:
                return (
                        self.v - car.v * self.tlc / 2 + self.L -
                        self.W / 2 * np.sin(self.θ_max) +
                        q1 * self.v * self.td +
                        q2 * ((self.v - car.v) * self.τ + self.ti / 2 +
                              (self.v - car.v) ** 2 / (2 * car.a_max))
                )

            return q1 * self.v * self.td + self.l

        # S03
        if self.v < car.v:
            return (
                    (car.v - self.v) * 3 / 4 * self.tlc + self.L +
                    q1 * car.v * self.td +
                    q2 * ((car.v - self.v) * self.τ + self.ti / 2 + (car.v - self.v) ** 2 / (2 * self.a_max))
            )

        return q1 * car.v * self.td + self.l

    def get_safety_payoff(self, cars, target_lane):
        payoff = 1
        temp_payoff = 1
        for car in cars:
            if self.lane != car.lane and self.lane == target_lane:
                continue

            Sk = np.abs(self.get_safety_distance(car, target_lane))
            Dk = np.abs(self.x - car.x)

            if Dk >= np.abs(Sk):
                temp_payoff = 1

            if Dk <= self.l:
                temp_payoff = -1

            if self.l < Dk < np.abs(Sk):
                temp_payoff = np.log(Dk / Sk + 1) / np.log(2)

            if temp_payoff < payoff:
                payoff = temp_payoff

        return payoff

    def get_velocity_payoff(self, cars, target_lane):
        car_in_front = self.get_car_in_front(cars, target_lane)
        if car_in_front is None:
            return 1

        if car_in_front.v == 0:
            return -1

        if car_in_front.v >= 2 * self.v:
            return 1

        return (car_in_front.v - self.v) / self.v

    def get_comfort_payoff(self, cars, target_lane):
        if target_lane == 1:
            return 0
        car_in_front = self.get_car_in_front(cars, 1)
        if car_in_front is None:
            return 0

        if self.v > car_in_front.v:
            _, tca = self.get_bezier_control_points(car_in_front)
            return 2 / (1 + np.exp(-tca)) - 2

        return 0

    def get_total_payoff(self, cars, target_lane, a=0.7, b=0.3, c=0.2):
        safety = self.get_safety_payoff(cars, target_lane)
        velocity = self.get_velocity_payoff(cars, target_lane)
        comfort = self.get_comfort_payoff(cars, target_lane)
        total = (
                a * safety +
                b * velocity
                # c * comfort
        )
        print(
            "Lane: ", target_lane,
            "\tSafety: ", "{:.2f}".format(safety),
            "\t Velocity: ", "{:.2f}".format(velocity),
            "\tComfort: ", "{:.2f}".format(comfort),
            " \tTotal: ", "{:.2f}".format(total)
        )
        return total


def get_cars_test_1():
    ego = Car(
        x=0,
        v=10,
        lane=1
    )
    cars = np.array([
        Car(
            "Car1",
            x=50,
            v=0,
            lane=1
        ),
        Car(
            "Car2",
            x=10,
            v=15,
            lane=2
        ),
        Car(
            "Car3",
            x=-20,
            v=15,
            lane=2
        ),
        Car(
            "Car4",
            x=-30,
            v=15,
            lane=2
        )
    ])
    return ego, cars


def get_cars_test_2():
    ego = Car(
        x=0,
        v=10,
        lane=1
    )
    cars = np.array([
        Car(
            "Car1",
            x=50,
            v=0,
            lane=1
        ),
        Car(
            "Car2",
            x=10,
            v=15,
            lane=2
        ),
        Car(
            "Car3",
            x=-8,
            v=15,
            lane=2
        ),
        Car(
            "Car4",
            x=-25,
            v=15,
            lane=2
        )
    ])
    return ego, cars


def get_cars_test_3():
    ego = Car(
        x=0,
        v=10,
        lane=1
    )
    cars = np.array([
        Car(
            "Car1",
            x=50,
            v=0,
            lane=1
        ),
        Car(
            "Car2",
            x=10,
            v=15,
            lane=2
        ),
        Car(
            "Car3",
            x=-8,
            v=15,
            lane=2
        ),
        Car(
            "Car4",
            x=-18,
            v=15,
            lane=2
        )
    ])
    return ego, cars


if __name__ == '__main__':
    ego, cars = get_cars_test_1()

    plt.figure()

    dt = 0.1
    t = np.arange(0, 5, dt)
    payoff = np.zeros([t.size, 2])

    print("Ego: ", ego.x)
    for car in cars:
        print(car.name, ": ", car.x)

    for i in range(t.size):
        payoff[i, :] = [
            ego.get_total_payoff(cars, target_lane=1),
            ego.get_total_payoff(cars, target_lane=2)
        ]
        ego.move(dt)
        print("Ego: ", ego.x)
        for car in cars:
            car.move(dt)
            print(car.name, ": ", car.x)
        # print(payoff)

    plt.plot(t, payoff)
    plt.grid()
    plt.legend(["Lane 1", "Lane 2"])
    plt.show()
