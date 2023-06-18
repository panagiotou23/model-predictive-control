import numpy as np
import bezier_curves
import matplotlib.pyplot as plt


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

    def get_safety_distance(self, car, target_lane, q1=0.65, q2=0.35):
        if target_lane == self.lane:
            return np.abs(self.x - car.x)

        if self.lane == car.lane:
            if self.x > car.x:
                return np.abs(self.x - car.x)

            # S01
            if self.v > car.v:
                S01_1 = self.v - car.v * self.tlc / 2 + self.L + self.W / 2 * np.sin(self.θ_max)
                S01_2 = (
                        q1 * self.v + self.td +
                        q2 * ((self.v - car.v) * self.τ + self.ti / 2 +
                              (self.v - car.v) ** 2 / (2 * car.a_max)) + self.l
                )
                return max(S01_1, S01_2)

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
                    car.v - self.v * 3 / 4 * self.tlc + self.L +
                    q1 * car.v * self.td +
                    q2 * ((car.v - self.v) * self.τ + self.ti / 2 + (car.v - self.v) ** 2 / (2 * self.a_max))
            )

        return q1 * car.v * self.td + self.l

    def get_safety_payoff(self, cars, target_lane):
        payoff = 1
        temp_payoff = 1
        for car in cars:
            Sk = self.get_safety_distance(car, target_lane)
            Dk = np.abs(self.x - car.x)

            if Dk >= Sk:
                temp_payoff = 1

            if Dk <= self.l:
                temp_payoff = -1

            if self.l < Dk < Sk:
                temp_payoff = np.log(Dk / Sk + 1) / np.log(2)

            # print(car.name, " payoff: ", temp_payoff)
            if temp_payoff < payoff:
                payoff = temp_payoff

        return payoff


if __name__ == '__main__':
    # Test 1
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

    plt.figure()
    t = np.arange(6)
    payoff = np.zeros(6)
    payoff[0] = np.array([ego.get_safety_payoff(cars, target_lane=2)])
    dt = 1
    for i in range(0, 6):
        ego.move(dt)
        print("Ego: ", ego.x)
        for car in cars:
            car.move(dt)
            print(car.name, ": ", car.x)

        payoff[i] = ego.get_safety_payoff(cars, target_lane=2)
        print(payoff)

    plt.plot(t, payoff)
    plt.show()
