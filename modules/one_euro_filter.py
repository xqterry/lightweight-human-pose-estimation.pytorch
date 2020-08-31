import math


def get_alpha(rate=30, cutoff=1):
    tau = 1 / (2 * math.pi * cutoff)
    te = 1 / rate
    return 1 / (1 + tau / te)


class LowPassFilter:
    def __init__(self):
        self.x_previous = None

    def __call__(self, x, alpha=0.5):
        if self.x_previous is None:
            self.x_previous = x
            return x
        x_filtered = alpha * x + (1 - alpha) * self.x_previous
        self.x_previous = x_filtered
        return x_filtered


class OneEuroFilter:
    def __init__(self, freq=15, mincutoff=1, beta=0.05, dcutoff=1):
        self.freq = freq
        self.mincutoff = mincutoff
        self.beta = beta
        self.dcutoff = dcutoff
        self.filter_x = LowPassFilter()
        self.filter_dx = LowPassFilter()
        self.x_previous = None
        self.dx = None

    def __call__(self, x):
        if self.dx is None:
            self.dx = 0
        else:
            self.dx = (x - self.x_previous) * self.freq
        dx_smoothed = self.filter_dx(self.dx, get_alpha(self.freq, self.dcutoff))
        cutoff = self.mincutoff + self.beta * abs(dx_smoothed)
        x_filtered = self.filter_x(x, get_alpha(self.freq, cutoff))
        self.x_previous = x
        return x_filtered

class KalmanFilter:
    def __init__(self, q=0.001, r=0.0015):
        self.param_Q = q
        self.param_R = r

        self.K = [0.0, 0.0]
        self.P = [0.0, 0.0]
        self.X = [0.0, 0.0]

        self.pos = None

    def __call__(self, now_xy):
        self.update()
        x = now_xy[0]
        y = now_xy[1]
        if self.pos is not None:
            self.pos[0] = self.X[0] + (x - self.X[0]) * self.K[0]
            self.pos[1] = self.X[1] + (y - self.X[1]) * self.K[1]
        else:
            self.pos = [float(x), float(y)]

        self.X = self.pos
        # print("IN", x, y, " OUT", self.X)
        return self.pos

    def update(self):
        self.K[0] = (self.P[0] + self.param_Q) / (self.P[0] + self.param_R + self.param_Q)
        self.K[1] = (self.P[1] + self.param_Q) / (self.P[1] + self.param_R + self.param_Q)
        self.P[0] = self.param_R * (self.P[0] + self.param_Q) / (self.param_R + self.P[0] + self.param_Q)
        self.P[1] = self.param_R * (self.P[1] + self.param_Q) / (self.param_R + self.P[1] + self.param_Q)

if __name__ == '__main__':
    filter = OneEuroFilter(freq=15, beta=0.1)
    for val in range(10):
        x = val + (-1)**(val % 2)
        x_filtered = filter(x)
        print(x_filtered, x)
