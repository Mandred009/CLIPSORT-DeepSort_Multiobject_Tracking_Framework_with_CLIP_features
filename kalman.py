import numpy as np


class KalmanFilter:
    """Constant-velocity box filter from DeepSORT (Wojke et al.).

    State is [u, v, aspect, height, and their velocities]. Process and
    measurement noise scale with box height so the Mahalanobis gate stays
    on a chi-square scale (9.4877 for 4 DoF at 95%).
    """

    std_weight_position = 1.0 / 20.0
    std_weight_velocity = 1.0 / 160.0

    def __init__(self, initial_state: np.ndarray):
        self.state = np.asarray(initial_state, dtype=np.float64).reshape(-1, 1)
        self.state[2, 0] = max(float(self.state[2, 0]), 1e-2)
        self.state[3, 0] = max(float(self.state[3, 0]), 1.0)

        h = float(self.state[3, 0])
        std = np.array([
            2 * self.std_weight_position * h,
            2 * self.std_weight_position * h,
            1e-2,
            2 * self.std_weight_position * h,
            10 * self.std_weight_velocity * h,
            10 * self.std_weight_velocity * h,
            1e-5,
            10 * self.std_weight_velocity * h,
        ])
        self.P = np.diag(np.square(std))

        self.F = np.eye(8)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.F[3, 7] = 1.0

        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

    def _height(self):
        return max(float(self.state[3, 0]), 1.0)

    def _motion_cov(self):
        h = self._height()
        std_pos = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-2,
            self.std_weight_position * h,
        ]
        std_vel = [
            self.std_weight_velocity * h,
            self.std_weight_velocity * h,
            1e-5,
            self.std_weight_velocity * h,
        ]
        return np.diag(np.square(np.r_[std_pos, std_vel]))

    def _measurement_cov(self):
        h = self._height()
        std = [
            self.std_weight_position * h,
            self.std_weight_position * h,
            1e-1,
            self.std_weight_position * h,
        ]
        return np.diag(np.square(std))

    def _clamp_state(self):
        self.state[2, 0] = max(float(self.state[2, 0]), 1e-2)
        self.state[3, 0] = max(float(self.state[3, 0]), 1.0)

    def predict(self):
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self._motion_cov()
        self._clamp_state()
        return self.state

    def update(self, measurement):
        residual = measurement.reshape(-1, 1) - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self._measurement_cov()
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))
        self.state = self.state + np.dot(K, residual)
        self.P = np.dot(np.eye(8) - np.dot(K, self.H), self.P)
        self._clamp_state()

    def mahalanobis_distance(self, measurement):
        residual = measurement.reshape(-1, 1) - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self._measurement_cov()
        dist = residual.T.dot(np.linalg.solve(S, residual))
        return float(np.asarray(dist).reshape(-1)[0])
