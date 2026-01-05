import numpy as np

class KalmanFilter:
    def __init__(self, initial_state:np.ndarray):
        # State vector [u, v, y, h, u_dot, v_dot, y_dot, h_dot]
        self.state = initial_state

        # State covariance matrix
        self.P = np.diag([0.5, 0.5, 0.5, 0.5, 10000.0, 10000.0, 10000.0, 10000.0])

        # State transition matrix
        self.F = np.eye(8)
        self.F[0, 4] = 1.0
        self.F[1, 5] = 1.0
        self.F[2, 6] = 1.0
        self.F[3, 7] = 1.0

        # Measurement matrix
        self.H = np.zeros((4, 8))
        self.H[0, 0] = 1.0
        self.H[1, 1] = 1.0
        self.H[2, 2] = 1.0
        self.H[3, 3] = 1.0

        # Measurement noise covariance
        self.R = np.eye(4) * 0.01

        # Process noise covariance
        self.Q = np.eye(8) * 0.01

    def predict(self):
        # Predict the next state
        self.state = np.dot(self.F, self.state)
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q
        return self.state

    def update(self, measurement):
        # Update the state with a new measurement
        y = measurement.reshape(-1, 1) - np.dot(self.H, self.state) # residual / innovation
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R # residual covariance / innovation covariance
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S)) # Kalman gain
        self.state += np.dot(K, y)
        I = np.eye(self.P.shape[0])
        self.P = (I - np.dot(K, self.H)).dot(self.P)
    
    # Compute malahanobis distance between measurement and predicted state
    def mahalanobis_distance(self, measurement):
        y = measurement.reshape(-1, 1) - np.dot(self.H, self.state)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R
        dist = np.dot(np.dot(y.T, np.linalg.inv(S)), y)
        # print(dist)
        return float(dist[0,0])
    