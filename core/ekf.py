import numpy as np


class ExtendedKalmanFilter:
    """
    Extended Kalman Filter (EKF) for Non-Linear Parameter Estimation.
    Uses Complex Step Differentiation to compute Jacobians automatically.
    """

    def __init__(self, f_dynamics, h_measurement, Q, R, x0, p_init_scale=0.1):
        """
        f_dynamics: Function f(x, u) -> x_dot (supports complex inputs)
        h_measurement: Function h(x) -> y (supports complex inputs)
        Q: Process Noise Covariance
        R: Measurement Noise Covariance
        x0: Initial State Guess
        p_init_scale: Scaling factor for initial P matrix (default 0.1)
        """
        self.f = f_dynamics
        self.h = h_measurement
        self.Q = np.array(Q, dtype=float)
        self.R = np.array(R, dtype=float)

        self.x_hat = np.array(x0, dtype=float).reshape(-1, 1)
        # Use the configured scaling factor for initial covariance
        self.P = np.eye(len(x0)) * p_init_scale
        self.n = len(x0)

    def compute_jacobian(self, func, x, u=None, epsilon=1e-20):
        """
        Computes Jacobian matrix using Complex Step Differentiation.
        J_ij = d(func_i)/d(x_j)
        """
        n_out = len(func(x, u)) if u is not None else len(func(x))
        n_in = len(x)
        J = np.zeros((n_out, n_in))

        # We assume x is real, but we need to pass complex perturbations
        x_complex = np.array(x, dtype=complex)

        for j in range(n_in):
            # Perturb j-th state by imaginary epsilon
            x_perturb = x_complex.copy()
            x_perturb[j] += 1j * epsilon

            # Evaluate function
            if u is not None:
                f_val = func(x_perturb, u)
            else:
                f_val = func(x_perturb)

            # The imaginary part / epsilon is the exact derivative
            J[:, j] = f_val.imag.flatten() / epsilon

        return J

    def predict(self, u, dt):
        """
        Time Update: x = x + f(x,u)*dt
        """
        # 1. Integrate State (Euler Method for simplicity in estimation)
        # x_k|k-1 = x_k-1 + f(x_k-1, u)*dt
        # Note: We keep x as real for storage
        x_dot = self.f(self.x_hat.astype(complex), u).real
        self.x_pred = self.x_hat + x_dot * dt

        # 2. Compute Jacobian of the Discrete Transition Matrix F
        # F = I + A_c * dt, where A_c = df/dx
        A_c = self.compute_jacobian(self.f, self.x_hat, u)
        F = np.eye(self.n) + A_c * dt

        # 3. Predict Covariance
        # P = F P F^T + Q
        self.P = F @ self.P @ F.T + self.Q

    def update(self, y_meas):
        """
        Measurement Update
        """
        # 1. Compute Jacobian H = dh/dx
        # We pass a lambda that accepts ONLY x, because u=None
        H = self.compute_jacobian(lambda x: self.h(x), self.x_hat, u=None)

        # 2. Innovation
        y_pred = self.h(self.x_hat).real
        y_err = y_meas - y_pred

        # 3. Kalman Gain
        S = H @ self.P @ H.T + self.R
        K = self.P @ H.T @ np.linalg.inv(S)

        # 4. Update State and Covariance
        self.x_hat = self.x_pred + K @ y_err
        I = np.eye(self.n)
        self.P = (I - K @ H) @ self.P

        return self.x_hat.flatten()
