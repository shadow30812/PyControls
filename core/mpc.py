import numpy as np


class ModelPredictiveControl:
    """
    Model Predictive Control (MPC) using Gradient Descent Optimization.

    Optimizes a sequence of control inputs u[0]...u[N-1] to minimize
    a cost function over a prediction horizon N.
    """

    def __init__(
        self, model_func, x0, horizon=10, dt=0.1, Q=None, R=None, u_min=-10, u_max=10
    ):
        """
        Args:
            model_func: Function f(x, u, dt) -> x_next
            x0: Initial State
            horizon: Prediction horizon steps (N)
            dt: Time step
            Q: State Cost Matrix (n x n)
            R: Input Cost Matrix (1 x 1 for SISO)
            u_min, u_max: Control constraints
        """
        self.f = model_func
        self.N = horizon
        self.dt = dt
        self.x_dim = len(x0)
        self.u_dim = 1

        self.Q = np.eye(self.x_dim) if Q is None else np.array(Q)
        self.R = np.eye(self.u_dim) if R is None else np.array(R)

        self.u_min = u_min
        self.u_max = u_max

        self.u_seq = np.zeros((self.N, self.u_dim))

    def _predict_trajectory(self, x_start, u_seq):
        """Simulates the system forward given a control sequence."""
        traj = [x_start]
        x_curr = x_start.copy()

        for i in range(self.N):
            x_curr = self.f(x_curr, u_seq[i], self.dt)
            traj.append(x_curr)

        return np.array(traj)

    def _cost_function(self, x_start, u_seq, x_ref):
        """Computes J = sum( (x-ref)^T Q (x-ref) + u^T R u )."""
        traj = self._predict_trajectory(x_start, u_seq)
        cost = 0.0

        for i in range(1, self.N + 1):
            error = traj[i] - x_ref
            cost += error.T @ self.Q @ error

        for i in range(self.N):
            cost += u_seq[i].T @ self.R @ u_seq[i]

        return cost

    def optimize(self, x_current, x_ref, learning_rate=0.01, iterations=50):
        """
        Solves the optimal control problem using projected gradient descent.
        Since we don't have analytical gradients, we use numerical finite differences.
        """
        epsilon = 1e-5

        for itr in range(iterations):
            grad_u = np.zeros_like(self.u_seq)
            base_cost = self._cost_function(x_current, self.u_seq, x_ref)

            look_ahead_grad = min(5, self.N)

            for k in range(look_ahead_grad):
                u_perturbed = self.u_seq.copy()
                u_perturbed[k] += epsilon

                new_cost = self._cost_function(x_current, u_perturbed, x_ref)

                grad_u[k] = (new_cost - base_cost) / epsilon

            self.u_seq = self.u_seq - learning_rate * grad_u

            self.u_seq = np.clip(self.u_seq, self.u_min, self.u_max)

        u_optimal = self.u_seq[0]

        self.u_seq = np.roll(self.u_seq, -1, axis=0)
        self.u_seq[-1] = 0.0

        return u_optimal
