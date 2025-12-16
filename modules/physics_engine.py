import numpy as np


def dc_motor_dynamics(t, x, u, params, disturbance=0.0):
    """
    Computes the continuous-time dynamics of a DC motor.

    This function models the coupled electrical and mechanical equations:
    1. Mechanical: J * d(omega)/dt + b * omega = K * i - T_load
    2. Electrical: L * d(i)/dt + R * i = V - K * omega

    Args:
        t (float): Current time (unused for time-invariant systems, but required by solvers).
        x (np.ndarray): State vector [omega, current], where:
            - omega: Angular velocity (rad/s).
            - current: Armature current (Amps).
        u (float): Input voltage (V).
        params (dict): System parameters containing:
            - 'J': Rotor inertia (kg*m^2).
            - 'b': Viscous friction coefficient (N*m*s).
            - 'K': Electromotive force / Torque constant (V/(rad/s) or N*m/A).
            - 'R': Armature resistance (Ohms).
            - 'L': Armature inductance (Henries).
        disturbance (float, optional): External load torque (N*m). Defaults to 0.0.

    Returns:
        np.ndarray: The time derivative of the state vector [d(omega)/dt, d(i)/dt].
    """
    omega, current = x

    J = params["J"]
    b = params["b"]
    K = params["K"]
    R = params["R"]
    L = params["L"]

    domega_dt = (K * current - b * omega - disturbance) / J
    dcurrent_dt = (u - R * current - K * omega) / L

    return np.array([domega_dt, dcurrent_dt])


def pendulum_dynamics(t, x, u, params, disturbance=0.0):
    """
    Computes the nonlinear dynamics of an inverted pendulum on a cart.

    The system is modeled using Lagrangian mechanics, resulting in coupled second-order
    differential equations for the cart position (x) and pendulum angle (theta).

    The dynamics account for:
    1. Gravitational torque acting on the pendulum.
    2. Coriolis and Centripetal forces coupling the cart and pendulum motion.
    3. Viscous friction at the pendulum pivot.
    4. External forces on the cart and torque disturbances on the pendulum.

    State Vector x:
        [cart_position, cart_velocity, theta_angle, theta_velocity]

    Args:
        t (float): Current time.
        x (np.ndarray): The state vector.
        u (float): Horizontal force applied to the cart (Newtons).
        params (dict): System parameters containing:
            - 'M': Mass of the cart (kg).
            - 'm': Mass of the pendulum bob (kg).
            - 'l': Length of the pendulum (m).
            - 'b': Damping coefficient at the pivot (N*m*s).
            - 'g': Acceleration due to gravity (m/s^2).
        disturbance (float, optional): External torque applied to the pendulum hinge.

    Returns:
        np.ndarray: The state derivatives [v, a, theta_dot, theta_ddot].
    """
    x_dot = x[1]
    theta = x[2]
    theta_dot = x[3]

    M = params["M"]
    m = params["m"]
    l = params["l"]
    b = params.get("b", 0.0)
    g = params.get("g", 9.81)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom = M + m * (1 - cos_t**2)

    term_grav = (M + m) * g * sin_t

    term_coupled = -cos_t * (u + m * l * theta_dot**2 * sin_t)

    term_fric = -(M + m) * b * theta_dot / (m * l)

    term_dist = disturbance * (M + m) / (m * l)

    theta_ddot = (term_grav + term_coupled + term_fric + term_dist) / (l * denom)

    term3 = u + m * l * theta_dot**2 * sin_t
    term4 = -m * g * sin_t * cos_t

    x_ddot = (term3 + term4) / denom

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


def rk4_fixed_step(dynamics_func, x, u, dt, params, disturbance=0.0):
    """
    Implements a fixed-step Runge-Kutta 4th Order (RK4) integrator.

    This method provides a balance between accuracy and computational cost,
    making it suitable for real-time simulation loops where adaptive stepping
    might introduce unpredictable latency.

    The state update is calculated as:
        x_next = x + (dt/6) * (k1 + 2*k2 + 2*k3 + k4)

    Args:
        dynamics_func (callable): The function computing dx/dt = f(t, x, u, ...).
        x (np.ndarray): Current state vector.
        u (float or np.ndarray): Control input.
        dt (float): Fixed simulation time step.
        params (dict): System parameters passed to the dynamics function.
        disturbance (float, optional): External disturbance value.

    Returns:
        np.ndarray: The estimated state vector at time t + dt.
    """
    k1 = dynamics_func(0.0, x, u, params, disturbance)
    k2 = dynamics_func(0.0, x + 0.5 * dt * k1, u, params, disturbance)
    k3 = dynamics_func(0.0, x + 0.5 * dt * k2, u, params, disturbance)
    k4 = dynamics_func(0.0, x + dt * k3, u, params, disturbance)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
