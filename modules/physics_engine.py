import numpy as np


def dc_motor_dynamics(t, x, u, params, disturbance=0.0):
    """
    Continuous-time DC motor dynamics.

    Args:
        t: time (unused, included for solver compatibility)
        x: state vector [omega, current]
        u: input voltage
        params: dict with keys J, b, K, R, L
        disturbance: load torque (T_load)

    Returns:
        dx/dt as numpy array
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
    Linearized inverted pendulum dynamics (upright equilibrium).

    State:
        x = [theta, theta_dot]

    Input:
        u = horizontal force applied to cart

    Params expected:
        M : cart mass
        m : pendulum mass
        l : pendulum length
        b : damping coefficient
        g : gravity
    """
    theta, theta_dot = x

    M = params["M"]
    m = params["m"]
    l = params["l"]
    b = params.get("b", 0.0)
    g = params.get("g", 9.81)

    theta_ddot = (
        (g / l) * theta
        + (1.0 / (M * l)) * u
        - (b / (M * l * l)) * theta_dot
        + disturbance
    )

    return np.array([theta_dot, theta_ddot])


def rk4_fixed_step(dynamics_func, x, u, dt, params, disturbance=0.0):
    """
    Fixed-step RK4 integrator for real-time simulation.

    Args:
        dynamics_func: function(t, x, u, params, disturbance) -> dx/dt
        x: current state vector
        u: control input
        dt: fixed time step
        params: system parameters
        disturbance: external disturbance

    Returns:
        x_next: state after dt
    """
    k1 = dynamics_func(0.0, x, u, params, disturbance)
    k2 = dynamics_func(0.0, x + 0.5 * dt * k1, u, params, disturbance)
    k3 = dynamics_func(0.0, x + 0.5 * dt * k2, u, params, disturbance)
    k4 = dynamics_func(0.0, x + dt * k3, u, params, disturbance)

    return x + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
