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
    x_dot = x[1]
    theta = x[2]
    theta_dot = x[3]

    M = params["M"]  # Cart Mass
    m = params["m"]  # Pendulum Mass
    l = params["l"]  # Pendulum Length
    b = params.get("b", 0.0)  # Pendulum Friction
    g = params.get("g", 9.81)

    sin_t = np.sin(theta)
    cos_t = np.cos(theta)

    denom = M + m * (1 - cos_t**2)

    # --- FORCES ---
    # 1. Gravity Torque term
    term_grav = (M + m) * g * sin_t

    # 2. Coriolis/Centripetal & Input term (Coupled via Cosine)
    # Note: Friction is NOT in here.
    term_coupled = -cos_t * (u + m * l * theta_dot**2 * sin_t)

    # 3. Friction Term (Opposes angular velocity)
    # Must be scaled by (M+m) to match the numerator scaling of the derived equation
    # Eq: theta_ddot = [ (M+m)g sin - cos(u + ...) - (M+m)b*theta_dot/(ml) ] / (l * denom/M...?)
    # Easier form: Generalized Force Q_theta = -b * theta_dot
    # In the decoupled algebraic form used here:
    term_fric = -(M + m) * b * theta_dot / (m * l)

    # 4. Disturbance (Torque on hinge)
    term_dist = disturbance * (M + m) / (m * l)

    theta_ddot = (term_grav + term_coupled + term_fric + term_dist) / (l * denom)

    # Cart Acceleration
    term3 = u + m * l * theta_dot**2 * sin_t
    term4 = -m * g * sin_t * cos_t
    # Cart friction could be added: - b_cart * x_dot

    x_ddot = (term3 + term4) / denom

    return np.array([x_dot, x_ddot, theta_dot, theta_ddot])


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
