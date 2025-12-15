"""
Central Configuration Module for PyControls.

This module acts as the control center for the simulation suite. It contains
all physical parameters, simulation settings, controller gains, and visualization
preferences."""

import numpy as np

MOTOR_PARAMS = {"J": 0.02, "b": 0.2, "K": 0.1, "R": 2.0, "L": 0.5}

PENDULUM_PARAMS = {
    "M": 1.0,  # cart mass (kg)
    "m": 0.1,  # pendulum mass (kg)
    "l": 0.5,  # pendulum length (m)
    "b": 0.05,  # damping coefficient
    "g": 9.81,  # gravity (m/s^2)
    "theta_limit": 0.5,  # radians
}


SIM_PARAMS = {
    "dt": 0.001,
    "t_end": 3.0,
    "step_volts": 1.0,
    "step_angle": 0.0,
}

DISTURBANCE_PARAMS = {
    "enabled": True,
    "time": 1.5,
    "magnitude": 0.5,
}

CONTROLLERS = [
    {"name": "P (Weak)", "Kp": 50, "Ki": 0, "Kd": 0, "color": "#d62728"},
    {"name": "PI (Balanced)", "Kp": 40, "Ki": 50, "Kd": 0, "color": "#1f77b4"},
    {"name": "PID (Aggressive)", "Kp": 60, "Ki": 40, "Kd": 10, "color": "#2ca02c"},
]

PLOT_PARAMS = {
    "figsize": (14, 6),
    "grid_alpha": 0.3,
    "bode_range": (-1, 3, 500),
    "marker_style": {"color": "black", "linestyle": "--", "alpha": 0.5},
    "marker_text": {
        "x_offset": 0.05,
        "y_pos": 0.1,
        "fontsize": 9,
        "label": "Load Torque\nApplied",
    },
}

CUSTOM_SIM_PARAMS = {
    "dt": 0.001,
    "t_end": 5.0,
    "step_time": 0.5,
    "step_magnitude": 1.0,
    "initial_state": (1, 1),
}

ESTIMATION_PARAMS = {
    "dt": 0.001,
    "t_end": 15.0,
    "true_system_params": {"J": 0.02, "b": 0.2, "K": 0.1, "R": 2.0, "L": 0.5},
    "initial_guess_J": 0.005,
    "initial_guess_b": 0.1,
    "p_init_scale": 0.1,
    "Q_init": [1e-4, 1e-4, 2e-4, 2e-4],
    "R": [0.01, 0.01],
    "sensor_noise_std": 0.05,
    "adaptive_enabled": True,
    "Q_search": [1e-4, 1e-4, 5e-4, 5e-4],
    "Q_lock": [1e-4, 1e-4, 1e-8, 1e-8],
    "input_amplitude": 5.0,
    "input_period": 2.0,
}

UKF_PARAMS = {
    "dt": 0.01,
    "t_end": 10.0,
    "noise_std": 0.1,
    "Q": [0.01, 0.01],
    "R": [0.1],
    "alpha": 1e-3,
    "beta": 2.0,
    "kappa": 0.0,
}

MPC_PARAMS = {
    "dt": 0.05,
    "t_end": 10.0,
    "horizon": 20,
    "Q_weight": [10.0, 0.0],
    "R_weight": [0.1],
    "u_min": -12.0,
    "u_max": 12.0,
    "learning_rate": 0.5,
    "iterations": 20,
}

DC_MOTOR_DEFAULTS = {"J": 0.01, "b": 0.1, "K": 0.01, "R": 1, "L": 0.5}

PENDULUM_LQR_PARAMS = {
    "Q_diag": [5.0, 1.0, 10.0, 1.0],
    "R_val": 0.1,
}

PRESET_SIM_PARAMS = {
    "kf_Q_scale": 1e-4,
    "kf_Q_dist_scale": 1e-2,
    "kf_R_scale": 0.01,
    "pid_output_limits": (-12, 12),
    "pid_tau": 0.02,
    "lqr_clip_min": -20,
    "lqr_clip_max": 20,
    "noise_std": 0.01,
}

INTERACTIVE_LAB_PARAMS = {
    "omega_ref": 1.0,
    "controller_min_dt": 0.001,
    "ekf_x0": [0.0, 0.0, 0.05, 0.0],
    "ekf_Q_diag": [1e-4, 1e-4, 1e-4, 1e-4],
    "ekf_R_diag": [1e-3, 1e-3],
}

MPC_SOLVER_PARAMS = {
    "rho": 1.0,
    "finite_diff_eps": 1e-5,
    "ilqr_reg": 1e-6,
    "default_linear_iters": 50,
    "default_nonlinear_iters": 10,
}

SOLVER_PARAMS = {
    "matrix_exp_order": 20,
    "adaptive_dt_min": 1e-5,
    "adaptive_dt_max": 0.5,
    "adaptive_tol": 1e-6,
    "adaptive_initial_dt": 0.001,
    "safety_factor_1": 0.9,
    "safety_factor_2": 0.2,
}

PENDULUM_ESTIMATION_PARAMS = {
    "dt": 0.01,
    "t_end": 20.0,
    "true_system_params": {"M": 1.0, "m": 0.1, "l": 0.5, "b": 0.05, "g": 9.81},
    "initial_guess_m": 0.02,  # Start with 20% of true mass
    "initial_guess_l": 0.2,  # Start with incorrect length
    "p_init_scale": 0.1,
    # State: [x, v, theta, omega, log(m), log(l)]
    "Q_init": [1e-5, 1e-5, 1e-5, 1e-5, 1e-2, 1e-2],
    "R": [0.001, 0.001],  # Measuring x and theta
    "sensor_noise_std": 0.01,
    "input_amplitude": 3.0,  # Force in Newtons
    "input_period": 4.0,  # Slower period for mechanical system
}

UKF_PENDULUM_PARAMS = {
    "dt": 0.01,
    "t_end": 10.0,
    "x0": [1.57, 0],  # Start at 90 degrees (horizontal)
    "P0": 0.1,  # Initial uncertainty
    "Q_diag": [0.001, 0.001],  # Process noise (Angle, Velocity)
    "R_diag": [0.01],  # Measurement noise (Angle only)
    "noise_std": 0.05,
    "alpha": 1e-3,
    "beta": 2.0,
    "kappa": 0.0,
}

UKF_MOTOR_PARAMS = {
    "dt": 0.001,
    "t_end": 4.0,
    "x0": [0.0, 0.0],  # [Speed, Current]
    "P0": 0.1,
    "Q_diag": [0.1, 0.1],  # Process noise
    "R_diag": [0.05, 0.05],  # Measurement noise
    "noise_std": 0.02,
    "alpha": 1e-3,
    "beta": 2.0,
    "kappa": 0.0,
    # Stiction Parameters
    "coulomb_friction": 0.05,  # Static friction torque (Nm)
    "viscous_friction": 0.1,  # Standard damping
}

MPC_MOTOR_PARAMS = {
    "dt": 0.05,
    "horizon": 20,
    "Q_diag": [20.0, 0.0],  # Heavy penalty on Speed error, some on Current
    "R_diag": [0.01],  # Small penalty on Voltage (Cheap control)
    "u_min": -12.0,
    "u_max": 12.0,
    "target_speed": 2.5,  # rad/s
    "iterations": 50,  # ADMM iterations
}

MPC_PENDULUM_PARAMS = {
    "dt": 0.02,
    "horizon": 100,  # Longer horizon needed to "see" the swing up
    "Q_diag": [1.0, 0.1, 20.0, 0.1],  # Penalize [x, v, theta, omega]
    # Note: Theta penalty is high to force upright position
    "R_diag": [0.1],
    "u_min": -20.0,
    "u_max": 20.0,
    "start_theta": np.pi,  # Hanging down
    "iterations": 30,  # iLQR iterations
}

"""
--------------------------------------------------------------------------------
1. MOTOR_PARAMS (Physical System)
--------------------------------------------------------------------------------
These parameters define the physics of the DC Motor plant model.
Differential Equations:
    (1) Electrical: V = I*R + L*(dI/dt) + K*omega
    (2) Mechanical: K*I = J*(domega/dt) + b*omega

Parameters:
- J (kg*m^2): Rotor Inertia.
    * Represents the resistance of the rotor to changes in rotation speed.
    * HIGHER J: Slower acceleration/deceleration; system feels "heavier".
    * LOWER J: Faster response time; system feels "lighter" and twitchier.

- b (N*m*s): Viscous Friction Coefficient.
    * Represents energy loss due to friction proportional to speed.
    * HIGHER b: Lower steady-state speed for the same voltage; more natural damping.
    * LOWER b: Higher max speed; system coasts longer after power is cut.

- K (V/(rad/s) or Nm/A): Electromotive Force / Torque Constant.
    * Relates current to torque and speed to back-EMF.
    * HIGHER K: More torque per amp (stronger acceleration) but also more back-EMF (limits max speed).
    * LOWER K: Less torque per amp but higher theoretical max speed.

- R (Ohms): Armature Resistance.
    * Electrical resistance of the motor windings.
    * HIGHER R: Reduces the stall current; limits max torque; slower electrical response.
    * LOWER R: Allows higher currents; higher power consumption and torque.

- L (Henries): Armature Inductance.
    * Electrical inertia of the motor windings.
    * HIGHER L: Smoothens current spikes; delays torque production (lag).
    * LOWER L: Current changes instantly with voltage changes.

--------------------------------------------------------------------------------
2. SIM_PARAMS (Linear Simulation)
--------------------------------------------------------------------------------
Settings for the standard Option 1 dashboard simulation.

Parameters:
- dt (s): Time Step.
    * The time increment between simulation iterations.
    * SMALLER dt: Higher accuracy; handles fast dynamics/stiff systems better; slower computation.
    * LARGER dt: Faster computation; risks instability if dt > system time constants.

- t_end (s): Simulation Duration.
    * Total length of time to simulate.

- step_volts (V): Input Magnitude.
    * The voltage applied at t=0 (Step Input).
    * Changing this tests linearity (in linear models, output scales perfectly; in non-linear, it may saturate).

--------------------------------------------------------------------------------
3. DISTURBANCE_PARAMS (External Load)
--------------------------------------------------------------------------------
Simulates an external torque applied to the motor shaft (e.g., trying to stop it with your hand).

Parameters:
- enabled (bool): Master switch for disturbance logic.
- time (s): Onset Time.
    * The exact moment in the simulation when the load torque is applied.
- magnitude (Nm): Load Torque.
    * The amount of opposing force applied. Positive values oppose forward rotation.

--------------------------------------------------------------------------------
4. CONTROLLERS (PID Tuning)
--------------------------------------------------------------------------------
A list of controller configurations to test simultaneously.
Formula: V = Kp*e + Ki*∫e + Kd*(de/dt)

Parameters:
- Kp (Proportional):
    * Reaction to current error.
    * INCREASING: Faster rise time; reduces steady-state error; increases overshoot.
- Ki (Integral):
    * Reaction to accumulated past error.
    * INCREASING: Eliminates steady-state error; increases overshoot/oscillation; can cause instability.
- Kd (Derivative):
    * Reaction to the rate of change of error.
    * INCREASING: Reduces overshoot; adds damping; slows down response; highly sensitive to noise.

--------------------------------------------------------------------------------
5. PLOT_PARAMS (Visualization)
--------------------------------------------------------------------------------
Settings for the Matplotlib interface.

Parameters:
- bode_range: (start_power, end_power, num_points).
    * Defines frequency range 10^start to 10^end for Bode plots.

--------------------------------------------------------------------------------
6. CUSTOM_SIM_PARAMS (Adaptive Solver)
--------------------------------------------------------------------------------
Settings for the RK45 non-linear solver (Option 2).

Parameters:
- dt: Initial time step guess (solver will adjust this automatically).
- step_time: Time when the step input triggers.
- step_magnitude: Amplitude of the input.
- initial_state: Tuple (rows, cols) defining the state vector shape.

--------------------------------------------------------------------------------
7. ESTIMATION_PARAMS (Extended Kalman Filter Demo)
--------------------------------------------------------------------------------
Configuration for the Option 6 Parameter Estimation experiment.

Simulation Timing:
- dt (s): Discrete time step for the EKF loop.
- t_end (s): Total duration of the experiment.

True System (The "Reality"):
- true_system_params: A dictionary of physical parameters (J, b, K, R, L) used to generate
  the "real world" data that the EKF will observe.

EKF Initialization (The "Learner"):
- initial_guess_J: Starting estimate for Inertia. Set widely different from 'true' J to test convergence.
- initial_guess_b: Starting estimate for Friction.
- p_init_scale: Initial Uncertainty.
    * Multiplier for the initial Covariance Matrix P.
    * HIGH VALUE: "I have no idea what the parameters are" (Filter moves fast).
    * LOW VALUE: "I am confident in my guess" (Filter is stubborn).

Noise Covariances:
- Q_init (Process Noise): [Speed, Current, J, b].
    * How much we think the state changes unpredictably.
    * High values on J/b tell the filter "Parameters might be changing", allowing it to adapt.
- R (Measurement Noise): [Speed, Current].
    * How much noise is in our sensors.
    * HIGH R: Trust model more (smooth estimate).
    * LOW R: Trust sensors more (noisy estimate).
- sensor_noise_std: The actual noise added to the synthetic "real" data.

Adaptive Logic:
- adaptive_enabled: If True, uses two different Q matrices.
- Q_search: High-variance Q used early to find parameters quickly.
- Q_lock: Low-variance Q used late to stabilize the result.

Input Signal:
- input_amplitude (V): Amplitude of the square wave excitation.
- input_period (s): Frequency of the excitation.
    * Constant voltage is bad for estimation; dynamic input is required to make parameters observable.

--------------------------------------------------------------------------------
8. UKF_PARAMS (Unscented Kalman Filter)
--------------------------------------------------------------------------------
- alpha, beta, kappa: Parameters determining the spread of sigma points.
  Standard values: alpha=1e-3, beta=2 (Gaussian), kappa=0.

- Q: Process Noise
- R: Sensor Noise

--------------------------------------------------------------------------------
9. MPC_PARAMS (Model Predictive Control)
--------------------------------------------------------------------------------
- horizon: How many steps into the future the controller looks.
- Q_weight: Penalty for state deviation from setpoint. High value = Tight tracking.
- R_weight: Penalty for using voltage. High value = Lazy controller / Save energy.
- learning_rate: For the internal gradient descent optimizer.
"""
