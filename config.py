"""
Central Configuration Module for PyControls.

This module acts as the control center for the simulation suite. It contains
all physical parameters, simulation settings, controller gains, and visualization
preferences.
"""

import numpy as np

MOTOR_PARAMS = {
    "J": 0.02,
    "b": 0.2,
    "K": 0.1,
    "R": 2.0,
    "L": 0.5,
}

PENDULUM_PARAMS = {
    "M": 1.0,
    "m": 0.1,
    "l": 0.5,
    "b": 0.05,
    "g": 9.81,
    "theta_limit": 0.5,
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
    {
        "name": "P (Weak)",
        "Kp": 50,
        "Ki": 0,
        "Kd": 0,
        "color": "#d62728",
    },
    {
        "name": "PI (Balanced)",
        "Kp": 40,
        "Ki": 50,
        "Kd": 0,
        "color": "#1f77b4",
    },
    {
        "name": "PID (Aggressive)",
        "Kp": 60,
        "Ki": 40,
        "Kd": 10,
        "color": "#2ca02c",
    },
]

PLOT_PARAMS = {
    "figsize": (14, 6),
    "grid_alpha": 0.3,
    "bode_range": (-1, 3, 500),
    "marker_style": {
        "color": "black",
        "linestyle": "--",
        "alpha": 0.5,
    },
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
    "true_system_params": {
        "J": 0.02,
        "b": 0.2,
        "K": 0.1,
        "R": 2.0,
        "L": 0.5,
    },
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

DC_MOTOR_DEFAULTS = {
    "J": 0.01,
    "b": 0.1,
    "K": 0.01,
    "R": 1,
    "L": 0.5,
}

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
    "true_system_params": {
        "M": 1.0,
        "m": 0.1,
        "l": 0.5,
        "b": 0.05,
        "g": 9.81,
    },
    "initial_guess_m": 0.02,
    "initial_guess_l": 0.2,
    "p_init_scale": 0.1,
    "Q_init": [1e-5, 1e-5, 1e-5, 1e-5, 1e-2, 1e-2],
    "R": [0.001, 0.001],
    "sensor_noise_std": 0.01,
    "input_amplitude": 3.0,
    "input_period": 4.0,
}

UKF_PENDULUM_PARAMS = {
    "dt": 0.01,
    "t_end": 10.0,
    "x0": [1.57, 0],
    "P0": 0.1,
    "Q_diag": [0.001, 0.001],
    "R_diag": [0.01],
    "noise_std": 0.05,
    "alpha": 1e-3,
    "beta": 2.0,
    "kappa": 0.0,
}

UKF_MOTOR_PARAMS = {
    "dt": 0.001,
    "t_end": 4.0,
    "x0": [0.0, 0.0],
    "P0": 0.1,
    "Q_diag": [0.1, 0.1],
    "R_diag": [0.05, 0.05],
    "noise_std": 0.02,
    "alpha": 1e-3,
    "beta": 2.0,
    "kappa": 0.0,
    "coulomb_friction": 0.05,
    "viscous_friction": 0.1,
}

MPC_MOTOR_PARAMS = {
    "dt": 0.05,
    "horizon": 20,
    "Q_diag": [20.0, 0.0],
    "R_diag": [0.01],
    "u_min": -12.0,
    "u_max": 12.0,
    "target_speed": 2.5,
    "iterations": 50,
}

MPC_PENDULUM_PARAMS = {
    "dt": 0.02,
    "horizon": 100,
    "Q_diag": [1.0, 0.1, 20.0, 0.1],
    "R_diag": [0.1],
    "u_min": -20.0,
    "u_max": 20.0,
    "start_theta": np.pi,
    "iterations": 30,
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
- b (N*m*s): Viscous Friction Coefficient.
    * Represents energy loss due to friction proportional to speed.
- K (V/(rad/s) or Nm/A): Electromotive Force / Torque Constant.
    * Relates current to torque and speed to back-EMF.
- R (Ohms): Armature Resistance.
    * Electrical resistance of the motor windings.
- L (Henries): Armature Inductance.
    * Electrical inertia of the motor windings.

--------------------------------------------------------------------------------
2. SIM_PARAMS (Linear Simulation)
--------------------------------------------------------------------------------
Settings for the standard Option 1 dashboard simulation.

Parameters:
- dt (s): Time Step.
    * The time increment between simulation iterations.
- t_end (s): Simulation Duration.
    * Total length of time to simulate.
- step_volts (V): Input Magnitude.
    * The voltage applied at t=0 (Step Input).

--------------------------------------------------------------------------------
3. DISTURBANCE_PARAMS (External Load)
--------------------------------------------------------------------------------
Simulates an external torque applied to the motor shaft.

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
- Kp (Proportional): Reaction to current error.
- Ki (Integral): Reaction to accumulated past error.
- Kd (Derivative): Reaction to the rate of change of error.

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
- dt: Initial time step guess.
- step_time: Time when the step input triggers.
- step_magnitude: Amplitude of the input.
- initial_state: Tuple (rows, cols) defining the state vector shape.

--------------------------------------------------------------------------------
7. ESTIMATION_PARAMS (EKF Demo) & PENDULUM_ESTIMATION_PARAMS
--------------------------------------------------------------------------------
Configuration for Option 6 Parameter Estimation.

- Q_init: Initial Process Noise Covariance. High values allow parameters to adapt.
- R: Measurement Noise Covariance.
- p_init_scale: Scale of the initial state covariance P0.
- initial_guess_*: Starting values for the parameters being estimated.
- adaptive_enabled: (Motor only) Toggles adaptive Q logic.

--------------------------------------------------------------------------------
8. UKF_PARAMS / UKF_PENDULUM_PARAMS / UKF_MOTOR_PARAMS
--------------------------------------------------------------------------------
Configuration for Option 7 Unscented Kalman Filter.

- alpha, beta, kappa: Sigma point spread parameters.
    * Alpha: Spread of sigma points around mean (usually small, 1e-3).
    * Beta: Incorporates prior knowledge of distribution (2 is optimal for Gaussian).
    * Kappa: Secondary scaling parameter (usually 0).
- Q_diag: Diagonal elements of Process Noise Covariance.
- R_diag: Diagonal elements of Measurement Noise Covariance.
- noise_std: Standard deviation of synthetic noise added to simulation.

--------------------------------------------------------------------------------
9. MPC_PARAMS / MPC_MOTOR_PARAMS / MPC_PENDULUM_PARAMS
--------------------------------------------------------------------------------
Configuration for Option 8 Model Predictive Control.

- horizon: Number of future time steps to optimize over.
    * Short horizon: Fast computation, myopic behavior.
    * Long horizon: Slower computation, better strategic planning (e.g., swing-up).
- dt: Time step used internally by the MPC solver.
- Q_diag: State deviation penalty weights.
    * High values force the controller to track that state closely.
- R_diag: Control effort penalty weights.
    * High values make the controller "lazy" or energy-efficient.
- u_min / u_max: Hard constraints on control input (Saturation).
- iterations: Number of optimization steps per cycle (ADMM or iLQR).
"""
