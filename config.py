"""
Central configuration for PyControls.
Adjust physical parameters, simulation settings, and controller gains here.
"""

# --- 1. Physical System Parameters (DC Motor) ---
MOTOR_PARAMS = {"J": 0.02, "b": 0.2, "K": 0.1, "R": 2.0, "L": 0.5}

# --- 2. Simulation Settings (Preset/Linear) ---
SIM_PARAMS = {
    "dt": 0.001,  # Time step (seconds)
    "t_end": 3.0,  # Duration of simulation (seconds)
    "step_volts": 1.0,  # Input voltage magnitude
}

# --- 3. Disturbance Injection ---
DISTURBANCE_PARAMS = {
    "enabled": True,
    "time": 1.5,  # Time to apply load (seconds)
    "magnitude": 0.5,  # Torque magnitude (N*m)
}

# --- 4. Controller Designs ---
CONTROLLERS = [
    {"name": "P (Weak)", "Kp": 50, "Ki": 0, "Kd": 0, "color": "#d62728"},
    {"name": "PI (Balanced)", "Kp": 40, "Ki": 50, "Kd": 0, "color": "#1f77b4"},
    {"name": "PID (Aggressive)", "Kp": 60, "Ki": 40, "Kd": 5, "color": "#2ca02c"},
]

# --- 5. Visualization Settings ---
PLOT_PARAMS = {
    "figsize": (14, 6),
    "grid_alpha": 0.3,
    "bode_range": (-1, 3, 500),  # (start_pow, end_pow, num_points)
    "marker_style": {"color": "black", "linestyle": "--", "alpha": 0.5},
    "marker_text": {
        "x_offset": 0.05,
        "y_pos": 0.1,
        "fontsize": 9,
        "label": "Load Torque\nApplied",
    },
}

# --- 6. Custom Non-Linear Simulation Settings ---
CUSTOM_SIM_PARAMS = {
    "dt": 0.001,
    "t_end": 5.0,
    "step_time": 0.5,  # Time when input steps up
    "step_magnitude": 1.0,  # Magnitude of input step
    "initial_state": (1, 1),  # Shape of the initial state vector (rows, cols)
}

# --- 7. Parameter Estimation Demo Settings (Option 6) ---
ESTIMATION_PARAMS = {
    # -- Simulation Timing --
    "dt": 0.001,  # Discrete time step for the estimation loop (s)
    "t_end": 15.0,  # Total duration of the estimation run (s)
    # -- The "True" Physical System --
    # These are the actual values we are trying to estimate.
    "true_system_params": {
        "J": 0.02,  # True Rotor Inertia (kg*m^2)
        "b": 0.2,  # True Viscous Friction (N*m*s)
        "K": 0.1,  # True Back-EMF/Torque Constant (V/(rad/s) or Nm/A)
        "R": 2.0,  # True Armature Resistance (Ohms)
        "L": 0.5,  # True Armature Inductance (H)
    },
    # -- EKF Initialization --
    # Initial guesses for the parameters (intentionally wrong to test convergence).
    "initial_guess_J": 0.005,
    "initial_guess_b": 0.1,
    # Scaling factor for the initial Error Covariance Matrix (P).
    # Higher values imply we are more uncertain about our initial guess.
    "p_init_scale": 0.1,
    # -- Noise Covariances --
    # Process Noise (Q) diagonals: [Speed, Current, J_est, b_est]
    # Represents uncertainty in the model evolution.
    "Q_init": [1e-4, 1e-4, 2e-4, 2e-4],
    # Measurement Noise (R) diagonals: [Speed, Current]
    # Represents uncertainty in sensors.
    "R": [0.01, 0.01],
    # Standard deviation of the random noise added to the simulated sensor readings.
    "sensor_noise_std": 0.05,
    # -- Adaptive Q Logic --
    # The demo switches Q matrices to "lock in" parameters after a certain time.
    "adaptive_enabled": True,
    # Q matrix during the 'Search' phase (allows parameters to change rapidly)
    "Q_search": [1e-4, 1e-4, 5e-4, 5e-4],
    # Q matrix during the 'Lock-in' phase (trusts the estimated parameters more)
    "Q_lock": [1e-4, 1e-4, 1e-8, 1e-8],
    # -- Input Signal --
    "input_amplitude": 5.0,  # Amplitude of the voltage square wave (V)
    "input_period": 2.0,  # Period of the input square wave (s)
}
