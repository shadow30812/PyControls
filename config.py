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
