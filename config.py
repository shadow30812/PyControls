"""
Central Configuration Module for PyControls.

This module acts as the control center for the simulation suite. It contains
all physical parameters, simulation settings, controller gains, and visualization
preferences."""

MOTOR_PARAMS = {"J": 0.02, "b": 0.2, "K": 0.1, "R": 2.0, "L": 0.5}

SIM_PARAMS = {
    "dt": 0.001,
    "t_end": 3.0,
    "step_volts": 1.0,
}

DISTURBANCE_PARAMS = {
    "enabled": True,
    "time": 1.5,
    "magnitude": 0.5,
}

CONTROLLERS = [
    {"name": "P (Weak)", "Kp": 50, "Ki": 0, "Kd": 0, "color": "#d62728"},
    {"name": "PI (Balanced)", "Kp": 40, "Ki": 50, "Kd": 0, "color": "#1f77b4"},
    {"name": "PID (Aggressive)", "Kp": 60, "Ki": 40, "Kd": 5, "color": "#2ca02c"},
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
