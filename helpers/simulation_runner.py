"""
Pure simulation runners for PyControls.
All functions here are headless and return raw numerical results.
"""

import numpy as np

import helpers.config as config
from core.control_utils import PIDController
from core.ekf import ExtendedKalmanFilter
from core.estimator import KalmanFilter
from core.math_utils import make_system_func
from core.mpc import ModelPredictiveControl
from core.solver import ExactSolver, NonlinearSolver
from core.ukf import UnscentedKalmanFilter
from modules.physics_engine import pendulum_dynamics, rk4_fixed_step


def run_linear_simulation(
    system_instance, system_id, ctrl_config, sim_params, dist_params
):
    dt = sim_params["dt"]
    t_end = sim_params["t_end"]

    if not hasattr(system_instance, "get_state_space"):
        return np.array([]), np.array([]), np.array([]), np.array([])

    ss_real = system_instance.get_state_space()
    solver_real = ExactSolver(ss_real.A, ss_real.B, ss_real.C, ss_real.D, dt)

    kf = None
    if hasattr(system_instance, "get_augmented_state_space"):
        ss_aug = system_instance.get_augmented_state_space()
        solver_aug_math = ExactSolver(ss_aug.A, ss_aug.B, ss_aug.C, ss_aug.D, dt)

        n_states = ss_aug.A.shape[0]
        Q = np.eye(n_states) * config.PRESET_SIM_PARAMS["kf_Q_scale"]
        Q[-1, -1] = config.PRESET_SIM_PARAMS["kf_Q_dist_scale"]
        R = np.eye(ss_aug.C.shape[0]) * config.PRESET_SIM_PARAMS["kf_R_scale"]

        kf = KalmanFilter(
            solver_aug_math.Phi,
            solver_aug_math.Gamma,
            ss_aug.C,
            Q,
            R,
            x0=np.zeros(n_states),
        )

    use_lqr = system_id == "pendulum"
    lqr_K = None
    pid = None

    if use_lqr:
        if hasattr(system_instance, "dlqr_gain"):
            lqr_K = system_instance.dlqr_gain()
        else:
            return np.array([]), np.array([]), np.array([]), np.array([])
    else:
        pid = PIDController(
            Kp=ctrl_config["Kp"],
            Ki=ctrl_config["Ki"],
            Kd=ctrl_config["Kd"],
            derivative_on_measurement=False,
            output_limits=config.PRESET_SIM_PARAMS["pid_output_limits"],
            tau=config.PRESET_SIM_PARAMS["pid_tau"],
        )

    t_values = np.linspace(0, t_end, int(t_end / dt))
    y_real_hist = []
    x_est_hist = []
    u_hist = []

    if use_lqr:
        solver_real.x = np.array([[0.0], [0.0], [0.1], [0.0]])
        if kf:
            kf.x_hat[:4] = solver_real.x

    for t in t_values:
        dist_val = 0.0
        if dist_params["enabled"] and t >= dist_params["time"]:
            dist_val = dist_params["magnitude"]

        if system_id == "pendulum":
            ref_signal = sim_params["step_angle"] if t > 0 else 0
        else:
            ref_signal = sim_params["step_volts"] if t > 0 else 0

        if system_id == "pendulum":
            feedback_vec = kf.x_hat[:4] if kf is not None else solver_real.x
        else:
            x_idx = 0
            feedback = kf.x_hat[x_idx, 0] if kf is not None else solver_real.x[x_idx, 0]

        if use_lqr:
            u_val = -(lqr_K @ feedback_vec).item()
            u_val = max(
                config.PRESET_SIM_PARAMS["lqr_clip_min"],
                min(config.PRESET_SIM_PARAMS["lqr_clip_max"], u_val),
            )
        else:
            u_val = pid.update(measurement=feedback, setpoint=ref_signal, dt=dt)

        u_hist.append(u_val)

        if system_id == "pendulum":
            u_vector = np.array([[u_val]])
        else:
            u_vector = np.array([[u_val], [dist_val]])

        y_real_vector = solver_real.step(u_vector)

        noise = np.random.normal(
            0, config.PRESET_SIM_PARAMS["noise_std"], size=y_real_vector.shape
        )
        y_meas = y_real_vector + noise

        if kf:
            kf.predict(np.array([[u_val]]))
            kf.update(y_meas)
            x_est_hist.append(kf.x_hat.flatten())

        y_real_hist.append(y_real_vector.flatten())

    return (
        t_values,
        np.array(y_real_hist),
        np.array(x_est_hist),
        np.array(u_hist),
    )


def run_ekf_simulation(SystemClass, system_id, est_cfg):
    if system_id == "dc_motor":
        param_keys = ["J", "b"]

        def h_meas(x):
            return x[:2]

        x0_est = [
            0,
            0,
            np.log(est_cfg["initial_guess_J"]),
            np.log(est_cfg["initial_guess_b"]),
        ]
        true_indices = [0, 1]
        est_indices = [0, 1]

    else:
        param_keys = ["m", "l"]

        def h_meas(x):
            return np.array([x[0], x[2]])

        x0_est = [
            0,
            0,
            0,
            0,
            np.log(est_cfg["initial_guess_m"]),
            np.log(est_cfg["initial_guess_l"]),
        ]
        true_indices = [0, 2]
        est_indices = [0, 2]

    dt = est_cfg["dt"]
    t_end = est_cfg["t_end"]
    true_params = est_cfg["true_system_params"]

    true_system = SystemClass(**true_params)
    f_dyn_est = true_system.get_parameter_estimation_func()

    Q = np.diag(est_cfg["Q_init"])
    R = np.diag(est_cfg["R"])

    ekf = ExtendedKalmanFilter(
        f_dyn_est,
        h_meas,
        Q,
        R,
        x0_est,
        p_init_scale=est_cfg["p_init_scale"],
    )

    t_vals = np.linspace(0, t_end, int(t_end / dt))

    history = {
        "t": t_vals,
        "y1_true": [],
        "y1_est": [],
        "y2_true": [],
        "y2_est": [],
        "p1_est": [],
        "p2_est": [],
    }

    amp = est_cfg["input_amplitude"]
    period = est_cfg["input_period"]
    noise_std = est_cfg["sensor_noise_std"]

    if system_id == "pendulum":
        x_true = np.zeros(4)
    else:
        ss_true = true_system.get_state_space()
        solver_true = ExactSolver(ss_true.A, ss_true.B, ss_true.C, ss_true.D, dt=dt)

    for t in t_vals:
        u_val = amp if (t % period) < (period / 2.0) else 0.0

        if system_id == "pendulum":
            x_true = rk4_fixed_step(pendulum_dynamics, x_true, u_val, dt, true_params)
            y_true_full = x_true
        else:
            y_true_full = solver_true.step(np.array([[u_val], [0]]))

        meas_clean = (
            np.array([y_true_full[0], y_true_full[2]])
            if system_id == "pendulum"
            else y_true_full.flatten()
        )

        y_meas = meas_clean.reshape(-1, 1) + np.random.normal(0, noise_std, (2, 1))

        ekf.predict(np.array([[u_val]]), dt)
        x_hat = ekf.update(y_meas)

        history["y1_true"].append(y_true_full[true_indices[0]])
        history["y1_est"].append(x_hat[est_indices[0]])
        history["y2_true"].append(y_true_full[true_indices[1]])
        history["y2_est"].append(x_hat[est_indices[1]])
        history["p1_est"].append(np.exp(x_hat[-2]))
        history["p2_est"].append(np.exp(x_hat[-1]))

    return t_vals, history, true_params, param_keys


def run_ukf_simulation(system, system_id, cfg):
    dt = cfg["dt"]

    f_dyn, h_meas = system.get_nonlinear_dynamics()

    x0 = cfg["x0"]
    P0 = np.eye(len(x0)) * cfg["P0"]
    Q = np.diag(cfg["Q_diag"])
    R = np.diag(cfg["R_diag"])

    ukf = UnscentedKalmanFilter(
        f_dyn,
        h_meas,
        Q,
        R,
        x0,
        P0,
        alpha=cfg["alpha"],
        beta=cfg["beta"],
        kappa=cfg["kappa"],
    )

    t_vals = np.arange(0, cfg["t_end"], dt)
    true_states = []
    est_states = []
    measurements = []

    curr_x = np.array(x0)

    for t in t_vals:
        u = 2.0 * np.sin(2.0 * t) if system_id == "dc_motor" else 0.0

        curr_x = f_dyn(curr_x, u, dt)
        true_states.append(curr_x)

        z_clean = h_meas(curr_x)
        z_noisy = z_clean + np.random.normal(0, cfg["noise_std"], size=z_clean.shape)
        measurements.append(z_noisy)

        ukf.predict(u, dt)
        est_x = ukf.update(z_noisy)
        est_states.append(est_x)

    return (
        t_vals,
        np.array(true_states),
        np.array(est_states),
        np.array(measurements),
    )


def run_mpc_simulation(system, system_id, cfg):
    dt = cfg["dt"]

    if system_id == "dc_motor":
        A_d, B_d = system.get_mpc_model(dt)
        model_func = None
        x0 = np.array([0.0, 0.0])
        ref = np.array([cfg["target_speed"], 0.0])
        plot_idx = 0
    else:
        model_func = system.get_mpc_model(dt)
        A_d, B_d = None, None
        x0 = np.array([0.0, 0.0, cfg["start_theta"], 0.0])
        ref = np.zeros(4)
        plot_idx = 2

    Q = np.diag(cfg["Q_diag"])
    R = np.diag(cfg["R_diag"])

    mpc = ModelPredictiveControl(
        model_func=model_func,
        A=A_d,
        B=B_d,
        x0=x0,
        horizon=cfg["horizon"],
        dt=dt,
        Q=Q,
        R=R,
        u_min=cfg["u_min"],
        u_max=cfg["u_max"],
    )

    t_vals = np.arange(0, dt * cfg["horizon"] * 3, dt)
    x_hist = []
    u_hist = []
    curr_x = x0.copy()

    mpc_stride = config.MPC_SOLVER_PARAMS["mpc_stride"]

    for i in range(len(t_vals)):
        u_opt = mpc.optimize(
            curr_x, ref, iterations=cfg["iterations"] if (i % mpc_stride == 0) else 0
        )
        x_hist.append(curr_x)
        u_hist.append(u_opt[0])

        if mpc.mode == "linear":
            curr_x = A_d @ curr_x + B_d @ u_opt
        else:
            curr_x = model_func(curr_x, u_opt, dt)

    return t_vals, np.array(x_hist), np.array(u_hist), ref, plot_idx


def run_custom_nonlinear_simulation(eqn, sim_cfg):
    dyn_func = make_system_func(eqn)
    x0 = np.zeros(sim_cfg["initial_state"]).flatten()

    solver = NonlinearSolver(
        dynamics_func=dyn_func,
        dt_min=sim_cfg["dt_min"],
        dt_max=sim_cfg["dt_max"],
    )

    step_time = sim_cfg["step_time"]

    def input_signal(t):
        return sim_cfg["step_magnitude"] if t > step_time else 0.0

    t_vals, states = solver.solve_adaptive(
        t_end=sim_cfg["t_end"], x0=x0, u_func=input_signal
    )

    y_vals = states[:, 0] if states.ndim > 1 else states

    return t_vals, y_vals
