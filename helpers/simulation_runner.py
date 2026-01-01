"""
Pure simulation runners for PyControls.
All functions here are headless and return raw numerical results.
"""

from typing import Any, Callable, Dict, List, Optional, Tuple, Type, TypeGuard, Union

import numpy as np
from numpy.typing import NDArray

import helpers.config as config
from core.control_utils import PIDController
from core.ekf import ExtendedKalmanFilter
from core.estimator import KalmanFilter
from core.math_utils import make_system_func
from core.mpc import ModelPredictiveControl
from core.solver import ExactSolver, NonlinearSolver
from core.state_space import StateSpace
from core.ukf import UnscentedKalmanFilter
from modules.physics_engine import pendulum_dynamics, rk4_fixed_step

ScalarFunc = Callable[..., Union[float, complex]]
VectorFunc = Callable[..., NDArray[Any]]
AnyFunc = Callable[..., Union[float, complex, NDArray[Any]]]


def is_scalar_func(f: AnyFunc) -> TypeGuard[ScalarFunc]:
    try:
        out = f(0.0, np.zeros(1), 0.0)
    except Exception:
        return False
    return not isinstance(out, np.ndarray)


def run_linear_simulation(
    system_instance: Any,
    system_id: str,
    ctrl_config: Dict[str, float],
    sim_params: Dict[str, Any],
    dist_params: Dict[str, Any],
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    dt: float = sim_params["dt"]
    t_end: float = sim_params["t_end"]

    if not hasattr(system_instance, "get_state_space"):
        return np.array([]), np.array([]), np.array([]), np.array([])

    ss_real: StateSpace = system_instance.get_state_space()
    solver_real = ExactSolver(ss_real.A, ss_real.B, ss_real.C, ss_real.D, dt)

    kf: Optional[KalmanFilter] = None
    if hasattr(system_instance, "get_augmented_state_space"):
        ss_aug: StateSpace = system_instance.get_augmented_state_space()
        solver_aug_math = ExactSolver(ss_aug.A, ss_aug.B, ss_aug.C, ss_aug.D, dt)

        n_states: int = ss_aug.A.shape[0]
        Q: NDArray[np.float64] = (
            np.eye(n_states) * config.PRESET_SIM_PARAMS["kf_Q_scale"]
        )
        Q[-1, -1] = config.PRESET_SIM_PARAMS["kf_Q_dist_scale"]
        R: NDArray[np.float64] = (
            np.eye(ss_aug.C.shape[0]) * config.PRESET_SIM_PARAMS["kf_R_scale"]
        )

        kf = KalmanFilter(
            solver_aug_math.Phi,
            solver_aug_math.Gamma,
            ss_aug.C,
            Q,
            R,
            x0=np.zeros(n_states),
        )

    use_lqr: bool = system_id == "pendulum"
    lqr_K: Optional[NDArray[np.float64]] = None
    pid: Optional[PIDController] = None

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

    t_values: NDArray[np.float64] = np.linspace(0, t_end, int(t_end / dt))
    y_real_hist: List[NDArray[np.float64]] = []
    x_est_hist: List[NDArray[np.float64]] = []
    u_hist: List[float] = []

    if use_lqr:
        solver_real.x = np.array([[0.0], [0.0], [0.1], [0.0]])
        if kf:
            kf.x_hat[:4] = solver_real.x

    feedback: Any = None
    feedback_vec: Optional[NDArray[np.float64]] = None
    for t in t_values:
        dist_val = 0.0
        if dist_params["enabled"] and t >= dist_params["time"]:
            dist_val: float = dist_params["magnitude"]

        if system_id == "pendulum":
            ref_signal: float = sim_params["step_angle"] if t > 0 else 0
        else:
            ref_signal = sim_params["step_volts"] if t > 0 else 0

        if system_id == "pendulum":
            feedback_vec = kf.x_hat[:4] if kf is not None else solver_real.x
        else:
            x_idx = 0
            feedback = kf.x_hat[x_idx, 0] if kf is not None else solver_real.x[x_idx, 0]

        if use_lqr:
            assert lqr_K is not None
            assert feedback_vec is not None
            u_val = -(lqr_K @ feedback_vec).item()
            u_val = max(
                config.PRESET_SIM_PARAMS["lqr_clip_min"],
                min(config.PRESET_SIM_PARAMS["lqr_clip_max"], u_val),
            )
        else:
            assert pid is not None
            assert feedback is not None
            u_val = pid.update(measurement=feedback, setpoint=ref_signal, dt=dt)

        u_hist.append(u_val)

        if system_id == "pendulum":
            u_vector: NDArray[np.float64] = np.array([[u_val]])
        else:
            u_vector = np.array([[u_val], [dist_val]])

        y_real_vector: NDArray[np.float64] = np.asarray(solver_real.step(u_vector))

        if isinstance(y_real_vector, np.ndarray):
            noise: NDArray[np.float64] = np.random.normal(
                0, config.PRESET_SIM_PARAMS["noise_std"], size=y_real_vector.shape
            )
            y_meas: NDArray[np.float64] = y_real_vector + noise

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


def run_ekf_simulation(
    SystemClass: Type[Any], system_id: str, est_cfg: Dict[str, Any]
) -> Tuple[NDArray[np.float64], Dict[str, Any], Dict[str, float], List[str]]:
    if system_id == "dc_motor":
        param_keys: List[str] = ["J", "b"]
        h_meas: Callable[[np.ndarray], np.ndarray] = lambda x: x[:2]

        x0_est: List[float] = [
            0,
            0,
            np.log(est_cfg["initial_guess_J"]),
            np.log(est_cfg["initial_guess_b"]),
        ]
        true_indices: List[int] = [0, 1]
        est_indices: List[int] = [0, 1]

    else:
        param_keys = ["m", "l"]
        h_meas = lambda x: np.array([x[0], x[2]])

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

    dt: float = est_cfg["dt"]
    t_end: float = est_cfg["t_end"]
    true_params: Dict[str, float] = est_cfg["true_system_params"]

    true_system: Any = SystemClass(**true_params)
    f_dyn_est: Callable[..., NDArray[Any]] = true_system.get_parameter_estimation_func()

    Q: NDArray[np.float64] = np.diag(est_cfg["Q_init"])
    R: NDArray[np.float64] = np.diag(est_cfg["R"])

    x_est: NDArray[Any] = np.asarray(x0_est)
    ekf = ExtendedKalmanFilter(
        f_dyn_est,
        h_meas,
        Q,
        R,
        x_est,
        p_init_scale=est_cfg["p_init_scale"],
    )

    t_vals: NDArray[np.float64] = np.linspace(0, t_end, int(t_end / dt))

    history: Dict[str, Any] = {
        "t": t_vals,
        "y1_true": [],
        "y1_est": [],
        "y2_true": [],
        "y2_est": [],
        "p1_est": [],
        "p2_est": [],
    }

    amp: float = est_cfg["input_amplitude"]
    period: float = est_cfg["input_period"]
    noise_std: float = est_cfg["sensor_noise_std"]

    x_true: Optional[np.ndarray] = None
    solver_true: Optional[ExactSolver] = None

    if system_id == "pendulum":
        x_true = np.zeros(4)
    else:
        ss_true: StateSpace = true_system.get_state_space()
        solver_true = ExactSolver(ss_true.A, ss_true.B, ss_true.C, ss_true.D, dt=dt)

    for t in t_vals:
        u_val: float = amp if (t % period) < (period / 2.0) else 0.0

        if system_id == "pendulum":
            assert x_true is not None
            x_true = rk4_fixed_step(pendulum_dynamics, x_true, u_val, dt, true_params)
            y_true_full: NDArray[np.float64] = x_true
        else:
            assert solver_true is not None
            y: float | NDArray[np.float64] = solver_true.step(np.array([[u_val], [0]]))
            y_true_full = y if isinstance(y, np.ndarray) else np.asarray(y)

        if isinstance(y_true_full, np.ndarray):
            meas_clean: NDArray[np.float64] = (
                np.array([y_true_full[0], y_true_full[2]])
                if system_id == "pendulum"
                else y_true_full.flatten()
            )

            y_meas: NDArray[np.float64] = meas_clean.reshape(-1, 1) + np.random.normal(
                0, noise_std, (2, 1)
            )

            ekf.predict(np.array([[u_val]]), dt)
            x_hat: NDArray[np.float64] = ekf.update(y_meas)

            history["y1_true"].append(y_true_full[true_indices[0]])
            history["y1_est"].append(x_hat[est_indices[0]])
            history["y2_true"].append(y_true_full[true_indices[1]])
            history["y2_est"].append(x_hat[est_indices[1]])
            history["p1_est"].append(np.exp(x_hat[-2]))
            history["p2_est"].append(np.exp(x_hat[-1]))

    return t_vals, history, true_params, param_keys


def run_ukf_simulation(
    system: Any, system_id: str, cfg: Dict[str, Any]
) -> Tuple[
    NDArray[np.float64], NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]
]:
    f_dyn: Callable[..., NDArray[np.float64]]
    h_meas: Callable[[NDArray[np.float64]], NDArray[np.float64]]
    f_dyn, h_meas = system.get_nonlinear_dynamics()

    dt: float = cfg["dt"]
    x0: NDArray[np.float64] = cfg["x0"]
    P0: NDArray[np.float64] = np.eye(len(x0)) * cfg["P0"]

    Q: NDArray[np.float64] = np.diag(cfg["Q_diag"])
    R: NDArray[np.float64] = np.diag(cfg["R_diag"])

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

    t_vals: NDArray[np.float64] = np.asarray(
        np.arange(0, cfg["t_end"], dt), dtype=float
    )
    true_states: List[NDArray[np.float64]] = []
    est_states: List[NDArray[np.float64]] = []
    measurements: List[NDArray[np.float64]] = []

    curr_x: NDArray[np.float64] = np.array(x0)

    for t in t_vals:
        u: float = 2.0 * np.sin(2.0 * t) if system_id == "dc_motor" else 0.0

        curr_x = f_dyn(curr_x, u, dt)
        true_states.append(curr_x)

        z_clean: NDArray[np.float64] = h_meas(curr_x)
        z_noisy: NDArray[np.float64] = z_clean + np.random.normal(
            0, cfg["noise_std"], size=z_clean.shape
        )
        measurements.append(z_noisy)

        ukf.predict(u, dt)
        est_x: NDArray[np.float64] = ukf.update(z_noisy)
        est_states.append(est_x)

    return (
        t_vals,
        np.array(true_states),
        np.array(est_states),
        np.array(measurements),
    )


def run_mpc_simulation(
    system: Any, system_id: str, cfg: Dict[str, Any]
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    NDArray[np.float64],
    int,
]:
    dt: float = cfg["dt"]

    A_d: Optional[NDArray[np.float64]]
    B_d: Optional[NDArray[np.float64]]
    model_func: Optional[Callable[..., NDArray[np.float64]]]

    if system_id == "dc_motor":
        A_d, B_d = system.get_mpc_model(dt)
        model_func = None
        x0: NDArray[np.float64] = np.array([0.0, 0.0])
        ref: NDArray[np.float64] = np.array([cfg["target_speed"], 0.0])
        plot_idx: int = 0
    else:
        model_func = system.get_mpc_model(dt)
        A_d, B_d = None, None
        x0 = np.array([0.0, 0.0, cfg["start_theta"], 0.0])
        ref = np.zeros(4)
        plot_idx = 2

    Q: NDArray[np.float64] = np.diag(cfg["Q_diag"])
    R: NDArray[np.float64] = np.diag(cfg["R_diag"])

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

    t_vals: NDArray[np.float64] = np.asarray(
        np.arange(0, dt * cfg["horizon"] * 3, dt), dtype=float
    )
    x_hist: List[NDArray[np.float64]] = []
    u_hist: List[float] = []
    curr_x: NDArray[np.float64] = x0.copy()

    mpc_stride: int = config.MPC_SOLVER_PARAMS["mpc_stride"]

    for i in range(len(t_vals)):
        u_opt: NDArray[np.float64] = mpc.optimize(
            curr_x, ref, iterations=cfg["iterations"] if (i % mpc_stride == 0) else 0
        )
        x_hist.append(curr_x)
        u_hist.append(u_opt[0])

        if mpc.mode == "linear" and A_d is not None and B_d is not None:
            curr_x = A_d @ curr_x + B_d @ u_opt
        elif model_func is not None:
            curr_x = model_func(curr_x, u_opt, dt)

    return t_vals, np.array(x_hist), np.array(u_hist), ref, plot_idx


def run_custom_nonlinear_simulation(
    eqn: str, sim_cfg: Dict[str, Any]
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    dyn: AnyFunc = make_system_func(eqn)

    if not is_scalar_func(dyn):
        raise RuntimeError("Vector-valued dynamics not supported")

    dyn_func: ScalarFunc = dyn
    x0: NDArray[np.float64] = np.zeros(sim_cfg["initial_state"]).flatten()

    solver = NonlinearSolver(
        dynamics_func=dyn_func,
        dt_min=sim_cfg["dt_min"],
        dt_max=sim_cfg["dt_max"],
    )

    step_time: float = sim_cfg["step_time"]

    def input_signal(t: float) -> float:
        return sim_cfg["step_magnitude"] if t > step_time else 0.0

    t_vals: NDArray[np.float64]
    states: NDArray[np.float64]

    t_vals, states = solver.solve_adaptive(
        t_end=sim_cfg["t_end"], x0=x0, u_func=input_signal
    )

    y_vals: NDArray[np.float64] = states[:, 0] if states.ndim > 1 else states

    return t_vals, y_vals
