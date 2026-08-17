"""
Benchmark 4.1, 4.2, 4.3 — Estimation Performance Metrics.

4.1: EKF Parameter Estimation Convergence — Measures time for estimated
     motor parameters (J, b) to converge within 5% of true values.

4.2: UKF vs EKF Accuracy on Highly Nonlinear Systems — Monte Carlo RMSE
     comparison using DC Motor with stiction/Coulomb friction.

4.3: Kalman Filter Noise Rejection Ratio — Computes SNR improvement (dB)
     of filtered estimates vs raw noisy measurements.

Run from the project root:
    python -m benchmarking.estimation_metrics
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from core.ekf import ExtendedKalmanFilter
from core.ukf import UnscentedKalmanFilter
from helpers.config import (
    DC_MOTOR_DEFAULTS,
    MOTOR_ESTIMATION_PARAMS,
    MOTOR_PARAMS,
    UKF_MOTOR_PARAMS,
)
from systems.dc_motor import DCMotor


def benchmark_ekf_convergence():
    """
    Simulates the DC Motor with known true parameters, runs the joint
    state/parameter EKF, and measures convergence time.
    """
    motor = DCMotor(**MOTOR_ESTIMATION_PARAMS["true_system_params"])
    f_dynamics = motor.get_parameter_estimation_func()

    def h_meas(x):
        return np.array([x[0], x[1]])

    J_true = MOTOR_ESTIMATION_PARAMS["true_system_params"]["J"]
    b_true = MOTOR_ESTIMATION_PARAMS["true_system_params"]["b"]

    J_guess = MOTOR_ESTIMATION_PARAMS["initial_guess_J"]
    b_guess = MOTOR_ESTIMATION_PARAMS["initial_guess_b"]

    dt = MOTOR_ESTIMATION_PARAMS["dt"]
    t_end = MOTOR_ESTIMATION_PARAMS["t_end"]
    n_steps = int(t_end / dt)
    noise_std = MOTOR_ESTIMATION_PARAMS["sensor_noise_std"]

    x0_ekf = np.array([0.0, 0.0, np.log(J_guess), np.log(b_guess)])

    ekf = ExtendedKalmanFilter(
        f_dynamics=f_dynamics,
        h_measurement=h_meas,
        Q=np.diag(MOTOR_ESTIMATION_PARAMS["Q_init"]),
        R=np.diag(MOTOR_ESTIMATION_PARAMS["R"]),
        x0=x0_ekf,
        p_init_scale=MOTOR_ESTIMATION_PARAMS["p_init_scale"],
    )

    K = MOTOR_ESTIMATION_PARAMS["true_system_params"]["K"]
    R_motor = MOTOR_ESTIMATION_PARAMS["true_system_params"]["R"]
    L = MOTOR_ESTIMATION_PARAMS["true_system_params"]["L"]

    x_true = np.array([0.0, 0.0])
    amplitude = MOTOR_ESTIMATION_PARAMS["input_amplitude"]
    period = MOTOR_ESTIMATION_PARAMS["input_period"]

    J_history = []
    b_history = []
    convergence_time_J = None
    convergence_time_b = None

    for k in range(n_steps):
        t = k * dt

        voltage = amplitude * np.sin(2 * np.pi * t / period)

        omega, current = x_true
        dw = (-b_true / J_true) * omega + (K / J_true) * current
        di = (-K / L) * omega - (R_motor / L) * current + (1.0 / L) * voltage
        x_true = x_true + np.array([dw, di]) * dt

        y_meas = x_true + np.random.randn(2) * noise_std
        y_meas = y_meas.reshape(-1, 1)

        ekf.predict(voltage, dt)
        ekf.update(y_meas)

        state = ekf.get_state()
        J_est = np.exp(state[2])
        b_est = np.exp(state[3])
        J_history.append(J_est)
        b_history.append(b_est)

        if convergence_time_J is None and abs(J_est - J_true) / J_true < 0.05:
            convergence_time_J = t
        elif convergence_time_J is not None and abs(J_est - J_true) / J_true >= 0.05:
            convergence_time_J = None

        if convergence_time_b is None and abs(b_est - b_true) / b_true < 0.05:
            convergence_time_b = t
        elif convergence_time_b is not None and abs(b_est - b_true) / b_true >= 0.05:
            convergence_time_b = None

    J_final = J_history[-1]
    b_final = b_history[-1]
    J_err_pct = abs(J_final - J_true) / J_true * 100
    b_err_pct = abs(b_final - b_true) / b_true * 100

    print("  --- 4.1: EKF Parameter Estimation Convergence ---")
    print(f"    True J={J_true}, True b={b_true}")
    print(f"    Initial guess J={J_guess}, b={b_guess}")
    print(f"    Final estimate J={J_final:.6f} (err={J_err_pct:.2f}%), b={b_final:.6f} (err={b_err_pct:.2f}%)")
    if convergence_time_J is not None:
        print(f"    J converged to 5% band at t={convergence_time_J:.3f}s")
    else:
        print(f"    J did NOT converge to 5% band within {t_end}s")
    if convergence_time_b is not None:
        print(f"    b converged to 5% band at t={convergence_time_b:.3f}s")
    else:
        print(f"    b did NOT converge to 5% band within {t_end}s")
    print()


def benchmark_ukf_vs_ekf():
    """
    Runs Monte Carlo simulations comparing EKF and UKF RMSE
    on the DC Motor with stiction/Coulomb friction nonlinearities.
    """
    motor = DCMotor(**DC_MOTOR_DEFAULTS)
    f_stiction, h_meas_ukf = motor.get_nonlinear_dynamics()

    n_monte_carlo = 100
    dt = UKF_MOTOR_PARAMS["dt"]
    t_end = UKF_MOTOR_PARAMS["t_end"]
    n_steps = int(t_end / dt)
    noise_std = UKF_MOTOR_PARAMS["noise_std"]

    Q_diag = UKF_MOTOR_PARAMS["Q_diag"]
    R_diag = UKF_MOTOR_PARAMS["R_diag"]

    ekf_rmses = []
    ukf_rmses = []

    K = DC_MOTOR_DEFAULTS["K"]
    R_motor = DC_MOTOR_DEFAULTS["R"]
    L = DC_MOTOR_DEFAULTS["L"]
    J = DC_MOTOR_DEFAULTS["J"]
    b = DC_MOTOR_DEFAULTS["b"]

    def ekf_dynamics(x, u):
        """Continuous-time dynamics for EKF (linear, no stiction)."""
        omega = x[0]
        current = x[1]
        voltage = float(u) if np.isscalar(u) else float(u.flat[0])
        dw = (-b / J) * omega + (K / J) * current
        di = (-K / L) * omega - (R_motor / L) * current + (1.0 / L) * voltage
        return np.array([dw, di])

    def ekf_h_meas(x):
        return np.array([x[0], x[1]])

    for mc in range(n_monte_carlo):
        x_true = np.array([0.0, 0.0])
        x0_init = np.array([0.0, 0.0])

        ukf = UnscentedKalmanFilter(
            f_dynamics=f_stiction,
            h_measurement=h_meas_ukf,
            Q=np.diag(Q_diag),
            R=np.diag(R_diag),
            x0=x0_init.copy(),
            P0=np.eye(2) * UKF_MOTOR_PARAMS["P0"],
            alpha=UKF_MOTOR_PARAMS["alpha"],
            beta=UKF_MOTOR_PARAMS["beta"],
            kappa=UKF_MOTOR_PARAMS["kappa"],
        )

        ekf = ExtendedKalmanFilter(
            f_dynamics=ekf_dynamics,
            h_measurement=ekf_h_meas,
            Q=np.diag(Q_diag),
            R=np.diag(R_diag),
            x0=x0_init.copy(),
            p_init_scale=UKF_MOTOR_PARAMS["P0"],
        )

        ekf_errors = []
        ukf_errors = []

        for k in range(n_steps):
            t = k * dt
            voltage = 5.0 * np.sin(2 * np.pi * t / 1.0)

            x_true = f_stiction(x_true, voltage, dt)

            y_meas = x_true + np.random.randn(2) * noise_std

            ukf.predict(voltage, dt)
            x_ukf = ukf.update(y_meas)
            ukf_errors.append(np.linalg.norm(x_true - x_ukf))

            ekf.predict(voltage, dt)
            x_ekf = ekf.update(y_meas.reshape(-1, 1))
            ekf_errors.append(np.linalg.norm(x_true - x_ekf[:2]))

        ekf_rmse = np.sqrt(np.mean(np.array(ekf_errors) ** 2))
        ukf_rmse = np.sqrt(np.mean(np.array(ukf_errors) ** 2))
        ekf_rmses.append(ekf_rmse)
        ukf_rmses.append(ukf_rmse)

    ekf_mean = np.mean(ekf_rmses)
    ukf_mean = np.mean(ukf_rmses)
    improvement = ((ekf_mean - ukf_mean) / ekf_mean) * 100 if ekf_mean > 0 else 0

    print("  --- 4.2: UKF vs EKF Accuracy (Stiction Nonlinearity) ---")
    print(f"    Monte Carlo runs:      {n_monte_carlo}")
    print(f"    Simulation:            {t_end}s at dt={dt}s")
    print(f"    EKF Mean RMSE:         {ekf_mean:.6f}")
    print(f"    UKF Mean RMSE:         {ukf_mean:.6f}")
    if improvement > 0:
        print(f"    UKF improvement:       {improvement:.1f}% lower RMSE")
    else:
        print(f"    EKF improvement:       {-improvement:.1f}% lower RMSE")
    print()


def benchmark_noise_rejection():
    """
    Computes SNR of raw sensor data vs filtered estimate to quantify
    noise attenuation in dB.
    """
    motor = DCMotor(**MOTOR_PARAMS)
    f_stiction, h_meas_ukf = motor.get_nonlinear_dynamics()

    dt = UKF_MOTOR_PARAMS["dt"]
    t_end = UKF_MOTOR_PARAMS["t_end"]
    n_steps = int(t_end / dt)
    noise_std = 0.2

    Q_diag = [1e-4, 1e-4]
    R_diag = [noise_std**2, noise_std**2]

    ukf = UnscentedKalmanFilter(
        f_dynamics=f_stiction,
        h_measurement=h_meas_ukf,
        Q=np.diag(Q_diag),
        R=np.diag(R_diag),
        x0=np.array([0.0, 0.0]),
        P0=np.eye(2) * 0.1,
        alpha=1e-3,
        beta=2.0,
        kappa=0.0,
    )

    true_signals = []
    noisy_signals = []
    filtered_signals = []

    x_true = np.array([0.0, 0.0])

    for k in range(n_steps):
        t = k * dt
        voltage = 5.0 * np.sin(2 * np.pi * t / 2.0)

        x_true = f_stiction(x_true, voltage, dt)
        true_signals.append(x_true[0])

        y_noisy = x_true + np.random.randn(2) * noise_std
        noisy_signals.append(y_noisy[0])

        ukf.predict(voltage, dt)
        x_filt = ukf.update(y_noisy)
        filtered_signals.append(x_filt[0])

    true_signals = np.array(true_signals)
    noisy_signals = np.array(noisy_signals)
    filtered_signals = np.array(filtered_signals)

    signal_power = np.mean(true_signals**2)

    noise_raw = noisy_signals - true_signals
    noise_filtered = filtered_signals - true_signals

    noise_power_raw = np.mean(noise_raw**2)
    noise_power_filtered = np.mean(noise_filtered**2)

    if signal_power < 1e-20:
        snr_raw = float("-inf")
        snr_filtered = float("-inf")
        attenuation = 0.0
    else:
        snr_raw = 10 * np.log10(signal_power / (noise_power_raw + 1e-30))
        snr_filtered = 10 * np.log10(signal_power / (noise_power_filtered + 1e-30))
        attenuation = snr_filtered - snr_raw

    print("  --- 4.3: Kalman Filter Noise Rejection Ratio ---")
    print(f"    Measurement noise σ:   {noise_std}")
    print(f"    Raw SNR:               {snr_raw:.1f} dB")
    print(f"    Filtered SNR:          {snr_filtered:.1f} dB")
    print(f"    Noise attenuation:     {attenuation:.1f} dB")
    if attenuation > 10:
        print(f"    ✓ Substantial noise attenuation achieved (>{10}dB).")
    print()


def benchmark_estimation():
    """Runs all estimation benchmarks."""
    print("=" * 60)
    print("Benchmarks 4.1–4.3: Estimation Performance Metrics")
    print("=" * 60)
    print()
    benchmark_ekf_convergence()
    benchmark_ukf_vs_ekf()
    benchmark_noise_rejection()


if __name__ == "__main__":
    benchmark_estimation()
