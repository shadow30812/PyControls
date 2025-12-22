"""
Centralized plotting utilities for PyControls.
All visualization logic lives here to keep the application core headless.
"""

import matplotlib.pyplot as plt
import numpy as np


def plot_time_response(
    t,
    y_real,
    x_est,
    u_hist,
    labels,
    indices,
    controllers,
    system_id,
    dist_params,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax1, ax2 = axes[0, 0], axes[0, 1]
    ax3, ax4 = axes[1, 0], axes[1, 1]

    for ctrl in controllers:
        name = ctrl["name"]
        color = ctrl["color"]

        ax1.plot(t, y_real[name][:, indices[0]], label=name, color=color)
        ax2.plot(t, y_real[name][:, indices[1]], label=name, color=color)
        ax3.plot(t, u_hist[name], label=name, color=color, linestyle="--")

        if system_id == "pendulum":
            ax4.plot(t, y_real[name][:, 3], label=name, color=color)
        else:
            if name in x_est and len(x_est[name]) > 0:
                ax4.plot(t, x_est[name][:, -1], label=name, color=color)

    ax1.set_title(labels[0])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)

    ax2.set_title(labels[1])
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)

    ax3.set_title(labels[2])
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=8)

    ax4.set_title(labels[3])
    ax4.grid(True, alpha=0.3)
    ax4.legend(fontsize=8)

    if system_id == "dc_motor" and dist_params.get("enabled"):
        ax4.axhline(
            dist_params.get("magnitude"),
            color="k",
            linestyle=":",
            label="True Load",
        )
        ax4.legend(fontsize=8)

    plt.tight_layout()
    plt.show()


def plot_analysis_dashboard(
    ss,
    w,
    mags,
    phases,
    t,
    y_real,
    x_est,
    system_id,
):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    ax_bode, ax_pz = axes[0, 0], axes[0, 1]
    ax_kalman, ax_phase = axes[1, 0], axes[1, 1]

    ax_bode.semilogx(w, mags, "k-", lw=2)
    ax_bode.set_title("Bode Magnitude (Input -> Primary State)")
    ax_bode.set_xlabel("Frequency (rad/s)")
    ax_bode.set_ylabel("Magnitude (dB)")
    ax_bode.grid(True, which="both", alpha=0.3)

    eigenvalues = np.linalg.eigvals(ss.A)
    ax_pz.scatter(
        eigenvalues.real,
        eigenvalues.imag,
        marker="x",
        color="r",
        s=100,
        label="Poles",
    )
    ax_pz.axhline(0, color="k", lw=1)
    ax_pz.axvline(0, color="k", lw=1)
    ax_pz.set_title("Pole-Zero Map (S-Plane)")
    ax_pz.set_xlabel("Real")
    ax_pz.set_ylabel("Imaginary")
    ax_pz.grid(True, alpha=0.3)
    ax_pz.legend()

    out_idx = 2 if system_id == "pendulum" else 0

    if len(x_est) > 0:
        err = y_real[:, out_idx] - x_est[:, out_idx]
        ax_kalman.plot(t, err, "b")
        ax_kalman.set_title("Kalman Estimation Error")
        ax_kalman.set_xlabel("Time (sec)")
        ax_kalman.set_ylabel("Error")
        ax_kalman.grid(True, alpha=0.3)
    else:
        ax_kalman.text(0.5, 0.5, "No Estimator Data", ha="center")

    if len(y_real) > 0:
        if system_id == "pendulum":
            ax_phase.plot(y_real[:, 2], y_real[:, 3], "g")
            ax_phase.set_xlabel("Angle (rad)")
            ax_phase.set_ylabel("Angular Velocity (rad/s)")
        else:
            ax_phase.plot(y_real[:, 0], y_real[:, 1], "purple")
            ax_phase.set_xlabel("Speed (rad/s)")
            ax_phase.set_ylabel("Current (A)")

    ax_phase.set_title("Phase Plane Trajectory")
    ax_phase.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def plot_estimation_history(t, history, labels, true_params, param_names):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    def _safe_1d(signal):
        try:
            arr = np.asarray(signal).squeeze()
        except Exception as e:
            print(e)
            return np.zeros_like(t)

        if arr.ndim == 0:
            return np.full_like(t, float(arr), dtype=float)

        if arr.ndim == 1:
            if len(arr) == len(t):
                return arr
            if len(arr) == 1:
                return np.full_like(t, float(arr[0]), dtype=float)

        return np.zeros_like(t)

    y1_true = _safe_1d(history.get("y1_true"))
    y1_est = _safe_1d(history.get("y1_est"))
    y2_true = _safe_1d(history.get("y2_true"))
    y2_est = _safe_1d(history.get("y2_est"))

    axes[0, 0].plot(t, y1_true, "k-", label="True")
    axes[0, 0].plot(t, y1_est, "r--", label="Est")
    axes[0, 0].set_title(f"State Tracking: {labels[0]}")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    axes[0, 1].plot(t, y2_true, "k-", label="True")
    axes[0, 1].plot(t, y2_est, "m--", label="Est")
    axes[0, 1].set_title(f"State Tracking: {labels[1]}")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    axes[1, 0].plot(t, history["p1_est"], "b-", label="Estimate")
    axes[1, 0].axhline(
        true_params[0], color="k", linestyle=":", label=f"True ({true_params[0]})"
    )
    axes[1, 0].set_title(f"Estimation: {param_names[0]}")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    axes[1, 1].plot(t, history["p2_est"], "g-", label="Estimate")
    axes[1, 1].axhline(
        true_params[1], color="k", linestyle=":", label=f"True ({true_params[1]})"
    )
    axes[1, 1].set_title(f"Estimation: {param_names[1]}")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_ukf_estimation(t_vals, true_states, est_states, measurements, labels):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t_vals, true_states[:, 0], "k-", label="True State")
    plt.plot(
        t_vals,
        measurements[:, 0],
        "g.",
        alpha=0.3,
        label="Noisy Measure",
    )
    plt.plot(
        t_vals,
        est_states[:, 0],
        "r--",
        linewidth=2,
        label="UKF Estimate",
    )
    plt.ylabel(labels[0])
    plt.title(f"UKF Estimation: {labels[0]}")
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t_vals, true_states[:, 1], "k-", label="True State")
    plt.plot(
        t_vals,
        est_states[:, 1],
        "b--",
        linewidth=2,
        label="UKF Estimate",
    )
    plt.ylabel(labels[1])
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_mpc_response(t, x_hist, u_hist, ref, labels, plot_idx, system_id, cfg):
    plt.figure(figsize=(10, 8))

    plt.subplot(2, 1, 1)
    plt.plot(t, x_hist[:, plot_idx], "b-", linewidth=2, label="System Output")
    plt.axhline(ref[plot_idx], color="k", linestyle="--", label="Target")

    if system_id == "pendulum":
        plt.axhline(0, color="g", linestyle=":", alpha=0.5)
        plt.axhline(np.pi, color="r", linestyle=":", alpha=0.5, label="Down")

    plt.title("MPC Response")
    plt.ylabel(labels[0])
    plt.legend()
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.step(t, u_hist, "r-", where="post", label="Control Input")
    plt.axhline(cfg["u_max"], color="k", linestyle="--", label="Limits")
    plt.axhline(cfg["u_min"], color="k", linestyle="--")
    plt.ylabel(labels[1])
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def plot_custom_simulation(t, y, eqn):
    plt.figure()
    plt.plot(t, y, label=f"dx/dt = {eqn}")
    plt.title(f"Adaptive RK45 Simulation: {eqn}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.show()
