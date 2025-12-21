# Technical Report: Hardware-in-the-Loop (HIL) Voltage Control of a PWM Inverter using Stochastic State Estimation

**Author:** Swastik Kumar Rout  
**Date:** December 21, 2025  
**Platform:** Python (PyControls), C++ (Firmware), Arduino Uno R3  
**Subject:** Embedded Control Systems & Power Electronics  

---

## 1. Executive Summary

This project explores the design, implementation, and optimization of a closed-loop digital control system for a voltage-source inverter. Using a custom Hardware-in-the-Loop (HIL) architecture, the system regulates the output voltage of a PWM-driven 2N7000 MOSFET circuit.

The primary engineering challenge was to stabilize a noisy, delay-constrained plant that exhibited significant environmental drift. By implementing a **Discrete-Time Kalman Filter** for state estimation and transitioning from a standard PID to an **Integral-Dominant PI controller**, the system achieved an industrial-grade steady-state precision of **$\pm 0.02\text{V}$** and a settling time of **$5.3\text{s}$**. This report documents the system identification process, stability boundary analysis, and the critical role of software-based estimation in compensating for low-cost hardware limitations.

---

## 2. Project Objective & Scope

The objective was to create a robust voltage controller capable of tracking a setpoint (2.5V) on a noisy power rail without using physical filter capacitors.

* **Scope:** Design of the physical plant, firmware development for the Arduino ADC/PWM bridge, and Python-based implementation of the Control Law and Estimator.
* **Constraint:** The controller must operate in real-time, handling the latency introduced by Serial communication and host-side processing (Python Global Interpreter Lock and OS scheduling).

---

## 3. Theoretical Framework

### 3.1 Plant Dynamics (The "Inverter" Model)

The physical plant is a common-source amplifier configuration. It acts as an inverting voltage source:

* **Logic:** $V_{out} \propto (1 - D)$, where $D$ is the PWM Duty Cycle.
* **Gain ($G$):** Negative. Increasing the control effort ($u$) decreases the system state ($x$).
* **Dynamics:** The system acts as a first-order lag process due to the thermal mass of the MOSFET and the electrical averaging of the firmware.
    $$\tau \dot{x} + x = G u$$

### 3.2 State Estimation (The Kalman Filter)

To reject ADC quantization noise and PWM ripple, a linear Kalman Filter was implemented. The continuous-time model was discretized using the matrix exponential:

* **Prediction:** $\hat{x}_{k|k-1} = \Phi \hat{x}_{k-1} + \Gamma u_{k-1}$
* **Update:** $\hat{x}_{k|k} = \hat{x}_{k|k-1} + K_k (y_k - C\hat{x}_{k|k-1})$

Where $K_k$ is the optimal Kalman Gain computed by minimizing the error covariance $P$. This allows the controller to act on the *estimated true voltage* rather than the noisy raw sensor data.

---

## 4. System Architecture

### 4.1 Hardware Layer

* **Microcontroller:** Arduino Uno R3 (ATmega328P).
* **Actuator:** 2N7000 N-Channel MOSFET.
  * *Configuration:* Low-side switch with Pull-up resistor network.
* **Sensing:** 10-bit Successive Approximation ADC (Arduino Pin A0).
* **Power:** 5V USB Bus (Subject to voltage sag during high CPU load).

### 4.2 Software Layer (HIL Architecture)

* **Firmware (C++):** Acts as a high-speed dumb terminal. It performs 50-sample averaging on the ADC to act as a software low-pass filter before transmitting data.
* **Control Loop (Python):**  
  * **PyControls Library:** Handles matrix operations for the Kalman Filter using Numba-optimized solvers.
    * **Physics Engine:** Calculates the PID response based on the estimated state.
    * **Visualization:** Renders real-time telemetry (State, Control Effort, Innovation, Covariance).

---

## 5. Experimental Methodology & Tuning Evolution

The development process followed a rigorous **Six-Stage Optimization Path**, documented below.

### Phase 1: The "Model Mismatch" Failure

* **Configuration:** PID Gains $(-75, -20, -2)$, Assumed $\tau = 0.05\text{s}$.
* **Observation:** The system failed to track. The control effort saturated at 0, while the voltage remained stuck at 5V.
* **Forensics:** The "Innovation" plot (Measurement minus Prediction) showed a persistent divergence. The State Estimator expected the voltage to drop instantly (fast $\tau$) when PWM was applied. The physical system, being slower, did not respond in time. The filter rejected the measurement as "sensor error," decoupling the control loop.
* **Correction:** Manual System Identification was performed, revising the time constant to $\tau = 0.5\text{s}$ to account for thermal and communication lags.

### Phase 2: The "Gain-Limited" Instability

* **Configuration:** PID Gains $(-75, -20, -2)$, Corrected $\tau = 0.5\text{s}$.
* **Observation:** The estimator converged, but the output oscillated violently around the setpoint.
* **Analysis:** The Proportional gain ($K_p = -75$) was too aggressive for the HIL latency. The delay between measuring $V$ and applying PWM caused the controller to "chase" the error, leading to ringing.

### Phase 3: Transition to Integral Dominance

* **Configuration:** PID Gains $(-40, -80, -1.5)$.
* **Strategy:** Shifted the control burden from Proportional to Integral.
* **Theory:** Since the plant behaves like a leaky integrator (Type-1), a high Integral gain ($K_i$) acts as a velocity ramp, driving the system through the "dead time" of the serial communication.
* **Result:** The system stabilized, reaching the 2.5V setpoint. However, significant "jitter" remained in the PWM signal.

### Phase 4: Elimination of Derivative Action

* **Configuration:** PI Gains $(-30, -90, 0)$.
* **Observation:** Removing $K_d$ immediately smoothed the control effort.
* **Analysis:** In discrete-time systems with quantization noise (10-bit ADC), the Derivative term ($\frac{e_t - e_{t-1}}{dt}$) amplifies high-frequency noise. By setting $K_d=0$, the noise floor of the control loop was lowered, improving steady-state tracking.

### Phase 5: The "Golden State" (Optimization)

* **Configuration:** PI Gains **$(-28, -120, 0)$**.
* **Performance:**
  * **Settling Time:** 5.3 seconds.
  * **Steady State Envelope:** 2.48V to 2.52V.
* **Justification:** This tuning represents the global optimum. The high ratio of $K_i/K_p$ allows for fast convergence without overshoot, while the Kalman Filter provides a noise-free estimate for the PI law to act upon.

### Phase 6: The Stability Boundary (Limit Testing)

* **Configuration:** PI Gains $(-28, -125, 0)$.
* **Observation:** Increasing $K_i$ further introduced low-frequency limit cycling.
* **Conclusion:** The system is **Delay-Limited**. The total loop latency (Serial transmission + Python execution + OS scheduling) creates a phase margin ceiling. Increasing gains beyond this point pushes the phase lag over -180 degrees at the crossover frequency, inducing instability.

---

## 6. Robustness & Environmental Analysis

During a 7-hour continuous stress test, the system demonstrated specific vulnerabilities that were mitigated via software:

### 6.1 Thermal Drift Compensation

As the 2N7000 MOSFET heated up, its internal resistance ($R_{DS(on)}$) increased, effectively altering the plant gain $G$. The **Integral-Dominant** controller proved essential here; the accumulator naturally "wound up" to compensate for the changing resistance, maintaining the 2.5V target despite the shifting physics.

### 6.2 Latency Jitter & CPU Throttling

Intense real-time plotting caused the host CPU to throttle, causing the sampling time ($dt$) to fluctuate between 10ms and 50ms.

* **Impact:** A standard PID controller relying on a fixed $dt$ for the Derivative term would have failed (spiking $D$ term).
* **Mitigation:** By removing the Derivative term and relying on the robust Integral action, the controller became insensitive to timing jitter.

---

## 7. Results Summary

| Metric | Initial PID Attempt | Final Optimized PI | Improvement |
| :--- | :--- | :--- | :--- |
| **Control Logic** | Proportional-Dominant | Integral-Dominant | Robustness against latency |
| **Noise Rejection** | Poor (Amplified by $K_d$) | Excellent (Filtered by KF) | Signal-to-Noise Ratio $\uparrow$ |
| **Settling Time** | Unstable / Divergent | **5.3s** | Stable Convergence |
| **Precision** | N/A (Saturated) | **$\pm 0.02\text{V}$** | Industrial Grade |
| **Steady-State Bias** | > 2.0V | **< 0.01V** | Near-Zero Error |

*(See attached Figures 1-6 for the graphical evolution of the system response)*

---

## 8. Engineering Conclusions

1. **PI > PID for Noisy HIL:** In discrete-time systems with significant measurement noise and communication latency, the Derivative term often degrades performance. A well-tuned PI controller is superior.
2. **Estimation is Critical:** The Kalman Filter allowed the controller to act on the "True" physics rather than the "Noisy" ADC, enabling higher gains than would otherwise be stable.
3. **Model Identification:** Theoretical models ($\tau=0.05$) often fail in the real world. Empirical system identification ($\tau=0.5$) is a mandatory step in commissioning.
