# Estimation-Centered Control of a PWM-DAC Voltage Source Using Reduced-Order LQG Principles

**Swastik Kumar Rout**  
Department of Electrical Engineering  
Indian Institute of Technology, Kharagpur  

---

## Abstract

This work presents the design, implementation, and experimental validation of an estimation-centered closed-loop control architecture for a programmable DC voltage source realized using Arduino-generated Pulse Width Modulation (PWM) and a discrete MOSFET output stage. Unlike classical control problems dominated by plant dynamics, the investigated system exhibits negligible intrinsic dynamics and is instead limited by measurement noise arising from PWM ripple and finite-resolution analog-to-digital conversion. The control problem is therefore reformulated as one of state estimation rather than stabilization. A discrete Kalman filter is employed to recover the true DC voltage from noisy measurements, followed by a Proportional–Integral (PI) control law acting on the estimated state. Experimental results demonstrate rapid convergence, strong bias rejection, and near-optimal performance consistent with an implicit infinite-horizon quadratic cost. The resulting architecture is shown to be equivalent to a reduced-order Linear Quadratic Gaussian (LQG) controller, with the optimal state-feedback law collapsing naturally to PI form.

---

## Index Terms

PWM-DAC, Kalman filtering, PI control, estimation-centered control, reduced-order LQG, embedded systems, noise-dominated systems.

---

## I. Introduction

Low-cost embedded platforms frequently employ PWM-based digital-to-analog conversion to synthesize DC voltage levels. While computationally efficient, PWM-DAC architectures inherently introduce high-frequency ripple and quantization noise, complicating closed-loop regulation. Classical PID control strategies, when applied directly to raw measurements, often amplify noise and exhibit poor robustness under latency and timing jitter.

This project investigates a hardware-in-the-loop (HIL) PWM voltage control system in which physical plant dynamics are negligible compared to stochastic measurement disturbances. It is shown that reframing the control problem around optimal state estimation fundamentally alters both controller structure and tuning priorities. Particular emphasis is placed on *experimental iteration*, with multiple controlled trials used to progressively refine both the system model and controller parameters.

---

## II. System Architecture

### A. Hardware Platform

The physical system consists of an Arduino Uno R3 generating a PWM signal that drives a discrete N-channel MOSFET configured as a low-side switch. The effective output voltage is obtained through resistive averaging of the PWM waveform. Voltage sensing is performed using the Arduino’s 10-bit successive-approximation ADC. The system is powered via a USB supply subject to load-induced voltage sag and environmental noise.

### B. Software Architecture

The control loop is implemented in a HIL configuration. Firmware running on the microcontroller performs high-rate PWM generation and ADC sampling with local averaging. Measurements are transmitted to a host computer, where estimation and control algorithms are executed in Python. This architecture introduces non-negligible communication latency and timing jitter, further motivating estimator-centric design.

---

## III. Experimental Methodology and Iterative Trials

A central contribution of this work is a sequence of **six controlled experimental trials**, each modifying a single modeling or control parameter while holding others fixed. Each trial is documented with its configuration, observed behavior, diagnostic analysis, and corrective action. This methodology follows standard system identification and controller co-design practice and forms the empirical backbone of the project.

### Phase 1: Model Mismatch Failure (Figure 1)

**Configuration:** PID gains $(-75, -20, -2)$ with assumed time constant $\tau = 0.05\text{s}$.

**Observation:** The system failed to track the reference. Control effort saturated at zero while the output voltage remained near 5 V.

**Forensics:** The innovation signal (measurement minus prediction) exhibited persistent divergence. The estimator expected an immediate voltage drop when PWM was applied, consistent with the assumed fast $\tau$. The physical system responded significantly more slowly due to thermal effects and HIL latency. Consequently, valid measurements were rejected as noise, effectively decoupling estimation from control.

**Correction:** Manual system identification was performed, revising the effective time constant to $\tau = 0.5\text{s}$ to account for thermal, firmware, and communication delays.

### Phase 2: Gain-Limited Instability (Figure 2)

**Configuration:** PID gains $(-75, -20, -2)$ with corrected $\tau = 0.5\text{s}$.

**Observation:** The estimator converged, but the output oscillated violently about the setpoint.

**Analysis:** The proportional gain ($K_p = -75$) was excessively aggressive for the HIL latency. The delay between measurement and actuation caused the controller to repeatedly overcorrect, resulting in ringing.

### Phase 3: Transition to Integral Dominance (Figure 3)

**Configuration:** PID gains $(-40, -80, -1.5)$.

**Strategy:** Control authority was shifted from proportional to integral action.

**Theory:** Because the plant behaves as a leaky integrator (Type-1 system), a higher integral gain effectively ramps the control effort through the communication dead time.

**Result:** The system stabilized and reached the 2.5 V setpoint; however, noticeable PWM jitter remained.

### Phase 4: Elimination of Derivative Action (Figure 4)

**Configuration:** PI gains $(-30, -90, 0)$.

**Observation:** Removing derivative action immediately smoothed the control signal.

**Analysis:** In discrete-time systems with quantization noise, derivative terms amplify high-frequency disturbances. Setting $K_d = 0$ reduced the control noise floor and improved steady-state tracking.

### Phase 5: Optimized Operating Point (Golden State) (Figure 5)

**Configuration:** PI gains $(-28, -120, 0)$.

**Performance:**  

* Settling time of approximately $5.3s$ with steady-state voltage constrained to $2.48V – 2.52 V$, with the average of voltage in the last $5s$ (accounting for the initial overshoot) constricted within $+0.05V$ of the setpoint— $2.5V$.  

* Settling time of approximately $1.2s$ with instantenous voltage contained within $+0.05V$ of the setpoint— $2.5V$.

**Justification:** The large ratio $K_i / K_p$ enabled rapid convergence without overshoot, while the Kalman filter provided a low-noise state estimate for feedback.

### Phase 6: Stability Boundary and Limit Testing (Figure 6)

**Configuration:** PI gains $(-28, -125, 0)$.

**Observation:** Further increase in integral gain produced low-frequency limit cycling.

**Conclusion:** The system is delay-limited. Combined serial, execution, and scheduling latency impose a phase-margin ceiling; increasing gain beyond this point drives the phase lag beyond $-180^\circ$ at crossover, inducing instability.

---

## IV. Technical Design Summary — PWM-DAC Voltage Restoration via Estimation-Centered Control

### A. System Overview

This project implements a programmable DC voltage source using Arduino-generated PWM and a MOSFET stage. The electrical output is inherently noisy due to PWM ripple and ADC quantization. No significant physical dynamics exist in the plant; the dominant limitation is measurement noise. Accordingly, the core problem addressed is state estimation rather than plant stabilization.

### B. System Model

The true DC voltage is modeled as a slowly varying scalar state

$
x_{k+1} = x_k + w_k
$  
$
y_k = x_k + v_k
$

where $w_k$ represents process noise due to slow duty-cycle updates and $v_k$ represents measurement noise induced by PWM ripple and quantization. This constitutes a quasi-algebraic stochastic system.

### C. Estimation Strategy

A discrete Kalman filter is employed to estimate the hidden state $x_k$. The filter is tuned with low process noise covariance to strongly reject high-frequency disturbances. Under open-loop conditions, the estimator converges slowly but produces a statistically optimal voltage estimate suitable for control.

### D. Control Objective

The control objective is to drive the estimated voltage to a reference value while minimizing steady-state bias, avoiding noise amplification, and allowing aggressive duty-cycle correction when required.

### E. Implicit Optimal Cost Function

The experimentally optimal closed-loop behavior implicitly minimizes the infinite-horizon quadratic cost

$J = \sum_{k=0}^{\infty} \left[ q_e (x_k - r)^2 + q_z \left( \sum_{i=0}^{k} (x_i - r) \right)^2 + r_u u_k^2 \right]$

with empirically observed weighting hierarchy $q_z \gg q_e \gg r_u$.

### F. Control Law

The resulting control law reduces to a PI controller acting on the estimated state

$u_k = -K_p (x_k - r) - K_i \sum (x_k - r)$

with optimal gains $(K_p, K_i, K_d) = (-28, -120, 0)$. Derivative action is excluded due to noise dominance and lack of a physical derivative state.

### G. Reduced-Order LQG Interpretation

The architecture corresponds to a reduced-order Linear Quadratic Gaussian controller. The Kalman filter minimizes estimation error variance, while the PI controller minimizes the stated quadratic cost. Because the plant is algebraic, the optimal state-feedback law collapses to PI form.

---

## V. Results and Figures

### A. Closed-Loop Response

Figure 5 illustrates the final closed-loop response of the system. The Kalman-filtered voltage estimate converges rapidly to the reference despite substantial raw ADC noise. Control effort increases smoothly and stabilizes without oscillation. The innovation sequence rapidly decays, and the filter covariance converges to a low steady-state value, indicating estimator confidence.

*(Figures 1–6 referenced throughout the text are attached at the end of this report.)*

### B. Block Diagram

Figure shown below depicts the system block diagram. The PWM generator and MOSFET stage form the physical plant. ADC measurements are processed by a Kalman filter, whose estimated state is fed to a PI controller. The controller output updates the PWM duty cycle, closing the loop.

```Markdown
      r
      ↓
    [ PI ] ──► u_k ──► PWM ──► MOSFET ──► ADC ──► y_k
      ▲                                               │
      │                                               ▼
      └──────────── x̂_k ◄──────── Kalman Filter ◄────┘
```

---

## VI. Discussion

The experimental results confirm that the system is fundamentally estimation-limited rather than dynamics-limited. Once the state is accurately estimated, control becomes trivial. Attempts to introduce derivative action or aggressive proportional gains consistently degraded performance due to noise amplification and latency sensitivity.

---

## VII. Conclusions

This work demonstrates that PWM-based voltage control on low-cost embedded platforms is best approached as an estimation problem. Kalman filtering enables recovery of the true DC state, while PI control provides an optimal and structurally appropriate feedback law. The resulting reduced-order LQG architecture achieves near-optimal performance without derivative action. The findings highlight the limitations of classical PID tuning in noise-dominated algebraic systems and emphasize the importance of experimental iteration in controller design.

---

## References

[1] R. E. Kalman, “A new approach to linear filtering and prediction problems,” *Journal of Basic Engineering*, vol. 82, no. 1, pp. 35–45, 1960.

[2] K. J. Åström and R. M. Murray, *Feedback Systems: An Introduction for Scientists and Engineers*, Princeton University Press, 2008.

[3] G. F. Franklin, J. D. Powell, and A. Emami-Naeini, *Feedback Control of Dynamic Systems*, 7th ed., Pearson, 2015.

---

## Images

### Phase 1

![Figure 1](./Figure%201.png)

### Phase 2

![Figure 2](./Figure%202.png)

### Phase 3

![Figure 3](./Figure%203.png)

### Phase 4

![Figure 4](./Figure%204.png)

### Phase 5

![Figure 5](./Figure%205.png)

### Phase 6

![Figure 6](./Figure%206.png)
