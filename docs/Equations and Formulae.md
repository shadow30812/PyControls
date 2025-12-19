# **PyControls Mathematical Reference V2.0**

This document serves as the definitive mathematical registry for the PyControls library. It details the physical first principles, numerical algorithms, control laws, and estimation filters implemented in the codebase, bridging the gap between theoretical control systems and their Python implementation.

## **Table of Contents**

1. [Mathematical Foundations](#1-mathematical-foundations)
   * [Numerical Differentiation](#11-numerical-differentiation)
   * [Root Finding Algorithms](#12-root-finding-algorithms)
   * [Expression Parsing & Evaluation](#13-expression-parsing--evaluation)
2. [Numerical Linear Algebra & Solvers](#2-numerical-linear-algebra--solvers)
   * [Matrix Exponential (Scaling & Squaring)](https://www.google.com/search?q=%2321-matrix-exponential)
   * [Exact Discretization (Zero-Order Hold)](#22-exact-discretization)
   * [Ordinary Differential Equation Solvers](#23-ode-solvers)
3. [Physical System Modeling](#3-physical-system-modeling)
   * [DC Motor Dynamics](#31-dc-motor)
   * [Inverted Pendulum on a Cart](#32-inverted-pendulum)
4. [Control Theory & Implementation](#4-control-theory--implementation)
   * [PID Controller (Robust Implementation)](#41-pid-controller)
   * [Linear Quadratic Regulator (LQR)](#42-linear-quadratic-regulator)
   * [Model Predictive Control (MPC)](#43-model-predictive-control)
5. [State Estimation](#5-state-estimation)
   * [Extended Kalman Filter (EKF)](#51-extended-kalman-filter)
   * [Unscented Kalman Filter (UKF)](#52-unscented-kalman-filter)
6. [System Analysis](#6-system-analysis)
   * [Frequency Response](#61-frequency-response)
   * [Stability Margins](#62-stability-margins)
   * [Time-Domain Metrics](#63-time-domain-metrics)

## **1. Mathematical Foundations**

**Source:** `core/math_utils.py`

### **1.1. Numerical Differentiation**

Differentiation is a critical operation for modern control algorithms, particularly for linearizing non-linear systems (Jacobian calculation) in the Extended Kalman Filter (EKF) and Iterative LQR (iLQR). The library implements two distinct methods to handle this, balancing general compatibility with high-precision requirements.

#### **A. Finite Difference (Real Step)**

This is the standard numerical approach used as a fallback when the target function cannot handle complex number inputs. It approximates the derivative using the slope of the secant line between two points very close to $x$.

$$f'(x) \approx \frac{f(x + h) - f(x - h)}{2h}$$

* **Step Size ($h$):** Defaults to $10^{-6}$.
* **Trade-off:** The choice of $h$ is a delicate balance. If $h$ is too large, the **truncation error** (deviation from the true tangent) dominates ($O(h^2)$). If $h$ is too small, **round-off error** dominates because floating-point subtraction $f(x+h) - f(x-h)$ results in "catastrophic cancellation" of significant digits.

#### **B. Complex Step Differentiation (CSD)**

This is the preferred method for high-precision derivatives. It exploits the Cauchy-Riemann equations of complex variables. By evaluating the function with a complex step $x + ih$, the derivative appears in the imaginary part of the result.

**Taylor Series Expansion:**

$$f(x + ih) = f(x) + ihf'(x) - \frac{h^2}{2}f''(x) - \frac{ih^3}{6}f'''(x) + \dots$$

Taking the imaginary part and dividing by $h$:

$$\frac{\text{Im}(f(x + ih))}{h} = f'(x) - \frac{h^2}{6}f'''(x) + \dots$$

**Formula:**

$$f'(x) \approx \frac{\text{Im}(f(x + ih))}{h}$$

* **Step Size ($h$):** Defaults to $10^{-12}$.
* **Key Advantage:** Unlike finite differences, this method involves **no subtraction** in the numerator. This eliminates subtractive cancellation errors, allowing us to use an extremely small $h$ to minimize truncation error. The result is accurate to near machine precision ($10^{-16}$).

### **1.2. Root Finding Algorithms**

These algorithms are essential for frequency domain analysis, specifically for finding crossover frequencies where the system gain is 0dB (Gain Crossover) or the phase is -180° (Phase Crossover).

#### **A. Newton-Raphson Method**

A fast, iterative method that uses the function's derivative to linearly approximate the root. It projects a tangent line from the current guess $(x_n, f(x_n))$ to the x-axis to find the next guess $x_{n+1}$.

$$x_{n+1} = x_n - \frac{f(x_n)}{f'(x_n)}$$

* **Convergence:** Quadratic (the number of correct significant digits roughly doubles with every iteration) when close to the root.
* **Failure Mode:** It is not guaranteed to converge globally. It can diverge or cycle if the derivative is near zero (flat slope) or if the initial guess is too far from the true root.

#### **B. Brent's Method**

The primary, robust root solver used in `core/analysis.py`. It is a hybrid algorithm designed to be as safe as Bisection but as fast as secant-based methods.

1. **Bisection Method:** The algorithm maintains a "bracket" $[a, b]$ where the function has opposite signs ($f(a) \cdot f(b) < 0$). It guarantees that a root exists within this interval. If other steps fail, it simply splits the interval in half.
2. **Secant Method:** Approximates the derivative using a secant line between two recent points.
3. **Inverse Quadratic Interpolation (IQI):** Fits a parabola $x = P(y)$ through three points $(y_a, a), (y_b, b), (y_c, c)$ to estimate the root ($y=0$). This provides extremely fast convergence when the function behaves smoothly near the root.

**IQI Formula:**

$$x = \frac{f_b f_c}{(f_a - f_b)(f_a - f_c)}a + \frac{f_a f_c}{(f_b - f_a)(f_b - f_c)}b + \frac{f_a f_b}{(f_c - f_a)(f_c - f_b)}c$$

### **1.3. Expression Parsing & Evaluation**

The system allows users to define custom dynamics or math functions as strings. These are compiled into executable Python functions.

* **Implicit Multiplication:** The parser uses RegEx to identify patterns like `3x` or `(a+b)c` and inserts the multiplication operator `*` (e.g., `3*x`), making the syntax more natural for mathematical input.
* **Power Operator:** Caret syntax `^` is converted to Python's `**` operator.
* **Safety:** The `make_func` and `make_system_func` utilities create a restricted execution environment (`safe_locals`). This environment provides access only to standard math libraries (`numpy`, `math`, `cmath`) and blocks access to system-level built-ins to prevent code injection vulnerabilities.

## **2. Numerical Linear Algebra & Solvers**

**Source:** `core/solver.py`

### **2.1. Matrix Exponential**

The calculation of the matrix exponential $e^{At}$ is the cornerstone of solving Linear Time-Invariant (LTI) differential equations of the form $\dot{x} = Ax$.

**Algorithm: Scaling and Squaring**

This method is chosen for its numerical stability compared to the power series definition, which can suffer from severe round-off errors if the matrix norm is large.

1. **Norm Reduction (Scaling):** The matrix $A$ is scaled by a factor of $1/2^s$ such that the infinity norm $||A/2^s||_\infty < 0.5$. This ensures the Taylor series converges very rapidly.
2. **Taylor Approximation:** The exponential of the scaled matrix $E \approx e^{A/2^s}$ is computed using a truncated Taylor series.
   $$E = I + \frac{A}{2^s} + \frac{1}{2!}\left(\frac{A}{2^s}\right)^2 + \dots + \frac{1}{k!}\left(\frac{A}{2^s}\right)^k$$
   *(The implementation typically uses order=10 or higher to ensure precision.)*
3. **Squaring:** The property $e^A = (e^{A/2})^2$ is exploited. The matrix $E$ is squared repeatedly $s$ times to recover the full exponential:
   $$e^A = (E)^{2^s} = \underbrace{E \cdot E \cdot \dots \cdot E}_{s \text{ times}}$$

### **2.2. Exact Discretization**

To simulate continuous physical systems on a digital computer, we must convert the continuous matrices $(A, B)$ into discrete matrices $(A_d, B_d)$. We use the **Zero-Order Hold (ZOH)** assumption, which assumes the control input $u(t)$ remains constant between time steps $k$ and $k+1$.

**Van Loan's Method:**
Instead of computing the convolution integral $\int e^{A\tau}B d\tau$ explicitly, we form a larger block matrix $M$ and compute a single matrix exponential. This is numerically robust and handles singular $A$ matrices correctly.

Construct the $(n+m) \times (n+m)$ block matrix $M$:

$$M = \begin{bmatrix} A & B \\ 0 & 0 \end{bmatrix} \cdot \Delta t$$

Compute the matrix exponential of $M$:

$$e^M = \begin{bmatrix} \Phi & \Gamma \\ 0 & I \end{bmatrix}$$

* **Discrete State Matrix:** $A_d = \Phi$
* **Discrete Input Matrix:** $B_d = \Gamma$
* **Discrete System Model:** $x[k+1] = A_d x[k] + B_d u[k]$

### **2.3. ODE Solvers**

For non-linear systems where linear discretization is impossible, we use numerical integration techniques.

#### **A. Runge-Kutta 4 (RK4) - Fixed Step**

This is the standard workhorse for real-time control loops and simulations where deterministic computation time is required. It estimates the state at the next time step by taking a weighted average of four "slopes" (derivatives) calculated at different points within the interval.

$$k_1 = f(t, x, u) \quad \text{(Slope at start)}$$
$$k_2 = f(t + \frac{dt}{2}, x + \frac{dt}{2}k_1, u) \quad \text{(Slope at midpoint)}$$
$$k_3 = f(t + \frac{dt}{2}, x + \frac{dt}{2}k_2, u) \quad \text{(Refined slope at midpoint)}$$
$$k_4 = f(t + dt, x + dt \cdot k_3, u) \quad \text{(Slope at end)}$$
$$x_{k+1} = x_k + \frac{dt}{6}(k_1 + 2k_2 + 2k_3 + k_4)$$

#### **B. Dormand-Prince (RK45) - Adaptive Step**

Used for high-accuracy offline simulation in the `NonlinearSolver`. It is an embedded method, meaning it simultaneously computes a $4^{th}$ order solution ($x^*$) and a $5^{th}$ order solution ($x$) using the same function evaluations.

**Butcher Tableau (Dormand-Prince):**
The coefficients $c_i, a_{ij}, b_i, \hat{b}_i$ are explicitly defined to minimize error terms.
$$k_i = f(t + c_i h, x + h \sum_{j=1}^{i-1} a_{ij} k_j)$$

* $5^{th}$ Order Estimate: $x_{n+1} = x_n + h \sum b_i k_i$
* $4^{th}$ Order Estimate: $x^*_{n+1} = x_n + h \sum \hat{b}_i k_i$

**Error & Adaptation:**
The difference between the two estimates provides a local error estimate $\epsilon$. The step size $h$ is then adjusted dynamically: if the error is low, the step size increases to save time; if high, it decreases to maintain accuracy.
$$\epsilon = \max(|x_{n+1} - x^*_{n+1}|)$$
$$h_{new} = h \cdot 0.9 \cdot \left( \frac{\text{tol}}{\epsilon} \right)^{0.2}$$

## **3. Physical System Modeling**

**Sources:** `systems/dc_motor.py`, `systems/pendulum.py`

### **3.1. DC Motor**

The DC motor is modeled as a Single-Input Multi-Output (SIMO) system coupling electrical and mechanical domains.

**Fundamental Equations:**

1. **Electrical Domain (Kirchhoff's Voltage Law):**
   The input voltage $V$ overcomes the resistive drop $iR$, the inductive inertia $L\frac{di}{dt}$, and the Back-EMF $e_b$ generated by the spinning rotor.
   $$V - iR - L\frac{di}{dt} - e_b = 0, \quad \text{where } e_b = K_b \omega$$

2. **Mechanical Domain (Newton's 2nd Law):**
   The motor torque $T_m$ (generated by Lorenz forces) overcomes viscous friction $b\omega$, external load $T_{load}$, and rotor inertia $J$.
   $$T_m - b\omega - T_{load} = J\frac{d\omega}{dt}, \quad \text{where } T_m = K_i i$$

**Transfer Functions:**

* **Plant ($V \to \omega$):** This describes how Voltage controls Speed.
  $$ P(s) = \frac{K}{(Ls+R)(Js+b) + K^2}$$
* **Disturbance ($T_{load} \to \omega$):** This describes how external Torque affects Speed.
  $$ G_d(s) = \frac{-(Ls+R)}{(Ls+R)(Js+b) + K^2}$$
  *(The negative sign indicates that a positive load torque opposes the direction of motion)*

**Nonlinear Stiction Model:**
For the UKF demos, a simple linear friction model $b\omega$ is insufficient to capture low-speed behavior. We add a Coulomb friction term (constant force opposing motion).
$$T_{fric} = b\omega + T_{coulomb}\text{sgn}(\omega)$$
**Deadband Logic (Stiction):** If the speed is effectively zero ($|\omega| < 0.1$) and the applied motor torque is less than the static friction limit, the motor remains stuck (acceleration is forced to zero).

### **3.2. Inverted Pendulum**

The equations of motion are derived using Lagrangian Dynamics ($\mathcal{L} = T - V$).

**Generalized Coordinates:** $q = [x, \theta]^T$, where $x$ is cart position and $\theta$ is pendulum angle from vertical.

**Energy:**

* **Kinetic Energy ($T$):** Includes cart translation, pendulum translation, and pendulum rotation.
  $$ T = \frac{1}{2}(M+m)\dot{x}^2 + \frac{1}{2}ml^2\dot{\theta}^2 + ml\dot{x}\dot{\theta}\cos\theta$$
* **Potential Energy ($V$):** Depends only on the height of the pendulum bob.
  $$ V = mgl\cos\theta$$

**Equations of Motion (Nonlinear):**
Let $D = M + m(1 - \cos^2\theta)$.
$$\ddot{\theta} = \frac{(M+m)g\sin\theta - \cos\theta(u + ml\dot{\theta}^2\sin\theta) - \frac{(M+m)b\dot{\theta}}{ml}}{l \cdot D}$$
$$\ddot{x} = \frac{u + ml\dot{\theta}^2\sin\theta - mg\sin\theta\cos\theta}{D}$$

**Linearization (Jacobian):**
Linearized around the unstable equilibrium $x=[0, 0, 0, 0]^T$ (Upright). Small angle approximations ($\sin\theta \approx \theta, \cos\theta \approx 1, \dot{\theta}^2 \approx 0$) yield the state matrix $A$:
$$\frac{\partial \ddot{\theta}}{\partial \theta} \approx \frac{(M+m)g}{Ml}, \quad \frac{\partial \ddot{x}}{\partial \theta} \approx \frac{-mg}{M}$$

**Augmented State Model:**
To estimate unknown external disturbances (like a constant wind or hand pushing the cart), we add a "Disturbance Bias" state $d$. We assume the disturbance is constant or slowly varying ($\dot{d} = 0 + \text{noise}$).
$$x_{aug} = [x, \dot{x}, \theta, \dot{\theta}, d]^T$$
The system matrix $A$ is expanded to $5 \times 5$, allowing the Kalman Filter to estimate the value of $d$ based on position/velocity errors.

## **4. Control Theory & Implementation**

**Sources:** `core/control_utils.py`, `core/mpc.py`

### **4.1. PID Controller**

The implementation includes features for industrial robustness, preventing common issues like "Derivative Kick" and "Integral Windup".

**Derivative on Measurement (Kick Prevention):**
Standard PID differentiators use the error term $e(t) = r(t) - y(t)$. If the setpoint $r(t)$ changes instantly (a step change), the derivative $\frac{de}{dt}$ becomes infinite, causing a violent spike in control output.
To prevent this, we differentiate the measurement $y(t)$ instead (assuming $r(t)$ is constant during the step):
$$D_{raw} = -\frac{y_k - y_{k-1}}{\Delta t}$$

**Low-Pass Filter (LPF):**
Real-world derivatives amplify sensor noise. We apply a first-order Low-Pass Filter to the derivative term to smooth it out.
$$D_{filtered}[k] = \alpha D_{raw} + (1-\alpha) D_{filtered}[k-1]$$
$$\alpha = \frac{\Delta t}{\tau + \Delta t}$$

* $\tau$: Filter time constant. Larger $\tau$ means more smoothing but more lag.

**Anti-Windup:**
If the actuator saturates (reaches $u_{max}$), the integral term keeps accumulating error ("winding up"). When the error sign eventually flips, the controller remains saturated for a long time while the integral unwinds, causing overshoot.

* **Implementation:** The output $u$ is strictly clamped to $[u_{min}, u_{max}]$. Ideally, the integrator should also stop accumulating when the output is clamped.

### **4.2. Linear Quadratic Regulator (LQR)**

LQR provides an optimal control law $u = -Kx$ for linear systems by balancing system performance against control effort.

**Cost Function:**

$$J = \sum_{k=0}^{\infty} (x_k^T Q x_k + u_k^T R u_k)$$

* $Q$ (State Cost): Penalizes deviation from the setpoint. High $Q$ results in aggressive control and fast settling.
* $R$ (Control Cost): Penalizes actuator usage. High $R$ results in smoother, more conservative control.

**Discrete Algebraic Riccati Equation (DARE):**
To find the optimal gain, we solve for the steady-state "Cost-to-Go" matrix $P$ via fixed-point iteration:
$$P_{new} = A^T P A - (A^T P B)(R + B^T P B)^{-1}(B^T P A) + Q$$

* **Convergence:** The iteration continues until $||P_{new} - P|| < 10^{-8}$.
* **Gain Calculation:** Once $P$ converges, the optimal gain is:
  $$ K = (R + B^T P B)^{-1} B^T P A$$

### **4.3. Model Predictive Control (MPC)**

MPC optimizes a trajectory of future control inputs over a finite horizon $N$, applies the first input, and then repeats the process (Receding Horizon Control).

#### **A. Linear MPC (ADMM Algorithm)**

Used when the system is linear ($A, B$). We formulate the problem as a Quadratic Program (QP).

**Condensed Formulation:**
We eliminate the intermediate states $x_1 \dots x_N$ by expressing them as functions of the initial state $x_0$ and the input sequence $U = [u_0, \dots, u_{N-1}]^T$.
$$X = S_u U + S_x x_0$$
The cost function becomes a quadratic form purely in terms of $U$:

$$J(U) = \frac{1}{2}U^T H U + q^T U$$

* $H = S_u^T \bar{Q} S_u + \bar{R}$ (Hessian Matrix)
* $q = S_u^T \bar{Q} (S_x x_0 - x_{ref})$ (Linear term)

**ADMM (Alternating Direction Method of Multipliers):**
We solve this QP by splitting it into three simpler steps:

1. **x-update (Unconstrained Optimization):** Solves the linear system $(H + \rho I)x = \rho(z - u) - q$. Since $H$ is constant, we pre-compute its Cholesky factorization for $O(1)$ solve time.
2. **z-update (Projection):** Handles constraints by simple clipping: $z = \text{clip}(x + u, u_{min}, u_{max})$.
3. **u-update (Dual Update):** Updates the Lagrange multipliers: $u = u + x - z$.

#### **B. Nonlinear MPC (iLQR / DDP)**

Used when the model is a generic non-linear function $x_{k+1} = f(x_k, u_k)$.

**Backward Pass (Value Function Approximation):**
We approximate the Value Function $V(x)$ as a quadratic around the nominal trajectory.
$$Q_x = l_x + f_x^T V_x, \quad Q_u = l_u + f_u^T V_x$$
$$Q_{uu} = l_{uu} + f_u^T V_{xx} f_u, \quad Q_{xx} = l_{xx} + f_x^T V_{xx} f_x$$
$$Q_{ux} = f_u^T V_{xx} f_x$$
From these terms, we compute the optimal changes to control:

* Feedforward Gain: $k = -Q_{uu}^{-1} Q_u$
* Feedback Gain: $K = -Q_{uu}^{-1} Q_{ux}$

**Regularization:**
To ensure $Q_{uu}$ is invertible (positive definite), we add a regularization term: $Q_{uu} \leftarrow Q_{uu} + \mu I$.

## **5. State Estimation**

**Sources:** `core/ekf.py`, `core/ukf.py`

### **5.1. Extended Kalman Filter (EKF)**

The EKF adapts the linear Kalman Filter to non-linear systems by linearizing the dynamics at the current estimate.

**Jacobian Linearization:**
The matrices $F$ and $H$ represent the slope of the non-linear functions $f$ and $h$.
$$F = I + \frac{\partial f}{\partial x}\Delta t, \quad H = \frac{\partial h}{\partial x}$$
These are computed dynamically at every step using Complex Step Differentiation (Section 1.1).

**Algorithm:**

1. **Predict (Time Update):** Propagate the state mean via the non-linear physics, and the covariance via the Jacobian $F$.
   $$x_{k|k-1} = x_{k-1} + \int f(x,u) dt$$
   $$P_{k|k-1} = F P_{k-1} F^T + Q$$
2. **Update (Measurement Update):** Correct the estimate using sensor data $z_k$.
   $$y_{res} = z_k - h(x_{k|k-1}) \quad (\text{Innovation})$$
   $$S = H P H^T + R \quad (\text{Innovation Covariance})$$
   $$K = P H^T S^{-1} \quad (\text{Kalman Gain})$$
   $$x_k = x_{k|k-1} + K y_{res}$$
   $$P_k = (I - KH) P_{k|k-1}$$

### **5.2. Unscented Kalman Filter (UKF)**

The UKF avoids analytical linearization (Jacobians) entirely. Instead, it uses a deterministic sampling approach called the **Unscented Transform** to propagate the mean and covariance through the non-linear functions. This often yields higher accuracy than the EKF, especially for highly non-linear systems.

**Sigma Point Generation:**
For a state vector of dimension $n$, we generate $2n+1$ Sigma Points $\chi$ that capture the statistics of the distribution.
Scaling parameter: $\lambda = \alpha^2(n+\kappa) - n$.
$$\chi_0 = \mu$$
$$\chi_i = \mu + [\sqrt{(n+\lambda)P}]_i \quad \text{for } i=1\dots n$$
$$\chi_{i+n} = \mu - [\sqrt{(n+\lambda)P}]_i \quad \text{for } i=1\dots n$$
*(The matrix square root is computed via Cholesky Decomposition)*

**Weights:**
The points are weighted to reconstruct the mean and covariance accurately.

* Mean Weight (Center): $W_m^{(0)} = \frac{\lambda}{n+\lambda}$
* Covariance Weight (Center): $W_c^{(0)} = \frac{\lambda}{n+\lambda} + (1 - \alpha^2 + \beta)$
* Other Weights: $W_m^{(i)} = W_c^{(i)} = \frac{1}{2(n+\lambda)}$

**Process:**

1. **Transform:** Pass every Sigma Point through the non-linear dynamics function: $\mathcal{Y}_i = f(\chi_i)$.
2. **Reconstruct:** Calculate the new mean and covariance from the transformed points.
   $$x = \sum_{i=0}^{2n} W_m^{(i)} \mathcal{Y}_i$$
   $$P = \sum_{i=0}^{2n} W_c^{(i)} (\mathcal{Y}_i - x)(\mathcal{Y}_i - x)^T + Q$$

## **6. System Analysis**

**Sources:** `core/analysis.py`, `core/state_space.py`

### **6.1. Frequency Response**

The frequency response $H(j\omega)$ describes how the system amplifies and phase-shifts sinusoidal inputs at a specific frequency $\omega$. It is calculated directly from State-Space matrices to avoid the numerical instability associated with converting to transfer function polynomials (roots of high-degree polynomials are sensitive to errors).

$$H(j\omega) = C(j\omega I - A)^{-1}B + D$$

**Implementation Detail:**
Instead of explicitly inverting the complex matrix $(j\omega I - A)$, which is computationally expensive ($O(n^3)$) and potentially unstable, we solve the linear system $(j\omega I - A)x = B$ for $x$. Then, the response is $Cx + D$.

### **6.2. Stability Margins**

These metrics quantify the robustness of a feedback loop.

* **Gain Margin (GM):** How much the system gain can increase before the closed loop becomes unstable. It is measured at the **Phase Crossover Frequency** ($\omega_{pc}$) where the phase lag reaches -180°.
  $$ \text{GM}_{dB} = -20\log_{10}(|H(j\omega_{pc})|)$$
  *(Positive dB indicates a stable system margin)*
* **Phase Margin (PM):** How much additional phase lag (time delay) the system can tolerate before instability. It is measured at the **Gain Crossover Frequency** ($\omega_{gc}$) where the magnitude gain is 1 (0 dB).
  $$ \text{PM} = 180^\circ + \angle H(j\omega_{gc})$$

### **6.3. Time-Domain Metrics**

These metrics are extracted from step response simulation data $(t_k, y_k)$.

**Linear Interpolation for Precision:**
Since simulation data is discrete, the "exact" time a threshold (e.g., 90% rise) is crossed usually lies between two time steps $t_k$ and $t_{k+1}$. We use linear interpolation to improve metric accuracy:
$$t_{target} = t_1 + (t_2 - t_1) \frac{y_{target} - y_1}{y_2 - y_1}$$

* **Rise Time ($t_r$):** The time taken for the response to rise from 10% to 90% of its final steady-state value.
  $$t_r = t_{90\%} - t_{10\%}$$
* **Settling Time ($t_s$):** The time after which the response enters and remains within a specified error band (typically $\pm 2\%$) of the final value.
  $$|y(t) - y_{final}| \le 0.02 \cdot y_{final} \quad \forall t > t_s$$
* **Overshoot (%):** The maximum peak value relative to the final steady-state value.
  $$ \%OS = \frac{\max(y) - y_{final}}{y_{final}} \times 100$$
