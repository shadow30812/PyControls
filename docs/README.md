# PyControls

A comprehensive, first-principles Python library for modeling, analysis, design, simulation, and estimation of linear and nonlinear control systems.

**Focus:** MIMO Systems, Derivative-Free Optimization, Real-Time MPC, and Hardware-in-the-Loop (HIL)

---

## 📖 Introduction

PyControls is designed to bridge the gap between abstract control theory and reliable numerical implementation. Unlike standard libraries that often wrap compiled binaries (such as LAPACK/BLAS) behind opaque function calls, PyControls implements core algorithms in pure, JIT-compiled Python. This ensures conceptual clarity for students and inspectability for researchers, without sacrificing performance.

---

## 🛠 Design Goals and Implementation Philosophy

### 1. Mathematical Transparency

Every algorithm exposes its internal mechanics:

* The **Extended Kalman Filter (EKF)** computes Jacobians dynamically using **Complex-Step Differentiation** rather than accepting pre-computed matrices.
* The **Model Predictive Controller (MPC)** explicitly pre-computes condensed quadratic program (QP) matrices, allowing inspection of Hessians and gradients before optimization.

### 2. Numerical Honesty & Robustness

PyControls prioritizes numerically stable methods:

* **Complex-Step Differentiation** replaces finite differences to eliminate subtractive cancellation errors (`core/ekf.py`).
* **Adaptive Integrators (RK45)** are implemented from scratch with access to internal error metrics (`core/solver.py`).
* **Numba JIT Compilation** accelerates tight computational loops while preserving Python readability.

### 3. MIMO by Default

All systems are treated as Multiple-Input Multiple-Output (MIMO). Single-input single-output (SISO) systems are handled as degenerate MIMO cases. All state-space representations (`A`, `B`, `C`, `D`) enforce matrix semantics.

---

## ⚡ Core Engines & Numerical Methods

### Matrix Exponentials: Scaling and Squaring

**File:** `core/solver.py`  
**Function:** `manual_matrix_exp`

Discretization of linear systems requires computing the matrix exponential:

```Markdown
exp(A t)
```

Eigen-decomposition methods are numerically unstable for defective matrices. PyControls instead implements **Scaling and Squaring**:

1. **Scaling** – The matrix `A` is divided by `2^s` until its norm is below 0.5.
2. **Approximation** – A truncated Taylor series or Padé approximant evaluates `exp(A / 2^s)`.
3. **Squaring** – The result is squared `s` times to recover `exp(A)`.

This routine is JIT-compiled with Numba, achieving near C-level performance without reliance on heavy external libraries.

---

### Nonlinear Integration: Adaptive RK5(4)7M

**File:** `core/solver.py`  
**Class:** `NonlinearSolver`

PyControls implements an adaptive Dormand–Prince (RK45) integrator:

* Two simultaneous state estimates are computed: 4th-order (`x₄`) and 5th-order (`x₅`).
* The local truncation error is estimated as `||x₅ − x₄||`.
* Step size adapts dynamically:

  * Error > tolerance → step rejected and reduced.
  * Error ≪ tolerance → step size increased.

This maximizes simulation speed while maintaining accuracy, even for stiff systems.

---

### Jacobian Computation: Complex-Step Differentiation

**Files:** `core/ekf.py`, `core/mpc.py`

Jacobian computation avoids finite differences:

```Markdown
f(x + i·h) ≈ f(x) + i·h·f'(x)
```

Taking the imaginary part yields exact derivatives to machine precision:

```Markdown
f'(x) = Im(f(x + i·h)) / h
```

PyControls uses `h = 1e-20`, achieving high accuracy with zero subtractive cancellation error.

---

## 🎮 Model Predictive Control (MPC)

**File:** `core/mpc.py`

The MPC module features a **dual-mode solver** that automatically selects the appropriate algorithm.

### Mode 1: ADMM for Linear Systems

**Trigger:** When `A` and `B` matrices are provided.

* Uses **Alternating Direction Method of Multipliers (ADMM)**.
* Employs a condensed formulation, optimizing only the input sequence.
* Hessian and inverse are pre-computed during initialization.
* Online execution uses only matrix–vector products and clipping for constraints.

This results in constant-time execution with respect to horizon length during runtime.

### Mode 2: iLQR for Nonlinear Systems

**Trigger:** When a nonlinear `model_func` is supplied.

* Uses **Iterative Linear Quadratic Regulator (iLQR)**.
* Avoids forming a global Hessian, maintaining O(N) memory and time complexity.
* Includes Levenberg–Marquardt regularization for robustness.
* Forward-pass line search ensures monotonic cost reduction.

---

## 📡 Estimation and Filtering

### Extended Kalman Filter (EKF)

**File:** `core/ekf.py`

* Supports simultaneous **state and parameter estimation**.
* Jacobians are computed using Complex-Step Differentiation.
* Automatically detects vectorized user models for efficient Jacobian evaluation.
* Physical parameters can be modeled as slowly varying states, enabling real-time model adaptation.

### Unscented Kalman Filter (UKF)

**File:** `core/ukf.py`

* Implements the Van der Merwe scaled sigma-point formulation.
* Ideal for highly nonlinear systems where EKF linearization is insufficient.

---

## 🎛 Hardware-in-the-Loop (HIL) Support

The `HIL_PWM_Firmware` directory contains embedded firmware for real-time hardware interfacing via Arduino/PlatformIO.

**Firmware Highlights (`src/main.cpp`):**

* Minimal ASCII serial protocol (`Q:VALUE`, `A:VALUE`).
* ADC oversampling with averaging to reduce noise and increase effective resolution.
* Watchdog safety logic zeros actuators on communication loss.

This enables safe and robust real-time control experiments.

---

## 📂 Project Structure

```java
PyControls/
├── core/
│   ├── __init__.py
│   ├── analysis.py
│   ├── control_utils.py
│   ├── ekf.py
│   ├── ekf_discrete.py
│   ├── estimator.py
│   ├── exceptions.py
│   ├── math_utils.py
│   ├── mpc.py
│   ├── solver.py
│   ├── state_space.py
│   ├── transfer_function.py
│   └── ukf.py
├── docs/
│   ├── Complexity Analysis.md
│   ├── Equations and Formulae.md
│   ├── LICENSE
│   └── README.md
├── Heater Project Report/
│   ├── Figure.png
│   └── Project Report.md
├── helpers/
│   ├── config.py
│   ├── exit.py
│   ├── plot.py
│   ├── simulation_runner.py
│   └── system_registry.py
├── HIL_Heater_Firmware/
│   ├── include/
│   │   └── README
│   ├── lib/
│   │   └── README
│   ├── src/
│   │   └── main.cpp
│   ├── test/
│   │   └── README
│   ├── .gitignore
│   └── platformio.ini
├── HIL_PWM_Firmware/
│   ├── include/
│   │   └── README
│   ├── lib/
│   │   └── README
│   ├── src/
│   │   └── main.cpp
│   ├── test/
│   │   └── README
│   ├── .gitignore
│   └── platformio.ini
├── modules/
│   ├── __init__.py
│   ├── interactive_lab.py
│   └── physics_engine.py
├── PWM Project Report/
│   ├── Figure 1.png
│   ├── Figure 2.png
│   ├── Figure 3.png
│   ├── Figure 4.png
│   ├── Figure 5.png
│   ├── Figure 6.png
│   └── Project Report.md
├── systems/
│   ├── __init__.py
│   ├── battery.py
│   ├── dc_motor.py
│   ├── pendulum.py
│   └── thermistor.py
├── tests/
│   ├── test_analysis.py
│   ├── test_control.py
│   ├── test_jit_vec.py
│   ├── test_kf_ekf.py
│   ├── test_math.py
│   ├── test_models.py
│   ├── test_modules.py
│   ├── test_solver.py
│   ├── test_systems.py
│   └── test_ukf_mpc.py
├── .gitignore
├── main.py
├── pyproject.toml
├── requirements.txt
├── script.py
└── test_runner.py
```

---

## 🚀 Getting Started

### Install Dependencies

```bash
pip install -r requirements.txt
```

> **Note:** `numba` is heavily used for solver performance. Ensure it is correctly installed for your architecture.

### Run a Test Simulation (Pendulum)

```bash
python main.py
```

### Run Unit Tests

```bash
python test_runner.py
```

Numerical correctness is critical—tests verify solver accuracy to tolerance.

---

## 📜 License

This project is licensed under the MIT License. See `docs/LICENSE` for details.

---

Version: 5.3.4  
Last Updated: 1 January 2026
