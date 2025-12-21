# PyControls

A comprehensive, first-principles Python library for modeling, analysis, design, simulation, and estimation of linear and nonlinear control systems, with a particular emphasis on Multiple-Input Multiple-Output (MIMO) systems and Kalman-filter–based state estimation.

This document is intentionally written to serve **both as a project README and as a course-companion text** for upper-level undergraduate and graduate courses in control systems, state estimation, and applied linear systems theory. The emphasis throughout is on *conceptual clarity*, *mathematical rigor*, and *numerically explicit implementation*.

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Design Goals and Philosophy](#2-design-goals-and-philosophy)
3. [Scope of the Library](#3-scope-of-the-library)
4. [Key Features Overview](#4-key-features-overview)
5. [Mathematical Foundations](#5-mathematical-foundations)
6. [System Representations](#6-system-representations)

   * [Continuous-Time Models](#continuous-time-models)
   * [Discrete-Time Models](#discrete-time-models)
   * [State-Space Formulation](#state-space-formulation)
   * [Transfer Functions and MIMO Considerations](#transfer-functions-and-mimo-considerations)
7. [Numerical Methods and Design Rationale (Core)](#7-numerical-methods-and-design-rationale-core)

   * [Matrix Exponentials](#matrix-exponentials)
   * [Discretization Methods](#discretization-methods)
   * [Numerical Integration](#numerical-integration)
   * [Differentiation and Jacobians](#differentiation-and-jacobians)
   * [Root Finding and Stability Computations](#root-finding-and-stability-computations)
8. [Control Design Capabilities](#8-control-design-capabilities)

   * [Classical Control Concepts](#classical-control-concepts)
   * [Modern State-Space Control](#modern-state-space-control)
   * [Optimal Control](#optimal-control)
9. [Estimation and Filtering](#9-estimation-and-filtering)

   * [Linear Kalman Filtering](#linear-kalman-filtering)
   * [Extended Kalman Filtering](#extended-kalman-filtering)
   * [Unscented Kalman Filtering](#unscented-kalman-filtering)
   * [Smoothing Algorithms](#smoothing-algorithms)
10. [Simulation Framework](#10-simulation-framework)
11. [Frequency-Domain Analysis](#11-frequency-domain-analysis)
12. [Time-Domain Analysis and Metrics](#12-time-domain-analysis-and-metrics)
13. [Stability Analysis](#13-stability-analysis)
14. [Performance Metrics and Evaluation](#14-performance-metrics-and-evaluation)
15. [Project Structure](#15-project-structure)
16. [Installation and Packaging](#16-installation-and-packaging)
17. [Comparison with Other Control Libraries](#17-comparison-with-other-control-libraries)
18. [Typical Use Cases](#18-typical-use-cases)
19. [License](#19-license)
20. [Acknowledgements](#20-acknowledgements)

---

## 1. Introduction

Control systems theory lies at the intersection of mathematics, physics, and computation. While the theory is well-established, translating mathematical results into reliable numerical implementations remains a nontrivial task. PyControls exists to explicitly bridge this gap.

This library is designed for users who want to understand *why* an algorithm works, not merely *how* to call it. As such, PyControls exposes internal matrices, intermediate quantities, and numerical assumptions that are often hidden in higher-level tools.

As a course companion, this document assumes familiarity with:

* Linear algebra (matrices, eigenvalues, vector spaces)
* Differential equations
* Basic probability and statistics

---

## 2. Design Goals and Philosophy

### Transparency over Convenience

Every major algorithm is implemented in a way that mirrors its mathematical derivation. For example, Kalman filtering is presented as a sequence of prediction and correction equations rather than a monolithic function call.

### MIMO as the Default

All systems are assumed to be potentially multi-input and multi-output. SISO systems are treated as a special case of the general formulation.

### Numerical Honesty

Floating-point limitations, conditioning issues, and algorithmic trade-offs are acknowledged explicitly. This is particularly important in educational settings, where idealized theory often collides with numerical reality.

---

## 3. Scope of the Library

PyControls supports:

* Linear time-invariant (LTI) systems
* Linear time-varying (LTV) systems (limited)
* Nonlinear systems for simulation and estimation
* Stochastic systems with Gaussian noise assumptions

It intentionally avoids graphical modeling tools, focusing instead on explicit computational representations.

---

## 4. Key Features Overview

* Explicit state-space modeling
* Time-domain and frequency-domain analysis
* Optimal control design (LQR, LQG)
* Kalman filtering and smoothing
* Numerically transparent core routines

---

## 5. Mathematical Foundations

At its core, PyControls relies on the theory of linear dynamical systems:

ẋ(t) = A x(t) + B u(t)

y(t) = C x(t) + D u(t)

Key theoretical pillars include:

* Eigenvalue-based stability analysis
* Controllability and observability
* Lyapunov stability theory
* Stochastic state estimation

Throughout this document, mathematical results are paired with their computational counterparts.

---

## 6. System Representations

### Continuous-Time Models

Continuous-time models are central to physical system modeling. PyControls represents these systems explicitly and provides tools for simulation, analysis, and discretization.

### Discrete-Time Models

Discrete-time representations are essential for digital control. PyControls provides explicit discretization routines rather than opaque wrappers.

### State-Space Formulation

State-space models are the primary internal representation used throughout the library, enabling uniform handling of MIMO systems.

### Transfer Functions and MIMO Considerations

Transfer functions are provided primarily for conceptual understanding and frequency-domain analysis. Internally, state-space realizations are preferred for numerical robustness.

---

## 7. Numerical Methods and Design Rationale (Core)

This section forms the numerical backbone of PyControls and is particularly important from an educational standpoint. Unlike many libraries that defer all low-level operations to external numerical packages, PyControls deliberately implements several foundational routines itself. This choice is pedagogical as much as it is practical: by owning these implementations, the library can expose assumptions, intermediate quantities, and numerical trade-offs that are usually hidden.

### Matrix Exponentials

The matrix exponential

e^{A t}

is central to the solution of linear time-invariant systems. In PyControls, matrix exponentials are implemented using the **scaling-and-squaring method combined with Padé approximants**. This method is chosen because:

* It is numerically stable across a wide range of matrix norms
* It avoids explicit eigen-decomposition, which can be ill-conditioned for defective matrices
* It provides predictable error behavior suitable for control applications

Rather than calling a single opaque routine, PyControls exposes the scaling factor, Padé order, and intermediate squared matrices. This allows students to directly observe how numerical conditioning changes with time step size and system dynamics.

This explicit implementation is particularly valuable when teaching discretization and continuous-to-discrete transformations, where misunderstanding of the matrix exponential is common.

### Discretization Methods

Zero-Order Hold (ZOH) discretization is implemented via matrix augmentation:

[ A  B ]
[ 0  0 ]

followed by a matrix exponential. This approach mirrors the theoretical derivation found in standard control textbooks and avoids the conceptual leap of treating discretization as a black-box operation.

Unlike some libraries that rely exclusively on built-in discretization routines, PyControls allows inspection of the augmented system matrix and the resulting discrete-time blocks. This is especially useful when studying sampling effects, discretization error, and multi-rate systems.

### Numerical Integration

For nonlinear simulation, PyControls employs adaptive Runge–Kutta methods (such as RK45). These methods are implemented to emphasize:

* Local truncation error estimation
* Adaptive step-size control
* Explicit handling of stiff vs non-stiff dynamics

While SciPy provides similar integrators, PyControls integrates them tightly with control-system abstractions, enabling seamless simulation of closed-loop systems with estimators and controllers in the loop.

### Differentiation and Jacobians

One area where PyControls diverges significantly from many standard libraries is in its treatment of Jacobians for nonlinear estimation.

Rather than relying solely on finite differences, PyControls uses **complex-step differentiation** wherever applicable. This method provides:

* Machine-precision derivatives
* No subtractive cancellation
* Minimal tuning parameters

This is particularly important for Extended Kalman Filters, where poor Jacobian accuracy can destabilize the estimator. Complex-step differentiation is rarely exposed in mainstream control libraries, despite its strong numerical properties.

### Root Finding and Stability Computations

Stability margins and time-domain metrics often require solving implicit equations. PyControls implements hybrid root-finding strategies that combine:

* Bracketing methods for guaranteed convergence
* Open methods (e.g., Newton-like updates) for speed

This hybrid approach ensures robustness while maintaining reasonable computational efficiency. Importantly, these routines are written to make failure modes explicit, allowing users to see when and why a computation becomes ill-conditioned.

---

## 8. Control Design Capabilities

### Classical Control Concepts

Frequency-domain tools support loop-shaping and robustness analysis.

### Modern State-Space Control

State-feedback controllers are designed using explicit pole placement and Riccati-based methods.

### Optimal Control

LQR design is presented as the solution to an optimization problem, with explicit derivation of the optimal gain.

---

## 9. Estimation and Filtering

### Linear Kalman Filtering

The Kalman filter is implemented as a recursive estimator with explicit covariance propagation.

### Extended Kalman Filtering

Nonlinear systems are handled through local linearization and explicit Jacobian computation.

### Unscented Kalman Filtering

The UKF propagates sigma points through nonlinear dynamics, avoiding explicit linearization.

### Smoothing Algorithms

Rauch–Tung–Striebel smoothing refines state estimates using future measurements.

---

## 10. Simulation Framework

Simulation tools allow controlled experimentation with system dynamics, disturbances, and noise.

---

## 11. Frequency-Domain Analysis

Frequency-domain analysis tools support stability and robustness assessment using classical methods.

---

## 12. Time-Domain Analysis and Metrics

Time-domain metrics quantify transient and steady-state performance using numerically robust algorithms.

---

## 13. Stability Analysis

Stability is assessed using eigenvalue analysis and Lyapunov-based methods.

---

## 14. Performance Metrics and Evaluation

Performance metrics enable objective comparison of competing control designs.

---

## 15. Project Structure

```
PyControls/
├── core/
│   ├── __init__.py
│   ├── analysis.py
│   ├── control_utils.py
│   ├── ekf.py
│   ├── estimator.py
│   ├── exceptions.py
│   ├── math_utils.py
│   ├── mpc.py
│   ├── solver.py
│   ├── state_space.py
│   ├── transfer_function.py
│   └── ukf.py
├── modules/
│   ├── __init__.py
│   ├── interactive_lab.py
│   └── physics_engine.py
├── systems/
│   ├── __init__.py
│   ├── dc_motor.py
│   └── pendulum.py
├── tests/
│   ├── test_analysis.py
│   ├── test_control.py
│   ├── test_core_models.py
│   ├── test_estimators.py
│   ├── test_jit_vec.py
│   ├── test_math.py
│   ├── test_modules.py
│   ├── test_mpc.py
│   ├── test_solvers.py
│   └── test_systems.py
├── .gitignore
├── Complexity Analysis.md
├── Equations and Formulae.md
├── MIT License.txt
├── README.md
├── config.py
├── exit.py
├── main.py
├── pyproject.toml
├── requirements.txt
├── system_registry.py
└── test_runner.py
```

---

## 16. Installation and Packaging

PyControls uses modern Python packaging via `pyproject.toml`.

To install:

pip install .

For development:

pip install -e .

---

## 17. Comparison with Other Control Libraries

Understanding PyControls is aided by contrasting it with existing tools. The goal here is not to rank libraries, but to clarify design intent and educational focus.

### MATLAB Control System Toolbox

MATLAB’s Control System Toolbox is a mature, industrial-grade environment offering extensive functionality, tight integration with Simulink, and highly optimized numerical routines. However, it is fundamentally a **closed system**:

* Many algorithms are exposed only through high-level interfaces
* Intermediate numerical steps are hidden
* Licensing restricts inspection and modification

PyControls differs by explicitly prioritizing:

* Open-source transparency
* Inspectable numerical workflows
* Alignment with textbook derivations

For educational and research contexts where understanding algorithmic structure is critical, this transparency is a decisive advantage.

### python-control

The `python-control` library provides a MATLAB-like API for Python users and is well-suited for users transitioning from MATLAB. Its design emphasizes familiarity and convenience.

PyControls takes a different approach:

* APIs are designed around mathematical structure rather than MATLAB compatibility
* Estimation (Kalman filtering, smoothing) is treated as a first-class concern
* Numerical primitives are more explicitly exposed

As a result, PyControls may feel lower-level, but it rewards users with deeper insight into system behavior.

### SciPy

SciPy provides foundational numerical routines such as linear algebra solvers and ODE integrators. However, it intentionally avoids domain-specific abstractions.

PyControls can be viewed as a **domain-specific layer built on similar numerical principles**, adding:

* Control-theoretic structure
* Consistent handling of MIMO systems
* Integrated analysis, design, and estimation workflows

Many algorithms in PyControls could be implemented using SciPy primitives alone, but PyControls organizes them in a way that reflects control theory itself.

---

## 18. Typical Use Cases

* Control systems education
* Research and algorithm development
* Numerical experimentation

---

## 19. License

This project is licensed under the MIT License.

---

## 20. Acknowledgements

PyControls is inspired by classical control theory texts and the open-source scientific computing community.

---

Version: 4.0.1  
Last Updated: 21 December 2025
