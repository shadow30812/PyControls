# PyControls: First-Principles Python Library for MIMO Controls and Kalman Filtering

A comprehensive Python library for modeling, analyzing, and implementing linear and nonlinear control systems with emphasis on Multiple-Input Multiple-Output (MIMO) systems and optimal estimation through Kalman filtering. PyControls provides production-grade implementations built from first principles with rigorous mathematical foundations for control systems design and analysis.

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Core Modules](#core-modules)
- [Mathematical Foundations](#mathematical-foundations)
- [Physical Systems](#physical-systems)
- [Solvers and Numerical Methods](#solvers-and-numerical-methods)
- [Estimation and Filtering](#estimation-and-filtering)
- [Root Finding Methods](#root-finding-methods)
- [MIMO Control Design](#mimo-control-design)
- [Kalman Filtering](#kalman-filtering)
- [Advanced Topics](#advanced-topics)
- [Examples](#examples)
- [API Reference](#api-reference)
- [Performance Considerations](#performance-considerations)
- [Contributing](#contributing)
- [License](#license)

## Overview

PyControls bridges the gap between theoretical control systems and practical implementation. The library provides mathematically rigorous implementations of classical and modern control theory, with particular strength in MIMO system analysis and state estimation. Whether you are designing feedback controllers for complex electromechanical systems, implementing optimal state estimators, or analyzing system dynamics, PyControls offers the tools needed for robust system design and validation.

The library is structured around core control theory principles including state-space representations, transfer functions, stability analysis, and optimal control techniques. All implementations prioritize numerical stability and computational efficiency while maintaining mathematical transparency.

## Key Features

### MIMO Control Systems

- **State-Space Representations**: Full support for continuous and discrete-time linear systems represented in state-space form
- **Transfer Function Analysis**: Conversion between state-space and transfer function representations with proper handling of multivariable systems
- **Controllability and Observability Analysis**: Comprehensive rank-based tests and structural analysis for system properties
- **Pole Placement and Eigenvalue Assignment**: Robust methods for feedback controller design
- **Decoupling and Modal Control**: Techniques for MIMO system decoupling and independent mode control

### Kalman Filtering

- **Linear Kalman Filter**: Optimal recursive state estimation for linear systems with process and measurement noise
- **Extended Kalman Filter (EKF)**: Estimation for nonlinear systems with Complex Step Differentiation for accurate Jacobian computation
- **Unscented Kalman Filter (UKF)**: Advanced nonlinear filtering without explicit linearization
- **Kalman Smoother (RTS)**: Rauch-Tung-Striebel fixed-interval smoothing for batch estimation
- **Multi-sensor Fusion**: Framework for combining measurements from multiple sensors with different characteristics and sampling rates

### Control Design Methods

- **Linear Quadratic Regulator (LQR)**: Optimal control synthesis for state feedback
- **Linear Quadratic Gaussian (LQG)**: Combined optimal control and estimation for stochastic systems
- **Model Predictive Control (MPC)**: Receding horizon control for constrained systems
- **Robust Control**: Analysis and design considering system uncertainty
- **Frequency Response Methods**: Bode, Nyquist, and Nichols plots with classical stability margins
- **PID Controller Design**: Classical proportional-integral-derivative control with transfer function analysis

### System Analysis

- **Stability Analysis**: Lyapunov methods, eigenvalue analysis, and frequency-domain stability tests
- **Transient Response**: Step response, impulse response, and general trajectory computation
- **Root Locus Analysis**: Interactive parameter variation analysis
- **Steady-State Analysis**: Error analysis and reference tracking

### Advanced Numerical Methods

- **Matrix Exponential Computation**: Scaling and Squaring method combined with Taylor series for numerical stability
- **Exact Discretization**: Zero-Order Hold (ZOH) discretization using Van Loan's method for exact state transition matrices
- **Adaptive Runge-Kutta**: Dormand-Prince RK45 method with automatic step size control for nonlinear systems
- **Complex Step Differentiation**: Machine-precision Jacobian computation avoiding subtractive cancellation errors
- **Hybrid Root Finding**: Brent's method with Newton-Raphson acceleration for robust and fast convergence
- **Riccati Equation Solvers**: Specialized solvers for continuous and discrete algebraic Riccati equations

## Installation

### Prerequisites

- Python 3.10 or later
- NumPy 1.19.0 or later
- Matplotlib 3.2.0 or later
- Numba 0.59.0 or later (optional, for performance optimization)
- SciPy (optional, for testing accuracy)

### Standard Installation

Install PyControls using pip from the Python Package Index:

```bash
pip install pycontrols
```

### Installation from Source

For development or to access the latest features before official release:

```bash
git clone https://github.com/shadow30812/pycontrols.git
cd pycontrols
pip install -e .
```

The `-e` flag enables editable mode, allowing you to modify source files and see changes immediately without reinstalling.

### Installation with Optional Dependencies

To include performance optimization and JIT compilation support via Numba:

```bash
pip install pycontrols[performance]
```

### Verification

Verify the installation by running a simple example:

```python
import numpy as np
from pycontrols import LinearSystem

# Create a simple second-order system
A = np.array([[0, 1], [-2, -3]])
B = np.array([[0], [1]])
C = np.array([[1, 0]])
D = np.array([[0]])

sys = LinearSystem(A, B, C, D)
print(sys.poles())
```

## Quick Start

### Creating and Analyzing a MIMO System

```python
import numpy as np
from pycontrols import LinearSystem, ControlTools

# Define a 2x2 MIMO system (two inputs, two outputs)
A = np.array([[0, 1, 0, 0],
              [-5, -2, 1, 0],
              [0, 0, 0, 1],
              [2, 0, -3, -4]])

B = np.array([[0, 0],
              [1, 0],
              [0, 0],
              [0, 1]])

C = np.array([[1, 0, 0, 0],
              [0, 0, 1, 0]])

D = np.zeros((2, 2))

# Create the system
mimo_sys = LinearSystem(A, B, C, D)

# Analyze controllability and observability
is_controllable = mimo_sys.is_controllable()
is_observable = mimo_sys.is_observable()

print(f"Controllable: {is_controllable}")
print(f"Observable: {is_observable}")

# Compute eigenvalues (poles)
poles = mimo_sys.poles()
print(f"Poles: {poles}")
```

### Implementing a Kalman Filter

```python
import numpy as np
from pycontrols.estimation import KalmanFilter

# System dynamics
dt = 0.01  # Sampling time
A = np.array([[1, dt], [0, 1]])  # Constant velocity model
B = np.array([[0], [dt]])

# Measurement model (position only)
H = np.array([[1, 0]])

# Process noise covariance
Q = np.array([[0.001, 0], [0, 0.001]])

# Measurement noise covariance
R = np.array([[0.1]])

# Initialize state and covariance
x0 = np.array([[0], [0]])
P0 = np.eye(2)

# Create filter
kf = KalmanFilter(A, H, Q, R, x0, P0, B)

# Simulate with noisy measurements
measurements = np.array([0.1, 0.15, 0.2, 0.25, 0.3])

estimates = []
for z in measurements:
    kf.predict(u=np.array([[0]]))
    kf.update(z)
    estimates.append(kf.state.copy())

print("Filtered estimates:", estimates)
```

### Designing an LQR Controller

```python
import numpy as np
from pycontrols import LinearSystem
from pycontrols.control import LQR

# System matrices
A = np.array([[0, 1], [-1, -2]])
B = np.array([[0], [1]])

# Create system
sys = LinearSystem(A, B, np.eye(2), np.zeros((2, 1)))

# Define cost matrices
Q = np.eye(2) * 10  # State cost
R = np.array([[1]])  # Input cost

# Compute optimal gain
K, P = LQR(A, B, Q, R)

print("Optimal feedback gain:\n", K)
print("Cost-to-go:\n", P)
```

## Core Modules

### pycontrols.systems

The systems module provides core functionality for representing and analyzing linear and nonlinear dynamical systems.

**Key Classes:**

- `LinearSystem`: Continuous-time linear system in state-space form
- `DiscreteLinearSystem`: Discrete-time linear system
- `TransferFunction`: Rational polynomial representation of system dynamics
- `NonlinearSystem`: Framework for nonlinear system representation and simulation
- `DCMotor`: Physical DC motor system with electrical and mechanical dynamics

**Common Methods:**

- `step_response()`: Compute system response to unit step input
- `impulse_response()`: Compute system response to unit impulse
- `frequency_response()`: Compute frequency response (magnitude and phase)
- `poles()`: Compute system poles
- `zeros()`: Compute system transmission zeros
- `is_stable()`: Test stability using eigenvalue criterion

### pycontrols.core.solver

Advanced numerical solvers for linear and nonlinear differential equations.

**Key Functions:**

- `matrix_exponential()`: Compute e^(At) using Scaling and Squaring method
- `exact_discretization()`: Zero-Order Hold discretization via Van Loan's method
- `adaptive_rk45()`: Dormand-Prince adaptive Runge-Kutta integrator
- `complex_step_jacobian()`: Compute Jacobians using Complex Step Differentiation
- `hybrid_root_find()`: Brent's method with Newton-Raphson acceleration

**Features:**

- Matrix exponential computed via Scaling and Squaring combined with Taylor series
- Exact discretization using Van Loan's block matrix method for state transition matrices
- Dormand-Prince RK45 with automatic step size control for nonlinear systems
- Complex Step Differentiation for machine-precision Jacobian computation
- Hybrid root finding for robust and efficient convergence

### pycontrols.estimation

Advanced filtering and state estimation implementations for optimal inference from noisy measurements.

**Key Classes:**

- `KalmanFilter`: Linear Kalman filter for Gaussian linear systems
- `ExtendedKalmanFilter`: Nonlinear filtering using first-order Taylor linearization with Complex Step Differentiation
- `UnscentedKalmanFilter`: Nonlinear filtering using sigma point approach
- `KalmanSmoother`: Fixed-interval smoothing for batch estimation
- `MultiSensorFusion`: Framework for fusing measurements from multiple sources

**Common Methods:**

- `predict()`: Propagate state estimate forward in time
- `update()`: Incorporate measurement to refine estimate
- `smooth()`: Apply backward smoothing pass

### pycontrols.control

Comprehensive control design and synthesis tools for feedback system design.

**Key Functions:**

- `pole_placement()`: Compute feedback gain for desired pole locations
- `LQR()`: Linear quadratic regulator synthesis
- `LQG()`: Gaussian linear quadratic controller with estimator
- `care()`: Continuous-time algebraic Riccati equation solver
- `dare()`: Discrete-time algebraic Riccati equation solver

**Key Classes:**

- `PIDController`: Classical proportional-integral-derivative control with transfer function representation
- `StateSpaceController`: General state feedback controller
- `ObserverController`: Observer-based output feedback
- `TransferFunctionController`: Controller represented as rational transfer function

### pycontrols.analysis

Tools for system analysis, stability verification, and performance evaluation.

**Key Functions:**

- `pole_stability()`: Verify pole locations for stability
- `stability_margin()`: Compute gain and phase margins
- `controllability_matrix()`: Construct controllability test matrix
- `observability_matrix()`: Construct observability test matrix
- `gramian()`: Compute Hankel, controllability, and observability gramians

**Key Classes:**

- `FrequencyResponse`: Bode, Nyquist, and Nichols plot generation
- `RootLocus`: Root locus analysis for parameter variation
- `StabilityMargin`: Gain and phase margin computation

### pycontrols.discretization

Continuous-to-discrete time conversion for digital implementation.

**Key Functions:**

- `zero_order_hold()`: Standard discretization with ZOH using exact formulas
- `first_order_hold()`: Higher-order accurate FOH method
- `bilinear()`: Tustin transformation
- `matched_pole_zero()`: Pole-zero matching method
- `sample_system()`: Direct system sampling

## Mathematical Foundations

### Physical Systems: DC Motor

The DC motor represents a fundamental electromechanical system combining electrical and mechanical dynamics. This serves as a reference implementation for understanding how physical systems are modeled in PyControls.

**Electrical Domain (Kirchhoff's Voltage Law):**

The voltage equation describes how electrical energy flows through the armature circuit:

V(t) = i(t)R + L(di/dt) + K_b ω(t)

- **i(t)R**: Resistive voltage drop across armature windings
- **L(di/dt)**: Inductive voltage drop opposing current changes
- **K_b ω(t)**: Back-EMF proportional to motor speed (couples mechanical motion into electrical domain)

**Mechanical Domain (Newton's Second Law for Rotation):**

The torque equation describes rotational dynamics:

T_m(t) = J(dω/dt) + bω(t) + T_load(t)

- **T_m(t) = K_i i(t)**: Motor torque from Lorentz force
- **J(dω/dt)**: Inertial torque for angular acceleration
- **bω(t)**: Viscous friction torque
- **T_load(t)**: External disturbance torque

**State-Space Representation (Continuous):**

State vector: **x = [ω, i]^T** (angular velocity and armature current)

```
[ω̇]   = [-b/J    -K_b/J] [ω]   + [0      1/J  ] [V       ]
[i̇]     [-K_i/L  -R/L  ] [i]     [1/L    0    ] [T_load  ]
```

**Transfer Function (SISO):**

Setting T_load = 0 and applying Laplace transform:

G_p(s) = ω(s)/V(s) = K_i / [(JL)s² + (JR + bL)s + (bR + K_i K_b)]

The characteristic equation denominator includes K_i K_b coupling term, reflecting electrical damping from back-EMF.

## Solvers and Numerical Methods

### Matrix Exponential Computation

For solving linear differential equations ẋ = Ax, we require the matrix exponential e^(At).

**Scaling and Squaring Method:**

1. Scale matrix A by 2^s to achieve ‖A/2^s‖ < 0.5
2. Approximate e^(A/2^s) using truncated Taylor series:

   e^X ≈ I + X + X²/2! + X³/3! + ... + X^k/k!

3. Square the result s times to recover e^(At)

This approach ensures rapid convergence and numerical stability compared to direct Taylor series.

### Exact Discretization (Zero-Order Hold)

For digital implementation with constant inputs u_k held between time steps Δt:

**State Transition Matrix (Φ):**
Φ = e^(AΔt)

**Input Matrix (Γ):**
Γ = ∫₀^(Δt) e^(Aτ) dτ B

**Van Loan's Method (Efficient Implementation):**

Construct block matrix:

```
M = [A    B]
    [0    0]
```

Compute e^(MΔt) = [Φ    Γ]
                  [0    I]

This recovers both transition matrices through a single matrix exponential.

### Adaptive Runge-Kutta (Dormand-Prince RK45)

For nonlinear systems where exact discretization is impossible:

**Method:**

- Evaluates derivative f(t, x, u) at 7 stages within step size h
- Computes two approximations: 5th order (y_{n+1}) and 4th order (y*_{n+1})
- Estimates local truncation error: ε = ||y_{n+1} - y*_{n+1}||

**Adaptive Timestepping:**
h_new = 0.9h · (ε_tol / ε)^0.2

- If ε << ε_tol: increases h to save computation
- If ε >> ε_tol: decreases h to maintain accuracy

### Complex Step Differentiation

**Problem with Finite Differences:**

Finite difference approximations suffer from a fundamental limitation—the tradeoff between truncation error and subtractive cancellation error. For step size h:

- Truncation error: O(h) or O(h²)
- Cancellation error: O(ε/h) where ε is machine precision

This inherent tradeoff prevents achieving accuracy better than around √ε ≈ 10^(-8) in double precision.

**Complex Step Solution:**

Exploit the Cauchy-Riemann equations to evaluate the derivative of f(x) at complex-valued points:

**f(x + ihε) = f_real(x, ε) + i·f_imag(x, ε)**

where i = √(-1) and h is a real scalar step size.

The derivative is recovered from the imaginary part:

**∂f/∂x ≈ Im(f(x + ihε)) / hε**

**Key Advantages:**

1. **Machine-Precision Accuracy**: With ε ≈ 10^(-20), achieves accuracy to ~10^(-14) in double precision—no tradeoff between truncation and cancellation error
2. **No Subtractive Cancellation**: The imaginary and real parts are computed independently; no subtraction of nearby numbers
3. **Jacobian Computation**: Compute entire rows of the Jacobian matrix J = ∂f/∂x with a single function evaluation (with complex argument)

**Application in PyControls:**

In the Extended Kalman Filter, the Jacobian of the nonlinear state transition function f(x, u) is computed using Complex Step Differentiation for maximum accuracy:

```python
from pycontrols.core.solver import complex_step_jacobian

# Compute Jacobian F = ∂f/∂x at the current state estimate
F = complex_step_jacobian(f, x_estimate, u_input, step=1e-20)

# Use F for covariance propagation in EKF prediction step
P_pred = F @ P_current @ F.T + Q
```

This ensures that linearization errors in the EKF are minimized, improving filter performance for mildly nonlinear systems.

## Root Finding Methods

### Hybrid Brent-Newton Method

For control systems applications, we often need to solve nonlinear equations of the form **f(x) = 0**. Examples include:

- Finding frequency response peaks in Bode plots
- Solving for equilibrium points in nonlinear systems
- Locating pole-zero cancellations
- Computing root-mean-square values iteratively

**Individual Algorithms:**

**Brent's Method:**

- Combines bisection, secant method, and inverse quadratic interpolation
- Guaranteed to converge if a sign change exists in the bracket
- Convergence rate: superlinear (1.618 between linear and quadratic)
- Bracketing required but very robust to initial conditions
- Standard choice when reliability is paramount

**Newton-Raphson Method:**

- Quadratic convergence when started near the root
- Requires derivative computation (or complex-step approximation)
- Can diverge or cycle if initial guess is poor
- Very fast near the solution but unreliable globally

**Hybrid Approach:**

PyControls implements a hybrid algorithm that combines the strengths of both:

1. **Start with Brent's method** to localize the root in a narrow region using bracketing
2. **Switch to Newton-Raphson** once the error is sufficiently small (|f(x)| < 10^(-4) or x changes by < 10^(-6))
3. **Fallback to Brent** if Newton steps diverge or stall

**Convergence Guarantee:**

- Initial bracketing ensures we never diverge far from the solution
- Newton-Raphson provides quadratic convergence near the root
- Typical convergence: 5-10 iterations for machine precision on well-behaved functions

**Implementation:**

```python
from pycontrols.core.solver import hybrid_root_find

# Solve f(x) = 0 in interval [a, b]
root = hybrid_root_find(f, a, b, f_derivative=None, tol=1e-12)

# If f_derivative is None, Complex Step Differentiation is used
# If provided, the derivative is used directly for Newton-Raphson steps
```

**Example: Finding Resonance Frequency**

```python
from pycontrols import TransferFunction
from pycontrols.core.solver import hybrid_root_find

# Define transfer function
G = TransferFunction([1], [1, 2, 5])  # Second-order system

# Define magnitude response to find peak
def magnitude_response(omega):
    H = G.frequency_response(omega)
    return abs(H)

# Find frequency where d|G(jω)|/dω = 0 (peak)
peak_freq = hybrid_root_find(
    lambda w: G.magnitude_derivative(w),
    0.1, 5.0
)
```

## Estimation and Filtering

### Standard Kalman Filter (Linear)

Optimal recursive estimator minimizing mean squared error for linear systems with Gaussian noise.

**Prediction (Time Update):**

Projects state and uncertainty forward:

x̂_{k|k-1} = Φ x̂_{k-1|k-1} + Γ u_k

P_{k|k-1} = Φ P_{k-1|k-1} Φ^T + Q

- **Q**: Process noise covariance (model uncertainty)

**Correction (Measurement Update):**

Adjusts estimate based on actual measurements:

ỹ_k = z_k - C x̂_{k|k-1}  (Innovation)

S_k = C P_{k|k-1} C^T + R  (Innovation Covariance)

K_k = P_{k|k-1} C^T S_k^(-1)  (Optimal Kalman Gain)

x̂_{k|k} = x̂_{k|k-1} + K_k ỹ_k

P_{k|k} = (I - K_k C) P_{k|k-1}

**Optimality:** The Kalman filter is the minimum variance linear filter—no other linear estimator can achieve lower error covariance.

### Extended Kalman Filter with Complex Step Differentiation

For nonlinear systems, the EKF linearizes around the current state estimate. Jacobian accuracy directly affects filter performance.

**Jacobian Computation (Complex Step Differentiation):**

```python
# State transition function: x_{k+1} = f(x_k, u_k)
def f(x, u):
    # Nonlinear dynamics
    return ...

# Measurement function: z_k = h(x_k)
def h(x):
    # Nonlinear measurement model
    return ...

# Compute Jacobians with machine precision
F_k = complex_step_jacobian(lambda x: f(x, u_k), x_est, step=1e-20)
H_k = complex_step_jacobian(h, x_est, step=1e-20)
```

**Prediction:**

Uses full nonlinear dynamics for state propagation, then linearizes for covariance:

x̂_{k|k-1} = f(x̂_{k-1|k-1}, u_k)

P_{k|k-1} = F_k P_{k-1|k-1} F_k^T + Q

**Correction:** (identical to Linear Kalman Filter after linearization)

**Application: Dual Estimation**

The augmented state vector enables simultaneous state and parameter estimation:

x_aug = [ω, i, θ_param]^T

where θ_param is modeled as a random walk (dθ/dt = 0 + process noise).

This allows tracking of slowly-varying parameters like load disturbances or system changes.

## MIMO Control Design

### MIMO System Properties

MIMO systems exhibit unique characteristics requiring specialized analysis:

**Decoupling**: MIMO systems may have cross-coupled channels where one input affects multiple outputs. The library provides tools to analyze and design decoupling compensators:

```python
from pycontrols.mimo import DecouplingSynthesis

# Analyze decoupling properties
decoupler = DecouplingSynthesis(A, B, C, D)
is_decouplable = decoupler.check_structural_properties()
decoupling_matrix = decoupler.compute_decoupler(method='dynamic')
```

**Transmission Zeros**: Unlike SISO systems, MIMO systems can have transmission zeros that limit achievable performance. The library includes zero computation and analysis:

```python
from pycontrols.mimo import TransmissionZeros

zeros = mimo_sys.transmission_zeros()
for z in zeros:
    print(f"Zero at: {z}")
```

**Relative Gain Array (RGA)**: Tool for evaluating channel interactions and controller pairing:

```python
from pycontrols.mimo import relative_gain_array

rga = relative_gain_array(sys.freq_response(frequencies))
# RGA close to identity matrix indicates weak interactions
```

### Multivariable Control Synthesis

Designing controllers for MIMO systems requires consideration of interactions and cross-coupling:

```python
from pycontrols.mimo import MIMO_LQR

# State and input cost matrices
Q = 10 * np.eye(4)  # Penalize state deviations
R = np.eye(2)       # Penalize control effort

# Compute MIMO LQR gain
K = MIMO_LQR(A, B, Q, R)

# Closed-loop system
A_cl = A - B @ K
print("Closed-loop poles:", np.linalg.eigvals(A_cl))
```

## Kalman Filtering

### Linear Kalman Filter

The linear Kalman filter provides optimal state estimation for linear systems with Gaussian noise:

```python
from pycontrols.estimation import KalmanFilter
import numpy as np

# Continuous-time system
A = np.array([[0, 1], [-1, -2]])
B = np.array([[0], [1]])
H = np.array([[1, 0]])

# Discretize with dt=0.01
from pycontrols.discretization import zero_order_hold
Ad, Bd = zero_order_hold(A, B, 0.01)

# Noise covariances
Q = 0.01 * np.eye(2)  # Process noise
R = np.array([[0.1]])  # Measurement noise

# Initial conditions
x0 = np.zeros((2, 1))
P0 = np.eye(2)

# Create filter
kf = KalmanFilter(Ad, H, Q, R, x0, P0, Bd)

# Filtering loop
for measurement in measurements:
    kf.predict(u=control_input)
    kf.update(measurement)
    state_estimate = kf.state
    state_covariance = kf.covariance
```

### Extended Kalman Filter for Nonlinear Systems

When system dynamics are nonlinear, the Extended Kalman Filter uses Jacobian linearization with Complex Step Differentiation for accuracy:

```python
from pycontrols.estimation import ExtendedKalmanFilter

# Nonlinear system model
def dynamics(x, u, t):
    return np.array([x[1], -np.sin(x[0]) - 0.5*x[1] + u[0]])

def measurement_model(x):
    return np.array([x[0]])

# Jacobian computation uses Complex Step Differentiation internally
ekf = ExtendedKalmanFilter(
    dynamics, measurement_model,
    Q, R, x0, P0,
    method='complex_step'  # Uses complex-step for Jacobians
)

# Filtering loop
for z in measurements:
    ekf.predict(u=control_input, dt=0.01)
    ekf.update(z)
```

### Multi-Sensor Fusion

Combining measurements from multiple sensors with different characteristics:

```python
from pycontrols.estimation import MultiSensorKalmanFilter

# Define multiple measurement models
measurement_models = [
    {'H': H_gps, 'R': R_gps, 'delay': 0.05},
    {'H': H_imu, 'R': R_imu, 'delay': 0.01}
]

mskf = MultiSensorKalmanFilter(
    A, measurement_models, Q, x0, P0
)

# Asynchronous update
for sensor_id, (z, timestamp) in enumerate(measurements):
    mskf.predict(timestamp)
    mskf.update(z, sensor_id)
```

### Fixed-Interval Smoothing (Rauch-Tung-Striebel)

For batch processing or offline applications, fixed-interval smoothing provides better estimates:

```python
from pycontrols.estimation import KalmanSmoother

# Forward pass: run standard Kalman filter
kf = KalmanFilter(A, H, Q, R, x0, P0)
forward_estimates = []

for z in measurements:
    kf.predict()
    kf.update(z)
    forward_estimates.append((kf.state.copy(), kf.covariance.copy()))

# Backward pass: apply RTS smoother
smoother = KalmanSmoother(A, H, Q, R, x0, P0)
smoothed_estimates = smoother.rts_smooth(forward_estimates, measurements)
```

## Advanced Topics

### Robust Control Synthesis

Account for system uncertainty in controller design:

```python
from pycontrols.robust import H_infinity_control

# Nominal system
A_nom = np.array([[0, 1], [-1, -2]])
B_nom = np.array([[0], [1]])
C_nom = np.eye(2)

# Uncertainty bounds
uncertainty_bound = 0.1  # 10% parametric uncertainty

# H-infinity design with noise attenuation level gamma=10
K, gamma = H_infinity_control(A_nom, B_nom, C_nom, gamma=10, uncertainty=uncertainty_bound)
```

### Model Predictive Control

Receding horizon control for constrained systems:

```python
from pycontrols.control import MPC

mpc = MPC(
    A, B, C, D,
    N=10,  # Prediction horizon
    Q=10*np.eye(2),  # State cost
    R=np.eye(1),     # Input cost
    u_min=np.array([-1]),  # Input lower bound
    u_max=np.array([1])    # Input upper bound
)

# Solve optimal control problem at each time step
for state in trajectory:
    u_optimal = mpc.solve(state, reference_trajectory)
    apply_control(u_optimal)
```

### Gain Scheduling

Automatic controller adaptation for parameter-varying systems:

```python
from pycontrols.adaptive import GainScheduledController

# Design controllers for multiple operating points
operating_points = [0, 5, 10, 15]  # Parameter values
controllers = {}

for param in operating_points:
    A_param = get_system_at_parameter(param)
    K = pole_placement(A_param, B, desired_poles)
    controllers[param] = K

# Interpolate gains during operation
gain_scheduler = GainScheduledController(controllers, operating_points)

# Online adaptation
for measurement in measurements:
    operating_param = estimate_parameter(measurement)
    K_current = gain_scheduler.interpolate(operating_param)
    control_input = -K_current @ state
```

## Examples

The library includes comprehensive examples covering common applications:

### Example 1: DC Motor Control

Control of DC motor speed using PID and state feedback:

```python
# See examples/dc_motor_control.py
# Demonstrates:
# - DC motor physics and state-space representation
# - PID controller tuning
# - Pole placement design
# - Comparison of control methods
```

### Example 2: Satellite Attitude Control

Three-axis attitude control of a spacecraft using reaction wheels:

```python
# See examples/satellite_attitude_control.py
# Demonstrates:
# - 3D rigid body dynamics
# - MIMO LQR design
# - Quaternion-based attitude representation
# - Momentum management
```

### Example 3: Inverted Pendulum

Classic nonlinear system with state feedback linearization:

```python
# See examples/inverted_pendulum.py
# Demonstrates:
# - Nonlinear system dynamics
# - Linearization about equilibrium
# - Pole placement design
# - Simulation and visualization
```

### Example 4: Multi-Rate Filtering

Sensor fusion with different measurement rates:

```python
# See examples/multi_rate_filtering.py
# Demonstrates:
# - Asynchronous Kalman filter
# - Multi-sensor fusion
# - Handling irregular measurement timing
```

### Example 5: Parameter Estimation

Simultaneous state and parameter estimation using EKF with Complex Step Differentiation:

```python
# See examples/parameter_estimation.py
# Demonstrates:
# - Augmented state for parameter tracking
# - Complex Step Differentiation for Jacobians
# - Dual estimation of states and unknowns
# - Disturbance rejection via parameter estimation
```

### Example 6: Robust Control Design

H-infinity control under uncertainty:

```python
# See examples/robust_design.py
# Demonstrates:
# - Uncertainty characterization
# - Robust performance margins
# - Structured singular value analysis
```

## API Reference

### pycontrols.systems.LinearSystem

**Signature:**

```python
LinearSystem(A, B, C, D, dt=None)
```

**Parameters:**

- `A`: State matrix (n x n)
- `B`: Input matrix (n x m)
- `C`: Output matrix (p x n)
- `D`: Feedthrough matrix (p x m)
- `dt`: Sampling time for discrete systems (None for continuous)

**Methods:**

- `poles()`: Return system eigenvalues
- `zeros()`: Return transmission zeros
- `step_response(t)`: Compute step response at times t
- `frequency_response(w)`: Compute magnitude and phase
- `is_controllable()`: Test controllability
- `is_observable()`: Test observability
- `is_stable()`: Test asymptotic stability

### pycontrols.estimation.KalmanFilter

**Signature:**

```python
KalmanFilter(A, H, Q, R, x0, P0, B=None, method='standard')
```

**Parameters:**

- `A`: State transition matrix (n x n)
- `H`: Measurement matrix (p x n)
- `Q`: Process noise covariance (n x n)
- `R`: Measurement noise covariance (p x p)
- `x0`: Initial state estimate (n x 1)
- `P0`: Initial error covariance (n x n)
- `B`: Control input matrix (n x m, optional)
- `method`: 'standard' or 'joseph' (for numerical stability)

**Methods:**

- `predict(u=None)`: Predict state at next time step
- `update(z)`: Update estimate based on measurement z
- `get_state()`: Return current state estimate
- `get_covariance()`: Return current error covariance

### pycontrols.core.solver

**Key Functions:**

- `matrix_exponential(A, t)`: Compute e^(At) using Scaling and Squaring method
- `exact_discretization(A, B, dt)`: Compute discrete-time system matrices via Van Loan method
- `adaptive_rk45(f, t0, x0, dt, tol)`: Integrate using Dormand-Prince RK45
- `complex_step_jacobian(f, x, step=1e-20)`: Compute Jacobian using Complex Step Differentiation
- `hybrid_root_find(f, a, b, f_derivative=None, tol=1e-12)`: Solve f(x)=0 using hybrid Brent-Newton method

## Performance Considerations

### Numerical Stability

The library implements numerically stable algorithms for ill-conditioned problems:

- Matrix exponential uses Scaling and Squaring with norm-based scaling criterion
- Kalman filters support Joseph form for guaranteed covariance symmetry
- Eigenvalue computations use QR algorithm with proper deflation
- Rank tests use SVD with configurable tolerance
- Zero-Order Hold discretization uses Van Loan's exact method avoiding integration errors
- Jacobian computation uses Complex Step Differentiation (machine precision, no subtractive cancellation)

Configure stability options:

```python
from pycontrols.config import set_numerical_tolerance

set_numerical_tolerance(1e-12)  # Default 1e-10
```

### Computational Complexity

Design considerations for real-time systems:

- Kalman filter: O(n^3) per update for n-state system
- Matrix exponential: O(n^3) per evaluation
- Complex Step Jacobian: O(n·m) function evaluations for n states, m parameters
- Pole placement: O(n^3) one-time computation
- LQR: O(n^3) for Riccati equation solution
- MPC: O((Nn)^3) for N-horizon prediction
- RK45 integration: O(n) per stage evaluation
- Hybrid root finding: typically 5-10 iterations for convergence

For large-scale systems (n > 1000), consider:

```python
# Use sparse matrix representations
from pycontrols.linalg import SparseLinearSystem

# Numba JIT compilation for performance
# from pycontrols.config import enable_numba_jit
# enable_numba_jit()

# Distributed computing approaches
# Reduced-order models for control design
```

### JIT Compilation with Numba

Enable just-in-time compilation for performance-critical code:

```python
from pycontrols.config import enable_numba_jit

enable_numba_jit()  # Requires Numba 0.59.0+

# Now integrators and filter loops execute at compiled speed
```

### Caching and Memoization

Reuse computed results for repeated evaluations:

```python
from pycontrols.utils import memoize

@memoize
def expensive_computation(x):
    # Computation cached based on input x
    return result
```

## Contributing

We welcome contributions from the community. To contribute:

1. Fork the repository on GitHub
2. Create a feature branch (`git checkout -b feature/your-feature`)
3. Implement your changes with comprehensive testing
4. Ensure all tests pass (`pytest tests/`)
5. Submit a pull request with clear description

### Development Setup

```bash
git clone https://github.com/shadow30812/pycontrols.git
cd pycontrols
pip install -e .[dev]
pytest tests/
```

### Code Standards

- Follow PEP 8 style guidelines
- Include docstrings for all public functions and classes
- Write unit tests for new functionality
- Maintain backward compatibility
- Include mathematical references in docstrings for algorithm implementations

## License

PyControls is released under the MIT License. See LICENSE file for full terms.

## Acknowledgments

This library draws upon decades of control theory research and incorporates best practices from:

- Classical control theory (Nyquist, Bode, root locus)
- Modern control theory (state space, optimal control)
- Estimation theory (Kalman, Bayesian filtering)
- Robust control (H-infinity, LMI methods)
- Numerical methods (matrix exponentials, discretization, adaptive integration)
- Numerical differentiation (Complex Step Differentiation for machine-precision Jacobians)
- Root finding (Brent, Newton-Raphson, and hybrid methods)

## References

For detailed mathematical foundations and algorithm descriptions, see:

- `Mathematical Reference`: Equations-and-Formulae.md
- `Computational Reference`: Complexity Analysis.md
- DC Motor modeling and state-space representation
- Matrix exponential computation via Scaling and Squaring
- Zero-Order Hold discretization using Van Loan's method
- Kalman filter prediction and correction cycles
- Dormand-Prince RK45 adaptive integration
- Complex Step Differentiation for Jacobian computation
- Hybrid Brent-Newton method for robust root finding

## Support and Documentation

For more information and support:

- GitHub Issues: Report bugs and request features
- Documentation: Full API documentation and tutorials
- Examples: Working code examples for common applications
- Mathematical Reference: Detailed equations and derivations
- Contact: Project maintainers available for questions

---

**Last Updated:** December 2025
**Version:** 1.0.0
