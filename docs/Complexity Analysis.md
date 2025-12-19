# PyControls Algorithm Complexity Analysis

This document details the theoretical Time and Space complexities for every function and class implemented in the PyControls library.

## Variable Definitions

- **n**: Number of state variables (State Dimension)
- **m**: Number of input/control variables
- **p**: Number of output/measurement variables
- **N**: Number of simulation time steps
- **H**: MPC Prediction Horizon
- **d**: Degree of Transfer Function polynomials
- **k**: Taylor series order (fixed at 20 for matrix exp)
- **I**: Number of iterations (for solvers/optimizers/root finders)
- **C_f**: Computational cost of evaluating a dynamics function f(x, u)
- **S**: Number of Sigma points in UKF (2n + 1)
- **L**: String length (for parsing functions)
- **T_tests**: Number of tests in the suite

---

## 1. Core Modules

### 1.1. Solver (`core/solver.py`)

#### `_mat_mul(A, B)`

Manual matrix multiplication.

- **Time**: **O(rows_A · cols_A · cols_B)**. For square matrices of size n: **O(n³)**.
- **Space**: **O(rows_A · cols_B)** to store the result C.

#### `manual_matrix_exp(A, order)`

Computes matrix exponential e^A via Taylor Series.

- **Time**: **O(n³ · (order + s))**, where `s` is the scaling factor.
- **Space**: **O(n²)** for intermediate matrix storage (E, term, A_scaled).

#### `jit(func)` (Decorator dummy)

- **Time**: **O(1)**.
- **Space**: **O(1)**.

#### `ExactSolver` (Class)

- **`__init__(A, B, C, D, dt)`**:
  - **Time**: **O((n+m)³)**. Computes matrix exponential of a block matrix.
  - **Space**: **O((n+m)²)**. Stores Φ (nxn) and Γ (nxm).
- **`step(u_input)`**:
  - **Time**: **O(n² + n·m)**. Matrix-vector multiplications.
  - **Space**: **O(n)**. Creates new state vector x.
- **`reset()`**:
  - **Time**: **O(n)**. Zeroes out the state vector.
  - **Space**: **O(1)**. (In-place modification or simple reassignment).

#### `NonlinearSolver` (Class)

- **`__init__(dynamics_func, ...)`**:
  - **Time**: **O(1)**. Initializes Butcher tableau constants.
  - **Space**: **O(1)**. Stores fixed-size coefficients (7x7 arrays).
- **`solve_adaptive(t_end, x0, u_func)`**:
  - **Time**: **O(N_adapt · (n + C_f))**. Performs 6 evaluations of f(x) per step.
  - **Space**: **O(N_adapt · n)**. Stores full state history lists `t_hist` and `x_hist`.

### 1.2. State Space (`core/state_space.py`)

#### `StateSpace` (Class)

- **`__init__(A, B, C, D)`**:
  - **Time**: **O(1)**. Dimension checks and assignment.
  - **Space**: **O(n² + n·m + p·n + p·m)**. Stores system matrices.
- **`get_frequency_response(omega_range, ...)`**:
  - **Time**: **O(N_freq · n³)**. Solves linear system (sI - A)x = B at each point.
  - **Space**: **O(N_freq + n²)**. Stores magnitude/phase arrays and intermediate matrices.

### 1.3. Model Predictive Control (`core/mpc.py`)

#### `ModelPredictiveControl` (Class)

- **`__init__(...)`**:
  - **Time**: **O(n²)**. Matrix copying and setup.
  - **Space**: **O(n²)**. Stores Q, R, A, B.
- **`_setup_admm()`**:
  - **Time**: **O(H · n³)**. Matrix powers and products for condensation.
  - **Space**: **O((H·n) · (H·m))**. Stores large condensed matrix `S_u`.
- **`_solve_admm(x_current, x_ref, iterations)`**:
  - **Time**: **O(I · (H·m)²)**. Matrix-vector math on horizon size.
  - **Space**: **O(H·m)**. Vector storage for z, u, q.
- **`_get_derivatives(x, u)`**:
  - **Time**: **O((n+m) · C_f)**. Finite difference Jacobian.
  - **Space**: **O(n·(n+m))**. Stores linearized A and B matrices.
- **`_solve_ilqr(x_current, x_ref, iterations)`**:
  - **Time**: **O(I · H · (n+m)³)**. Riccati recursion involves matrix inversions.
  - **Space**: **O(H · (n+m)²)**. Stores gain matrices K and k for entire horizon.
- **`optimize(x_current, x_ref, ...)`**:
  - **Time**: **O(1)** overhead + Solver Time (see above).
  - **Space**: **O(1)** overhead + Solver Space.

### 1.4. Control Utils (`core/control_utils.py`)

#### `solve_discrete_riccati(A, B, Q, R, ...)`

- **Time**: **O(I · n³)**. Matrix multiplications and inversion in loop.
- **Space**: **O(n²)**. Stores P and P_next.

#### `dlqr(A, B, Q, R)`

- **Time**: **O(I · n³)**. Calls Riccati solver.
- **Space**: **O(n²)**. Stores gain matrix K.

#### `PIDController` (Class)

- **`__init__(...)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`reset()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`update(measurement, setpoint, dt)`**:
  - **Time**: **O(1)**. Scalar arithmetic.
  - **Space**: **O(1)**.

### 1.5. Math Utils (`core/math_utils.py`)

#### `implicit_mul(expr)`

- **Time**: **O(L)**. Regex substitution.
- **Space**: **O(L)**. Creates new string.

#### `preprocess_power(expr)`

- **Time**: **O(L)**. Regex substitution.
- **Space**: **O(L)**. Creates new string.

#### `make_func(expr_string)`

- **Time**: **O(1)** (Compilation). Execution **O(1)**.
- **Space**: **O(1)** (Function object).

#### `make_system_func(expr_string)`

- **Time**: **O(1)** (Compilation). Execution **O(1)** (assuming scalar math).
- **Space**: **O(1)**.

#### `Differentiation` (Class)

- **`real_diff(func, point)`**:
  - **Time**: **O(C_f)**. Function evaluation.
  - **Space**: **O(1)**.

#### `Root` (Class)

- **`brent_root(...)`**:
  - **Time**: **O(I · C_f)**.
  - **Space**: **O(1)**.
- **`newton_root(...)`**:
  - **Time**: **O(I · C_f)**.
  - **Space**: **O(1)**.
- **`find_root(...)`**:
  - **Time**: **O(I · C_f)**.
  - **Space**: **O(1)**.

### 1.6. Extended Kalman Filter (`core/ekf.py`)

#### `ExtendedKalmanFilter` (Class)

- **`__init__(...)`**:
  - **Time**: **O(n²)**. Initialization of P, Q, R.
  - **Space**: **O(n²)**.
- **`compute_jacobian(func, x, ...)`**:
  - **Time**: **O(n · C_f)**. Complex step perturbation.
  - **Space**: **O(n)**. Perturbed state vector.
- **`predict(u, dt)`**:
  - **Time**: **O(n³ + n · C_f)**. Jacobian + P update.
  - **Space**: **O(n²)**. Intermediate matrices (F).
- **`update(y_meas)`**:
  - **Time**: **O(n³ + m³ + n · C_h)**. Gain calculation + inversion.
  - **Space**: **O(n² + m²)**. Intermediate matrices (S, K).

### 1.7. Unscented Kalman Filter (`core/ukf.py`)

#### `UnscentedKalmanFilter` (Class)

- **`__init__(...)`**:
  - **Time**: **O(n)**. Weights calculation.
  - **Space**: **O(n²)**. P, Q, R matrices.
- **`_compute_weights()`**:
  - **Time**: **O(n)**.
  - **Space**: **O(n)**.
- **`_generate_sigma_points(x, P)`**:
  - **Time**: **O(n³)**. Cholesky decomposition.
  - **Space**: **O(n²)**. Sigma points array (2n+1, n).
- **`predict(u, dt)`**:
  - **Time**: **O(n · C_f + n²)**. Propagates S points.
  - **Space**: **O(n²)**. Sigma points storage.
- **`update(z)`**:
  - **Time**: **O(n · C_h + n³)**. Inversion of S (mxm) and P update.
  - **Space**: **O(n² + n·m)**. Cross-covariance Pxz.

### 1.8. Transfer Function (`core/transfer_function.py`)

#### `TransferFunction` (Class)

- **`__init__(num, den)`**:
  - **Time**: **O(d)**. Array creation.
  - **Space**: **O(d)**. Coefficients storage.
- **`__repr__()`**:
  - **Time**: **O(d)**. String formatting.
  - **Space**: **O(d)**. String output.
- **`evaluate(s)`**:
  - **Time**: **O(d)**. Horner's method.
  - **Space**: **O(1)**.
- **`bode_response(omega_range)`**:
  - **Time**: **O(N_freq · d)**.
  - **Space**: **O(N_freq)**. Magnitude and phase arrays.
- **`to_state_space()`**:
  - **Time**: **O(d)**. Copying values.
  - **Space**: **O(d²)**. A, B, C, D matrices (A is dxd).

### 1.9. Analysis (`core/analysis.py`)

#### `get_stability_margins(tf, ...)`

- **Time**: **O(N_freq · d + I · d)**. Grid scan + Root finding.
- **Space**: **O(N_freq)**. Frequency grid storage.

#### `get_exact_time_idx(time, response, target_val)`

- **Time**: **O(N)**. Linear scan.
- **Space**: **O(1)**.

#### `get_step_metrics(time, response)`

- **Time**: **O(N)**. Linear scan + multiple calls to `get_exact_time_idx`.
- **Space**: **O(1)**.

### 1.10. Estimator (`core/estimator.py`)

#### `KalmanFilter` (Class)

- **`__init__(...)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n²)**.
- **`update(u, y_meas)`**:
  - **Time**: **O(n³ + m³)**. Matrix multiplication and inversion.
  - **Space**: **O(n²)**. Intermediate matrices.

### 1.11. Exceptions (`core/exceptions.py`)

Includes `PyControlsError`, `DimensionMismatchError`, `SingularMatrixError`, `ConvergenceError`, `UnstableSystemError`, `InvalidParameterError`, `SolverError`, `ControllerConfigError`.

- **Time**: **O(1)** (Class definitions).
- **Space**: **O(1)**.

### 1.12. Initialization (`core/__init__.py`)

- **Time**: **O(1)**.
- **Space**: **O(1)**.

---

## 2. Systems Modules

### 2.1. DC Motor (`systems/dc_motor.py`)

#### `DCMotor` (Class)

- **`__init__(...)`**:
  - **Time**: **O(1)**. Dictionary assignment.
  - **Space**: **O(1)**.
- **`get_open_loop_tf`, `get_closed_loop_tf`, `get_disturbance_tf`**:
  - **Time**: **O(d²)**. Polynomial convolution (multiplication).
  - **Space**: **O(d)**. Resulting coefficients.
- **`get_state_space()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n²)**. Matrix allocation.
- **`get_augmented_state_space()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n_aug²)**.
- **`get_parameter_estimation_func()`**:
  - **Time**: **O(1)** (Returns closure).
  - **Space**: **O(1)**.
- **`get_nonlinear_dynamics()`**:
  - **Time**: **O(1)** (Returns closure).
  - **Space**: **O(1)**.
- **`get_mpc_model(dt)`**:
  - **Time**: **O((n+m)³)**. Calls `manual_matrix_exp`.
  - **Space**: **O((n+m)²)**.

### 2.2. Inverted Pendulum (`systems/pendulum.py`)

#### `InvertedPendulum` (Class)

- **`__init__(...)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n²)**.
- **`_linear_matrices()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n²)**.
- **`get_parameter_estimation_func()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`get_state_space()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n²)**.
- **`get_augmented_state_space()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n_aug²)**.
- **`dlqr_gain(dt)`**:
  - **Time**: **O(I · n³)**. Calls `dlqr`.
  - **Space**: **O(n²)**.
- **`get_open_loop_tf(K)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)** (Returns wrapper object).
- **`measurement(x)`, `measurement_jacobian(x)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n·p)** (Jacobian).
- **`dynamics(...)`, `dynamics_continuous(...)`**:
  - **Time**: **O(C_f)**.
  - **Space**: **O(n)**.
- **`get_nonlinear_dynamics()`, `get_mpc_model(dt)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.

#### `LQRLoopTransferFunction` (Class)

- **`__init__(A, B, K)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n²)**. Ref storage.
- **`evaluate(s)`**:
  - **Time**: **O(n³)**. Linear solve.
  - **Space**: **O(n²)**. Intermediate computation.

### 2.3. Package Initialization (`systems/__init__.py`)

- **Time**: **O(1)**.
- **Space**: **O(1)**.

---

## 3. Modules Package

### 3.1. Physics Engine (`modules/physics_engine.py`)

#### `dc_motor_dynamics(...)`

- **Time**: **O(1)**.
- **Space**: **O(1)**. (O(n) for state return).

#### `pendulum_dynamics(...)`

- **Time**: **O(1)**. Trig math.
- **Space**: **O(1)**. (O(n) for state return).

#### `rk4_fixed_step(...)`

- **Time**: **4 · C_f**. 4 calls to dynamics.
- **Space**: **O(n)**. Intermediate k vectors.

### 3.2. Interactive Lab (`modules/interactive_lab.py`)

#### `InteractiveLab` (Class)

- **`__init__(...)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`initialize()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(n)**. State vector allocation.
- **`step(disturbance)`**:
  - **Time**: **O(C_f + Cost_Estimator)**.
  - **Space**: **O(n)**.
- **`reset()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`evaluate_rules()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`get_control_input()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`set_manual_input()`, `set_auto_controller()`, `set_manual_mode()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`handle_keyboard_input()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`init_visualization()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(N_history)**. Matplotlib lists.
- **`update_visualization()`**:
  - **Time**: **O(N_history)**. Plot redraw.
  - **Space**: **O(N_history)**. Appending points.
- **`set_estimator(...)`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.

#### `simple_dc_motor_pid(...)`, `pendulum_lqr_controller(...)`

- **Time**: **O(1)** (Factory creation). Closure execution **O(1)**.
- **Space**: **O(1)**.

### 3.3. Package Initialization (`modules/__init__.py`)

- **Time**: **O(1)**.
- **Space**: **O(1)**.

---

## 4. Application Logic

### 4.1. Main Application (`main.py`)

#### `load_available_systems()`

- **Time**: **O(Files · Classes)**. Dynamic import scan.
- **Space**: **O(Classes)**. Storage of class references.

#### `PyControlsApp` (Class)

- **`__init__()`**:
  - **Time**: **O(1)**. System instantiation.
  - **Space**: **O(1)**.
- **`clear_screen()`, `print_header()`**:
  - **Time**: **O(1)**.
  - **Space**: **O(1)**.
- **`main_menu()`**:
  - **Time**: User-dependent loop.
  - **Space**: **O(1)**.
- **`simulate_preset_system(system_instance, ctrl_config)`**:
  - **Time**: **O(N · (n³ + Cost_Control))**. Steps through ExactSolver and KF.
  - **Space**: **O(N · n)**. Stores history arrays (y_real_hist, x_est_hist).
- **`run_preset_dashboard()`**:
  - **Time**: **O(N · n³)**. Runs simulation.
  - **Space**: **O(N · n)**.
- **`run_analysis_dashboard()`**:
  - **Time**: **O(N_freq · n³)**.
  - **Space**: **O(N_freq)**.
- **`edit_params_menu()`, `edit_disturbance_menu()`, `switch_system_menu()`**:
  - **Time**: **O(1)**. Interaction.
  - **Space**: **O(1)**.
- **`run_ekf()`**:
  - **Time**: **O(N · (n³ + n·C_f))**.
  - **Space**: **O(N · n)**.
- **`run_ukf()`**:
  - **Time**: **O(N · (n³ + n·C_f))**.
  - **Space**: **O(N · n)**.
- **`run_mpc()`**:
  - **Time**: **O(N_sim · Cost_MPC_Optimize)**.
  - **Space**: **O(N_sim · n)**.
- **`run_interactive_lab()`**:
  - **Time**: Real-time loop.
  - **Space**: **O(N_history)**.
- **`run_custom_simulation()`**:
  - **Time**: **O(N_adapt · C_f)**.
  - **Space**: **O(N_adapt · n)**.

#### Global execution block (`if __name__ == "__main__":`)

- **Time**: Dependent on user session length.
- **Space**: **O(N · n)** (Max memory usage during simulation).

### 4.2. Configuration (`config.py`)

Contains dictionaries (`MOTOR_PARAMS`, `SIM_PARAMS`, etc.).

- **Time**: **O(1)**.
- **Space**: **O(1)**. (Constant size data structures).

### 4.3. Exit Utils (`exit.py`)

#### `flush()`, `kill()`, `stop()`

- **Time**: **O(1)**. System calls.
- **Space**: **O(1)**.

### 4.4. System Registry (`system_registry.py`)

#### `SystemDescriptor` (Class)

- **Time**: **O(1)**.
- **Space**: **O(1)**.

#### `SYSTEM_REGISTRY` (Dict)

- **Time**: **O(1)**.
- **Space**: **O(1)**.

### 4.5. Test Runner (`test_runner.py`)

#### `run_tests()`

- **Time**: **O(Σ Cost_Test_i)**.
- **Space**: **O(1)** overhead.

---
