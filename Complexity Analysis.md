# PyControls Algorithm Complexity Analysis

This document details the theoretical Time and Space complexities for the algorithms implemented in the PyControls library.

## Variable Definitions

- **n**: Number of states in the system (State Dimension)
- **m**: Number of inputs or measurements
- **N**: Number of simulation time steps
- **d**: Degree of Transfer Function polynomials
- **k**: Taylor series order (fixed at 20)
- **C_f**: Computational cost of evaluating the dynamics function f(x, u)

## 1. Solvers (core/solver.py)

### 1.1. Matrix Exponential (manual_matrix_exp)

Used to discretize continuous State-Space matrices. Implements Scaling and Squaring with Taylor Series.

**Time Complexity:**

- **Logic**: Matrix multiplication is O(n³). The Taylor series runs for k iterations. The scaling/squaring step runs logarithmic to the matrix norm.
- **Effective Complexity**: O(n³ · (k + log(∥A∥))) since k is constant and k is typically small (< 10) in this project.

**Space Complexity:**

- **Logic**: Stores matrices of size n × n.
- **Complexity**: O(n²)

### 1.2. Exact Solver Initialization (ExactSolver.**init**)

Calculates discrete matrices Φ and Γ using the Matrix Exponential on a block matrix.

**Time Complexity:**

- **Logic**: Performs matrix exponential on a block matrix of size (n + m) × (n + m).
- **Complexity**: O((n + m)³)

**Space Complexity:**

- **Complexity**: O((n + m)²)

### 1.3. Exact Solver Step (ExactSolver.step)

Advances the linear system by one discrete step: x_{k+1} = Φx_k + Γu_k

**Time Complexity:**

- **Logic**: Matrix-vector multiplication.
- **Complexity**: O(n² + n·m)

**Space Complexity:**

- **Logic**: to store Φ and Γ.
- **Complexity**: O(n²)

### 1.4. Adaptive RK45 (NonlinearSolver.solve_adaptive)

Solves non-linear ODEs using the Dormand-Prince method with adaptive step sizing.

**Time Complexity:**

- **Logic**: In each step, it performs 6 function evaluations (6·C_f) and vector operations (O(n)) to combine the results (Butcher tableau). N_adapt depends on system stiffness and tolerance.
- **Complexity**: O(N_adapt · (n + C_f))

**Space Complexity:**

- **Logic**: Stores the full state history for every successful time step.
- **Complexity**: O(N_adapt · n)

## 2. Estimation (core/estimator.py, core/ekf.py)

### 2.1. Linear Kalman Filter Update (KalmanFilter.update)

Performs Prediction and Correction for linear systems.

**Time Complexity:**

- **Logic**: Dominated by matrix multiplications (PΦT, CPCT) and matrix inversion (S^{-1} where S is m × m).
- **Complexity**: O(n³ + m³)

**Space Complexity:**

- **Logic**: Stores Covariance Matrix P (n × n).
- **Complexity**: O(n²)

### 2.2. Extended Kalman Filter Jacobian (EKF.compute_jacobian)

Computes partial derivatives using Complex Step Differentiation.

**Time Complexity:**

- **Logic**: Must perturb each of the n state variables independently and evaluate the function f.
- **Complexity**: O(n · C_f)

**Space Complexity:**

- **Complexity**: O(n) (Transient storage for perturbed state vector)

### 2.3. Extended Kalman Filter Step (EKF.predict + EKF.update)

Non-linear estimation step.

**Time Complexity:**

- **Logic**:
  1. Compute Jacobian F: O(n · C_f)
  2. Predict Covariance (FPF^T): O(n³)
  3. Compute Jacobian H: O(n · C_h)
  4. Kalman Gain (PH^T S^{-1}): O(n³)
- **Complexity**: O(n³ + n·C_f + n·C_h)

**Space Complexity:**

- **Complexity**: O(n²)

## 3. Analysis (core/analysis.py, core/transfer_function.py)

### 3.1. Frequency Response Evaluation (StateSpace.get_frequency_response)

Computes H(jω) = C(sI - A)^{-1}B + D for a range of frequencies.

**Time Complexity:**

- **Logic**: For each frequency point, it solves a linear system (sI - A)x = B, which is O(n³) (or O(n²) if factorized, but np.linalg.solve is roughly cubic).
- **Complexity**: O(N_freq · n³)

**Space Complexity:**

- **Complexity**: O(N_freq + n²)

### 3.2. Bode Response (TransferFunction.bode_response)

Evaluates polynomial transfer function G(s) for a range of frequencies.

**Time Complexity:**

- **Logic**: Horner's method (np.polyval) takes O(d) per evaluation.
- **Complexity**: O(N_freq · d)

**Space Complexity:**

- **Complexity**: O(d)

### 3.3. Stability Margins (get_stability_margins)

Finds Gain and Phase margins using root finding on the Bode response.

**Time Complexity:**

- **Logic**:
  1. Scans a frequency grid (Grid ≈ 100)
  2. If sign change detected, calls Root Finder (I_root ≈ 100 iterations)
  3. Function evaluation cost is O(d)
- **Effective Complexity**: Very low, as d is usually small (2-5)
- **Complexity**: O(Grid · I_root · d)

### 3.4. Step Metrics (get_step_metrics)

Calculates Rise Time, Overshoot, etc.

**Time Complexity:**

- **Logic**: Linear scan through the simulation response array of size N.
- **Complexity**: O(N)

## 4. Math Utilities (core/math_utils.py)

### 4.1. Root Finding (Root.brent_root)

Iterative root finding combining Bisection, Secant, and Inverse Quadratic Interpolation.

**Time Complexity:**

- **Logic**: Converges superlinearly. I is max iterations (capped at 100).
- **Complexity**: O(I · C_f)

**Space Complexity:**

- **Complexity**: O(1)

### 4.2. Root Finding (Root.newton_root)

Newton-Raphson method with numerical differentiation.

**Time Complexity:**

- **Logic**: Quadratic convergence. Requires 2 evaluations per step (value + derivative).
- **Complexity**: O(I · C_f)

## 5. Main Simulation Loops (main.py)

### 5.1. Standard Simulation (Option 1)

Simulates Linear System + Linear Kalman Filter.

**Total Time Complexity:**

- Breakdown: N steps of ExactSolver (n²) + Kalman Filter (n³).
- **Complexity**: O(N · n³)

**Total Space Complexity:**

- **Complexity**: O(N · n) (to store history)

### 5.2. Parameter Estimation Demo (Option 6)

Simulates Linear System (Reality) + EKF (Estimator).

**Total Time Complexity:**

- **Logic**:
  - True System Step: O(n²)
  - EKF Step: O(n_aug³ + n_aug · C_f) where n_aug is the augmented state size (4 in this case: Speed, Current, J, b)
- **Effective**: Since n = 2 and n_aug = 4, this runs very fast in real-time, effectively O(N)
- **Complexity**: O(N · (n_aug³ + n_aug · C_f))
