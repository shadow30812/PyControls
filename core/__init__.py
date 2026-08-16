"""
PyControls Core — Algorithmic kernel for estimation, control, and simulation.

Thread Safety
-------------
All stateful classes (KalmanFilter, ExtendedKalmanFilter, UnscentedKalmanFilter,
PIDController, ModelPredictiveControl, ExactSolver) are **NOT thread-safe** when
sharing a single instance across threads. Each thread must use its own instance.

The ControllerManager in interfaces/manager.py is also not thread-safe by default.
For multi-threaded use, either:
  1. Use one ControllerManager per thread, or
  2. Wrap manager calls with a threading.Lock.

Stateless classes (TransferFunction, StateSpace, Check, Root, Differentiation)
are safe to share across threads.
"""
