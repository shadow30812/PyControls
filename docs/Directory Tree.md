# PyControls — Directory Tree & Architecture Structure

```text
PyControls/
├── benchmark_runner.py                # Master CLI benchmark orchestrator for running test suites
├── extra_reqs.txt                     # Optional development/benchmarking dependencies
├── final_success_plot_dc_motor.png    # DC Motor simulation validation plot
├── final_success_plot_pendulum.png    # Inverted pendulum simulation validation plot
├── main.py                            # Interactive CLI entrypoint & simulation workbench
├── pyproject.toml                     # Package configuration and build metadata
├── requirements.txt                   # Core runtime dependencies (numpy, scipy, numba, matplotlib)
├── script.py                          # Scratch/automation script
├── test_runner.py                     # Master test execution entrypoint
├── uv.lock                            # UV package manager locked dependency tree
│
├── benchmarking/                      # Comprehensive empirical benchmarking & profiling suite
│   ├── __init__.py
│   ├── benchmark_csd.py               # Complex-Step Differentiation vs analytical Jacobians
│   ├── benchmark_dare.py              # Discrete Algebraic Riccati Equation solver convergence
│   ├── benchmark_ekf_ukf_timing.py    # Predict-Update step latency profiling for EKF/UKF
│   ├── benchmark_expm.py              # Matrix exponential accuracy vs SciPy Padé approximant
│   ├── benchmark_jit.py               # Numba JIT acceleration vs pure-Python benchmarks
│   ├── benchmark_mpc.py               # ADMM & iLQR solve times across prediction horizons
│   ├── benchmark_mpc_optimality.py    # MPC asymptotic convergence to infinite-horizon LQR
│   ├── benchmark_ode.py               # Adaptive RK5(4)7M Dormand-Prince integrator accuracy
│   ├── benchmark_scalability.py       # Asymptotic scaling study (O(n^p)) up to dim n=50
│   ├── benchmark_stability_margins.py # Frequency-domain Gain & Phase Margin robustness
│   ├── control_metrics.py             # Step response & disturbance rejection metrics (IAE/ITAE)
│   └── estimation_metrics.py          # State/parameter estimation convergence & SNR attenuation
│
├── core/                              # First-principles numerical and control algorithms
│   ├── __init__.py
│   ├── analysis.py                    # Controllability, observability, Bode, and pole analysis
│   ├── base.py                        # Base classes for states, controllers, and models
│   ├── configs.py                     # Global configuration models and default settings
│   ├── control_utils.py               # DARE solver, DLQR synthesizer, and PID controllers
│   ├── data_logger.py                 # Structured time-series simulation data recorder
│   ├── ekf.py                         # Continuous-discrete Extended Kalman Filter (CSD-based)
│   ├── ekf_discrete.py                # Pure discrete-time Extended Kalman Filter
│   ├── estimator.py                   # High-level state estimation abstractions
│   ├── exceptions.py                  # Custom domain exceptions and numerical error handlers
│   ├── hooks.py                       # Extensible lifecycle and step hooks
│   ├── math_utils.py                  # Complex-Step Differentiation and parsing helpers
│   ├── mpc.py                         # Model Predictive Control (ADMM QP & iLQR solvers)
│   ├── profiler.py                    # Execution timing and memory profiler utilities
│   ├── solver.py                      # Exact ZOH matrix exp & Adaptive RK5(4)7M integrators
│   ├── state_space.py                 # LTI/LTV Continuous & Discrete State-Space models
│   ├── transfer_function.py           # SISO & MIMO rational Transfer Function representations
│   └── ukf.py                         # Scaled Unscented Kalman Filter (deterministic sigma points)
│
├── docs/                              # Technical guides, math derivations, and benchmarks
│   ├── Benchmarks Guide.md            # Detailed profiling guide, empirical data, and CV points
│   ├── Complexity Analysis.md         # Rigorous asymptotic time/space complexity analysis
│   ├── Equations and Formulae.md      # Full mathematical derivations for all core modules
│   ├── LICENSE                        # MIT Open Source License
│   └── README.md                      # Comprehensive project documentation and quickstart
│
├── helpers/                           # Auxiliary utilities and CLI helpers
│   ├── config.py                      # Configuration loaders and environment parsers
│   ├── exit.py                        # Graceful shutdown and signal handling
│   ├── plot.py                        # Matplotlib plotting routines and visualization helpers
│   ├── simulation_runner.py           # Batch simulation harness and test runners
│   └── system_registry.py             # Dynamic system model factory and registry
│
├── HIL_Heater_Firmware/               # Embedded firmware for thermal HIL control
│   ├── platformio.ini                 # PlatformIO project configuration (Arduino/C++)
│   ├── Project Report/
│   │   ├── Figure.png                 # Thermal tracking experimental response curve
│   │   └── Heater Project Report.md   # HIL thermal test documentation and results
│   ├── include/
│   │   └── README
│   ├── lib/
│   │   └── README
│   ├── src/
│   │   └── main.cpp                   # C++ firmware for thermistor ADC reading & PWM drive
│   └── test/
│       └── README
│
├── HIL_PWM_Firmware/                  # Embedded firmware for PWM-DAC voltage HIL control
│   ├── platformio.ini                 # PlatformIO project configuration (Arduino/C++)
│   ├── Project Report/
│   │   ├── Figure 1.png               # Step response and linearity analysis
│   │   ├── Figure 2.png               # Frequency response under PWM excitation
│   │   ├── Figure 3.png               # Filtered vs unfiltered ripple waveform
│   │   ├── Figure 4.png               # Closed-loop tracking response
│   │   ├── Figure 5.png               # Disturbance rejection test
│   │   ├── Figure 6.png               # Multi-phase HIL progression
│   │   └── PWM Project Report.md      # 6-phase systematic HIL testing report
│   ├── include/
│   │   └── README
│   ├── lib/
│   │   └── README
│   ├── src/
│   │   └── main.cpp                   # C++ firmware with ASCII protocol & watchdog safety
│   └── test/
│       └── README
│
├── interfaces/                        # External integration bridges and interfaces
│   ├── __init__.py
│   ├── manager.py                     # Multi-interface lifecycle manager
│   └── matlab/                        # MATLAB / Simulink bidirectional bridge
│       ├── __init__.py
│       └── bridge.py                  # MATLAB Engine API wrapper for co-simulation
│
├── MATLAB/                            # MATLAB integration planning and feasibility notes
│   ├── feasibility_study.md           # Architectural study for MATLAB/Simulink bridge
│   ├── implementation_plan.md         # Co-simulation and MEX execution plan
│   └── matplan.md                     # Roadmap and timeline
│
├── modules/                           # Interactive experimentation modules
│   ├── __init__.py
│   ├── interactive_lab.py             # Real-time interactive control parameter tuning lab
│   └── physics_engine.py              # Discrete physics simulation backend
│
├── systems/                           # Physical dynamic plant models
│   ├── __init__.py
│   ├── battery.py                     # Equivalent circuit lithium-ion battery model
│   ├── dc_motor.py                    # Electromechanical DC motor with back-EMF dynamics
│   ├── pendulum.py                    # Nonlinear inverted pendulum on a cart
│   └── thermistor.py                  # First-order thermal RC heating system
│
└── tests/                             # Comprehensive unit & integration test suite
    ├── test_analysis.py               # Controllability, observability & Bode tests
    ├── test_base_classes.py           # State, model, and controller abstraction tests
    ├── test_control.py                # LQR, PID, and Riccati solver unit tests
    ├── test_jit_vec.py                # Numba JIT acceleration and vectorization tests
    ├── test_kf_ekf.py                 # EKF state estimation and CSD Jacobian tests
    ├── test_lifecycle.py              # Simulation initialization and reset lifecycle tests
    ├── test_manager.py                # System and interface manager tests
    ├── test_math.py                   # Complex-Step Differentiation and parsing tests
    ├── test_matlab_bridge.py          # MATLAB bridge and mock integration tests
    ├── test_models.py                 # State-space and transfer function model tests
    ├── test_modules.py                # Physics engine and interactive lab tests
    ├── test_profiler_logger.py        # Profiling and data logger verification tests
    ├── test_serialization.py          # Config export and state serialization tests
    ├── test_solver.py                 # Adaptive RK5(4)7M and exact ZOH solver tests
    ├── test_systems.py                # DC Motor, Pendulum, Battery & Thermal model tests
    └── test_ukf_mpc.py                # ADMM MPC, iLQR, and UKF unit tests
```
