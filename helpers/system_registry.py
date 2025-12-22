import importlib
import inspect
import os
import pkgutil

from systems.battery import Battery
from systems.dc_motor import DCMotor
from systems.pendulum import InvertedPendulum
from systems.thermistor import Thermistor


class SystemDescriptor:
    def __init__(
        self,
        system_id,
        display_name,
        system_class,
        input_type,
        state_labels,
        supports_analysis,
        supports_estimation,
        supports_mpc,
        supports_interactive_lab,
        is_hardware=False,
    ):
        self.system_id = system_id
        self.display_name = display_name
        self.system_class = system_class
        self.input_type = input_type
        self.state_labels = state_labels
        self.supports_analysis = supports_analysis
        self.supports_estimation = supports_estimation
        self.supports_mpc = supports_mpc
        self.supports_interactive_lab = supports_interactive_lab
        self.is_hardware = is_hardware


SYSTEM_REGISTRY = {
    "dc_motor": SystemDescriptor(
        system_id="dc_motor",
        display_name="DC Motor",
        system_class=DCMotor,
        input_type="voltage",
        state_labels=["ω (rad/s)", "i (A)"],
        supports_analysis=True,
        supports_estimation=True,
        supports_mpc=True,
        supports_interactive_lab=True,
    ),
    "pendulum": SystemDescriptor(
        system_id="pendulum",
        display_name="Inverted Pendulum",
        system_class=InvertedPendulum,
        input_type="torque",
        state_labels=["θ (rad)", "ω (rad/s)"],
        supports_analysis=True,
        supports_estimation=True,
        supports_mpc=True,
        supports_interactive_lab=True,
    ),
    "thermistor": SystemDescriptor(
        system_id="thermistor",
        display_name="Thermistor HIL Control",
        system_class=Thermistor,
        input_type="PWM",
        state_labels=["Temp (°C)"],
        supports_analysis=False,
        supports_estimation=False,
        supports_mpc=False,
        supports_interactive_lab=True,
        is_hardware=True,
    ),
    "battery": SystemDescriptor(
        system_id="battery",
        display_name="HIL Battery Source",
        system_class=Battery,
        input_type="PWM",
        state_labels=["Voltage (V)"],
        supports_analysis=False,
        supports_estimation=False,
        supports_mpc=False,
        supports_interactive_lab=True,
        is_hardware=True,
    ),
}


def load_available_systems():
    systems = {}
    systems_path = os.path.join(os.getcwd(), "systems")

    for _, name, _ in pkgutil.iter_modules([systems_path]):
        module_name = f"systems.{name}"
        try:
            module = importlib.import_module(module_name)
            for member_name, member_obj in inspect.getmembers(module, inspect.isclass):
                if (
                    (
                        hasattr(member_obj, "get_closed_loop_tf")
                        and hasattr(member_obj, "get_disturbance_tf")
                    )
                    or member_name in ("thermistor", "battery")
                ) and member_obj.__module__ == module_name:
                    systems[member_name] = member_obj
        except Exception as e:
            print(f"Warning: Could not load system '{name}': {e}")

    return systems
