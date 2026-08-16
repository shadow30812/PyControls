import importlib
import inspect
import os
import pkgutil
from types import ModuleType
from typing import Any, Dict, Final, List, Type

from systems.battery import Battery
from systems.dc_motor import DCMotor
from systems.pendulum import InvertedPendulum
from systems.thermistor import Thermistor


class SystemDescriptor:
    def __init__(
        self,
        system_id: str,
        display_name: str,
        system_class: Type[Any],
        input_type: str,
        state_labels: List[str],
        supports_analysis: bool,
        supports_estimation: bool,
        supports_mpc: bool,
        supports_interactive_lab: bool,
        is_hardware: bool = False,
    ) -> None:
        self.system_id: str = system_id
        self.display_name: Final[str] = display_name
        self.system_class: Final[Type[Any]] = system_class
        self.input_type: Final[str] = input_type
        self.state_labels: Final[List[str]] = state_labels
        self.supports_analysis: Final[bool] = supports_analysis
        self.supports_estimation: Final[bool] = supports_estimation
        self.supports_mpc: Final[bool] = supports_mpc
        self.supports_interactive_lab: Final[bool] = supports_interactive_lab
        self.is_hardware: bool = is_hardware


SYSTEM_REGISTRY: Dict[str, SystemDescriptor] = {
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


def load_available_systems() -> Dict[str, Type[Any]]:
    systems: Dict[str, Type[Any]] = {}
    systems_path: str = os.path.join(os.getcwd(), "systems")

    for _, name, _ in pkgutil.iter_modules([systems_path]):
        module_name: str = f"systems.{name}"
        try:
            module: ModuleType = importlib.import_module(module_name)
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
