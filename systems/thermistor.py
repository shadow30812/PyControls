import math
import time
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
import serial

from helpers.config import THERMISTOR_PARAMS


class Thermistor:
    """
    Physical HIL Adapter for Thermistor Control.
    Handles Non-Linear Sensor Calibration and Arduino Serial Communication.
    """

    def __init__(self, **kwargs: Any) -> None:
        self.params: Dict[str, Any] = THERMISTOR_PARAMS.copy()
        if kwargs:
            self.params.update(kwargs)
        self.arduino: Optional[serial.Serial] = None
        self.current_temp: float = 25.0

    def connect(self) -> None:
        try:
            self.arduino = serial.Serial(
                self.params["port"], self.params["baud"], timeout=0.1
            )
            time.sleep(2)
            self.arduino.reset_input_buffer()
            print(f"HIL: Connected to Arduino on {self.params['port']}")
            self.write_pwm(0)
        except Exception as e:
            print(f"HIL Error: Connection failed. {e}")
            self.arduino = None

    def close(self) -> None:
        if self.arduino and self.arduino.is_open:
            self.write_pwm(0)
            self.arduino.close()
            print("HIL: Connection closed.")

    def write_pwm(self, u: float) -> None:
        if self.arduino and self.arduino.is_open:
            pwm: int = int(np.clip(u, 0, 255))
            self.arduino.write(f"Q:{pwm}\n".encode())

    def read_temp(self) -> float:
        if not self.arduino or not self.arduino.is_open:
            return self.current_temp

        while self.arduino.in_waiting:
            try:
                line: str = self.arduino.readline().decode().strip()
                if line.startswith("A:"):
                    raw: int = int(line.split(":")[1])
                    self.current_temp = self._adc_to_celsius(raw)
                    print(f"HIL Debug: raw ADC = {raw}")

            except Exception as e:
                print("Error in thermistor", e, sep="\n")
                pass

        return self.current_temp

    def _adc_to_celsius(self, adc_val: Union[int, float, str]) -> float:
        try:
            adc_val = int(adc_val)
        except Exception:
            return self.current_temp

        if adc_val <= 0 or adc_val >= 1023:
            print(
                f"HIL Debug: ADC out-of-range: {adc_val} -> keeping last temp {self.current_temp:.2f}°C"
            )
            return self.current_temp

        r_th: float = self.params["R_divider"] * ((1023.0 / adc_val) - 1.0)
        if r_th <= 0:
            print(f"HIL Debug: computed non-positive R_th {r_th} from ADC {adc_val}")
            return self.current_temp

        try:
            inv_T: float = (1.0 / self.params["T0"]) + (
                1.0 / self.params["Beta"]
            ) * math.log(r_th / self.params["R0"])
            return (1.0 / inv_T) - 273.15
        except Exception as e:
            print("HIL Error: _adc_to_celsius calculation failed:", e)
            return self.current_temp

    def get_state_space(self) -> None:
        return None

    def get_disturbance_tf(self, *args: Tuple[Any, ...]) -> None:
        return None
