import math
import time

import numpy as np
import serial

from helpers.config import THERMISTOR_PARAMS


class Thermistor:
    """
    Physical HIL Adapter for Thermistor Control.
    Handles Non-Linear Sensor Calibration and Arduino Serial Communication.
    """

    def __init__(self, **kwargs):
        self.params = THERMISTOR_PARAMS.copy()
        if kwargs:
            self.params.update(kwargs)
        self.arduino = None
        self.current_temp = 25.0

    def connect(self):
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

    def close(self):
        if self.arduino and self.arduino.is_open:
            self.write_pwm(0)
            self.arduino.close()
            print("HIL: Connection closed.")

    def write_pwm(self, u):
        if self.arduino and self.arduino.is_open:
            pwm = int(np.clip(u, 0, 255))
            self.arduino.write(f"Q:{pwm}\n".encode())

    def read_temp(self):
        if not self.arduino or not self.arduino.is_open:
            return self.current_temp

        while self.arduino.in_waiting:
            try:
                line = self.arduino.readline().decode().strip()
                if line.startswith("A:"):
                    self.current_temp = self._adc_to_celsius(int(line.split(":")[1]))
            except Exception:
                pass

        return self.current_temp

    def _adc_to_celsius(self, adc_val):
        if adc_val <= 0 or adc_val >= 1023:
            return 25.0
        r_th = self.params["R_divider"] * ((1023.0 / adc_val) - 1.0)
        try:
            inv_T = (1.0 / self.params["T0"]) + (1.0 / self.params["Beta"]) * math.log(
                r_th / self.params["R0"]
            )
            return (1.0 / inv_T) - 273.15
        except Exception:
            return 25.0

    def get_state_space(self):
        return None

    def get_disturbance_tf(self, *args):
        return None
