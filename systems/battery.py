import numpy as np
import serial
from core.state_space import StateSpace
from helpers.config import BATTERY_PARAMS


class Battery:
    def __init__(self, **kwargs):
        self.params = BATTERY_PARAMS.copy()
        if kwargs:
            self.params.update(kwargs)
        self.ser = None
        self.current_voltage = 0.0

    def connect(self):
        try:
            self.ser = serial.Serial(
                self.params["port"], self.params["baud"], timeout=0.1
            )
            print(f"\n[HARDWARE] Connected successfully to {self.params['port']}")
        except Exception as e:
            print(
                f"\n[WARNING] Could not connect to hardware on {self.params['port']}."
            )
            print(f"          Reason: {e}")
            print("          -> Running in DUMMY MODE (Reads will be 0.0V)")
            self.ser = None

    def write_pwm(self, val):
        if self.ser:
            pwm = int(np.clip(val, 0, 255))
            self.ser.write(f"Q:{int(pwm)}\n".encode())

    def read_voltage(self):
        if not self.ser:
            return self.current_voltage

        while self.ser.in_waiting:
            try:
                line = self.ser.readline().decode().strip()
                if line.startswith("A:"):
                    adc = int(line[2:])
                    if adc >= 1023:
                        return 5.0
                    elif adc <= 0:
                        return 0.0
                    self.current_voltage = (adc / 1023) * 5
            except Exception:
                print("SERIAL ERROR")

        return self.current_voltage

    def get_state_space(self):
        # First-order model: tau*dV/dt + V = G*u
        # A = [-1/tau], B = [G/tau], C = [1], D = [0]
        tau = 0.5  # Estimated time constant of the averaging
        G = -5.0 / 255  # The MOSFET is an inverter (Higher PWM = Lower Drain Voltage)

        A = np.array([[-1.0 / tau]])
        B = np.array([[G / tau]])
        C = np.array([[1.0]])
        D = np.array([[0.0]])
        return StateSpace(A, B, C, D)

    def close(self):
        if self.ser:
            self.ser.close()
