from typing import Any, Dict, Final, Optional

import numpy as np
import serial
from numpy.typing import NDArray

from core.state_space import StateSpace
from helpers.config import BATTERY_PARAMS


class Battery:
    def __init__(self, **kwargs: Any) -> None:
        self.params: Dict[str, Any] = BATTERY_PARAMS.copy()
        if kwargs:
            self.params.update(kwargs)
        self.ser: Optional[serial.Serial] = None
        self.current_voltage: float = 0.0

    def connect(self) -> None:
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

    def write_pwm(self, val: float) -> None:
        if self.ser:
            pwm: int = int(np.clip(val, 0, 255))
            self.ser.write(f"Q:{int(pwm)}\n".encode())

    def read_voltage(self) -> float:
        if not self.ser:
            return self.current_voltage

        while self.ser.in_waiting:
            try:
                line: str = self.ser.readline().decode().strip()
                if line.startswith("A:"):
                    adc: int = int(line[2:])
                    if adc >= 1023:
                        return 5.0
                    elif adc <= 0:
                        return 0.0
                    self.current_voltage = (adc / 1023) * 5
            except Exception:
                print("SERIAL ERROR")

        return self.current_voltage

    def get_state_space(self) -> StateSpace:
        tau: Final[float] = 0.5
        G: Final[float] = -5.0 / 255

        A: NDArray[np.float64] = np.array([[-1.0 / tau]])
        B: NDArray[np.float64] = np.array([[G / tau]])
        C: NDArray[np.float64] = np.array([[1.0]])
        D: NDArray[np.float64] = np.array([[0.0]])
        return StateSpace(A, B, C, D)

    def close(self) -> None:
        if self.ser:
            self.ser.close()
