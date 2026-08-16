#include <Arduino.h>

// HIL_Thermistor_Plant.ino
// Arduino Firmware for Python-Based HIL Temp Control
// Baud Rate: 115200

const int HEATER_PIN = 3;  // PWM Output to MOSFET Gate
const int SENSOR_PIN = A0; // Analog Input from Thermistor Divider
unsigned long lastTelemetry = 0;
int currentPower = 0;

void setup()
{
  Serial.begin(115200);
  pinMode(HEATER_PIN, OUTPUT);
  analogWrite(HEATER_PIN, 0); // Safety: Start with heater OFF
}

void loop()
{
  // 1. Process Incoming Control Commands (Format: "Q:PWM_VALUE\n")
  if (Serial.available() > 0)
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("Q:"))
    {
      currentPower = cmd.substring(2).toInt();
      currentPower = constrain(currentPower, 0, 255);
      analogWrite(HEATER_PIN, currentPower);
    }
  }

  // 2. Transmit Periodic Telemetry (Format: "A:RAW_ADC\n")
  // Frequency: 10Hz (Every 100ms)
  if (millis() - lastTelemetry >= 100)
  {
    lastTelemetry = millis();
    // Simulate thermal mass by averaging the PWM signal
    // We read many times quickly and average to get a stable "temp"
    long sum = 0;
    for (int i = 0; i < 50; i++)
    {
      sum += analogRead(SENSOR_PIN);
      delayMicroseconds(100);
    }
    int averagedADC = sum / 50;

    Serial.print("A:");
    Serial.println(averagedADC);
  }
}