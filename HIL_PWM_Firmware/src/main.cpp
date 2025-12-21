#include <Arduino.h>

// Arduino Firmware for Python-Based HIL Battery Control
// Baud Rate: 115200

const int PWM_PIN = 3;     // PWM Output to MOSFET Gate
const int SENSOR_PIN = A0; // Analog Input from MOSFET Drain
unsigned long lastTelemetry = 0;

void setup()
{
  Serial.begin(115200);
  pinMode(PWM_PIN, OUTPUT);
  analogWrite(PWM_PIN, 0); // Start at 0V
}

void loop()
{
  // 1. Process Control Command from Python (Format: "Q:VALUE\n")
  if (Serial.available() > 0)
  {
    String cmd = Serial.readStringUntil('\n');
    cmd.trim();
    if (cmd.startsWith("Q:"))
    {
      int pwmVal = cmd.substring(2).toInt();
      analogWrite(PWM_PIN, constrain(pwmVal, 0, 255));
    }
  }

  // 2. Transmit Periodic Telemetry (10Hz)
  if (millis() - lastTelemetry >= 100)
  {
    lastTelemetry = millis();
    // High-speed averaging to create a "software" thermal/electrical mass
    long sum = 0;
    for (int i = 0; i < 50; i++)
    {
      sum += analogRead(SENSOR_PIN);
      delayMicroseconds(50);
    }
    int averagedADC = sum / 50;

    Serial.print("A:");
    Serial.println(averagedADC);
  }
}