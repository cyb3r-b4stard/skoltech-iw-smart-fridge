/*
  =============================================================================
  Gas Sensors CSV Dataset Collector for TinyML / Machine Learning
  =============================================================================
  Sensors: MQ-3 (A5), MQ-5 (A3), MQ-135 (A0)
  Averaging: 20 consecutive samples per output record
  Output: CSV format via Serial (115200 baud)
  
  Instructions:
  1. Upload code to Arduino.
  2. Open Serial Monitor or Serial Plotter.
  3. Copy the output or redirect Serial stream to a .csv file using 
     Python (pyserial), TeraTerm, or PuTTY.
  =============================================================================
*/

#include <Arduino.h>

// Sensor Pin Definitions
const uint8_t MQ3_PIN   = A5;
const uint8_t MQ5_PIN   = A3;
const uint8_t MQ135_PIN = A0;

// ADC & System Parameters
const float ADC_REFERENCE_VOLTAGE = 5.0f;
const float ADC_MAX_VALUE         = 1023.0f;
const uint8_t NUM_SAMPLES         = 20;    // Number of samples to average
const unsigned long SAMPLE_DELAY  = 50;    // Delay between individual readings (ms) -> 20 * 50ms = 1s per record

// Metadata Config (Change these values before recording each session!)
const char FRUIT_TYPE[] = "banana";  // Options e.g.: "apple", "banana", "tomato", "no_fruit"
const char STATE_LABEL[] = "starts_to_spoil";  // Options: "fresh", "starts_to_spoil", "spoiled", "no_fruit"

// Tracking variables
unsigned long sampleCounter = 0;

// Convert raw ADC value (0-1023) to Voltage (0.0 - 5.0V)
float adcToVoltage(float adcValue) {
  return adcValue * ADC_REFERENCE_VOLTAGE / ADC_MAX_VALUE;
}

void printCsvHeader() {
  Serial.println(F("timestamp_ms,sample_id,mq3_adc,mq5_adc,mq135_adc,mq3_volts,mq5_volts,mq135_volts,fruit,label"));
}

void setup() {
  Serial.begin(115200);
  while (!Serial) { ; } // Wait for serial port to connect

  // Allow sensors a brief moment to settle
  delay(1000);

  // Output CSV Header
  printCsvHeader();
}

void loop() {
  float sumMQ3   = 0.0f;
  float sumMQ5   = 0.0f;
  float sumMQ135 = 0.0f;

  // 1. Collect 20 samples and calculate accumulated sum
  for (uint8_t i = 0; i < NUM_SAMPLES; ++i) {
    sumMQ3   += analogRead(MQ3_PIN);
    sumMQ5   += analogRead(MQ5_PIN);
    sumMQ135 += analogRead(MQ135_PIN);
    delay(SAMPLE_DELAY);
  }

  // 2. Compute averages
  float avgMQ3   = sumMQ3 / (float)NUM_SAMPLES;
  float avgMQ5   = sumMQ5 / (float)NUM_SAMPLES;
  float avgMQ135 = sumMQ135 / (float)NUM_SAMPLES;

  // 3. Compute voltages from averaged ADC
  float voltMQ3   = adcToVoltage(avgMQ3);
  float voltMQ5   = adcToVoltage(avgMQ5);
  float voltMQ135 = adcToVoltage(avgMQ135);

  sampleCounter++;

  // 4. Output dataset row in CSV format
  Serial.print(millis());
  Serial.print(F(","));
  Serial.print(sampleCounter);
  Serial.print(F(","));
  Serial.print(avgMQ3, 2);
  Serial.print(F(","));
  Serial.print(avgMQ5, 2);
  Serial.print(F(","));
  Serial.print(avgMQ135, 2);
  Serial.print(F(","));
  Serial.print(voltMQ3, 3);
  Serial.print(F(","));
  Serial.print(voltMQ5, 3);
  Serial.print(F(","));
  Serial.print(voltMQ135, 3);
  Serial.print(F(","));
  Serial.print(FRUIT_TYPE);
  Serial.print(F(","));
  Serial.println(STATE_LABEL);
}