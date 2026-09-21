#include "freshness_tinyml_model.h"
#include <GxEPD2_BW.h>
#include <Adafruit_GFX.h>
#include <math.h>
#include <string.h>

const uint8_t MQ3_PIN = A5;
const uint8_t MQ5_PIN = A4;
const uint8_t MQ135_PIN = A3;
const float ADC_REFERENCE_VOLTAGE = 5.0f;
const float ADC_MAX_VALUE = 1023.0f;
const uint8_t NUM_SAMPLES = 20; // Number of consecutive samples to average

GxEPD2_BW<GxEPD2_154_D67, 8> display(
  GxEPD2_154_D67(/*CS=*/ 10, /*DC=*/ 9, /*RST=*/ 8, /*BUSY=*/ 7));

static float feature_vector[TINYML_INPUTS];
static float hidden0[40];
static float hidden1[28];
static float hidden2[14];
static float output[TINYML_FRUIT_OUTPUTS];
static int8_t lastFreshness = -1;

// Forward declarations
void drawFreshApple();
void drawRottenApple();

float adcToVoltage(float adcValue) {
  return adcValue * ADC_REFERENCE_VOLTAGE / ADC_MAX_VALUE;
}

void buildFeatures(float mq3, float mq5, float mq135) {
  feature_vector[0] = mq3 / (mq135 + TINYML_EPSILON);
  feature_vector[1] = mq3 / (mq5 + TINYML_EPSILON);
  feature_vector[2] = mq3 - mq5;
  feature_vector[3] = mq3 - mq135;
  feature_vector[4] = mq5 / (mq135 + TINYML_EPSILON);
  float mean = (mq3 + mq5 + mq135) / 3.0f;
  float d0 = mq3 - mean;
  float d1 = mq5 - mean;
  float d2 = mq135 - mean;
  feature_vector[5] = sqrt((d0 * d0 + d1 * d1 + d2 * d2) / 2.0f);
  feature_vector[6] = mq5 - mq135;
  feature_vector[7] = mq5 + mq135;
  feature_vector[8] = mq3;
}

void normalizeFeatures() {
  for (uint8_t i = 0; i < TINYML_INPUTS; ++i) {
    float mean = pgm_read_float(&kFeatureMean[i]);
    float scale = pgm_read_float(&kFeatureScale[i]);
    feature_vector[i] = (feature_vector[i] - mean) / scale;
  }
}

void runModel(const TinyMlModel& model, float* result) {
  for (uint8_t neuron = 0; neuron < model.hidden0; ++neuron) {
    float accumulator = (int32_t)pgm_read_dword(&model.biases0[neuron]);
    for (uint8_t inputIndex = 0; inputIndex < TINYML_INPUTS; ++inputIndex) {
      uint16_t offset = inputIndex * model.hidden0 + neuron;
      accumulator += feature_vector[inputIndex] * (int8_t)pgm_read_byte(&model.weights0[offset]);
    }
    hidden0[neuron] = max(0.0f, accumulator * model.scale0);
  }
  for (uint8_t neuron = 0; neuron < model.hidden1; ++neuron) {
    float accumulator = (int32_t)pgm_read_dword(&model.biases1[neuron]);
    for (uint8_t inputIndex = 0; inputIndex < model.hidden0; ++inputIndex) {
      uint16_t offset = inputIndex * model.hidden1 + neuron;
      accumulator += hidden0[inputIndex] * (int8_t)pgm_read_byte(&model.weights1[offset]);
    }
    hidden1[neuron] = max(0.0f, accumulator * model.scale1);
  }
  for (uint8_t neuron = 0; neuron < model.hidden2; ++neuron) {
    float accumulator = (int32_t)pgm_read_dword(&model.biases2[neuron]);
    for (uint8_t inputIndex = 0; inputIndex < model.hidden1; ++inputIndex) {
      uint16_t offset = inputIndex * model.hidden2 + neuron;
      accumulator += hidden1[inputIndex] * (int8_t)pgm_read_byte(&model.weights2[offset]);
    }
    hidden2[neuron] = max(0.0f, accumulator * model.scale2);
  }
  for (uint8_t neuron = 0; neuron < model.outputs; ++neuron) {
    float accumulator = (int32_t)pgm_read_dword(&model.biases3[neuron]);
    for (uint8_t inputIndex = 0; inputIndex < model.hidden2; ++inputIndex) {
      uint16_t offset = inputIndex * model.outputs + neuron;
      accumulator += hidden2[inputIndex] * (int8_t)pgm_read_byte(&model.weights3[offset]);
    }
    result[neuron] = accumulator * model.scale3;
  }
}

void setup() {
  Serial.begin(115200);
  display.init(115200);
  display.setRotation(1);
  Serial.println(F("Freshness TinyML ready - Serial 115200 baud"));
}

void loop() {
  float sum_mq3 = 0.0f;
  float sum_mq5 = 0.0f;
  float sum_mq135 = 0.0f;

  // Collect NUM_SAMPLES consecutive measurements
  for (uint8_t i = 0; i < NUM_SAMPLES; ++i) {
    sum_mq3 += analogRead(MQ3_PIN);
    sum_mq5 += analogRead(MQ5_PIN);
    sum_mq135 += analogRead(MQ135_PIN);
    delay(50); // Short delay between individual sample readings
  }

  // Calculate averages
  float avg_mq3 = sum_mq3 / NUM_SAMPLES;
  float avg_mq5 = sum_mq5 / NUM_SAMPLES;
  float avg_mq135 = sum_mq135 / NUM_SAMPLES;

  // Run feature extraction and inference on averaged values
  buildFeatures(avg_mq3, avg_mq5, avg_mq135);
  normalizeFeatures();
  
  runModel(kFreshnessModel, output);
  bool fresh = output[0] >= 0.0f;
  
  runModel(kFruitModel, output);
  uint8_t fruitIndex = 0;
  for (uint8_t i = 1; i < TINYML_FRUIT_OUTPUTS; ++i) {
    if (output[i] > output[fruitIndex]) fruitIndex = i;
  }

  char fruit[12];
  strcpy_P(fruit, kFruitLabels[fruitIndex]);

  // Serial logging using averaged values
  Serial.print(F("ADC[MQ3="));
  Serial.print(static_cast<int>(avg_mq3));
  Serial.print(F(", MQ5="));
  Serial.print(static_cast<int>(avg_mq5));
  Serial.print(F(", MQ135="));
  Serial.print(static_cast<int>(avg_mq135));
  Serial.print(F("] V[MQ3="));
  Serial.print(adcToVoltage(avg_mq3), 3);
  Serial.print(F(", MQ5="));
  Serial.print(adcToVoltage(avg_mq5), 3);
  Serial.print(F(", MQ135="));
  Serial.print(adcToVoltage(avg_mq135), 3);
  Serial.print(F("] RESULT["));
  Serial.print(fresh ? F("FRESH") : F("SPOILED"));
  Serial.print(F(", fruit="));
  Serial.print(fruit);
  Serial.println(F("]"));

  if (lastFreshness != static_cast<int8_t>(fresh)) {
    if (fresh) {
      drawFreshApple();
    } else {
      drawRottenApple();
    }
    lastFreshness = static_cast<int8_t>(fresh);
  }
}

void drawFreshApple() {
  display.setFullWindow();
  display.firstPage();
  do {
    display.fillScreen(GxEPD_WHITE);
    display.setTextColor(GxEPD_BLACK);
    display.setTextSize(3);
    display.setCursor(55, 18);
    display.print(F("FRESH"));

    display.drawCircle(82, 112, 43, GxEPD_BLACK);
    display.drawCircle(118, 112, 43, GxEPD_BLACK);
    display.fillCircle(100, 125, 35, GxEPD_WHITE);
    display.drawLine(100, 69, 100, 48, GxEPD_BLACK);
    display.drawLine(100, 51, 116, 42, GxEPD_BLACK);
    display.drawLine(116, 42, 132, 49, GxEPD_BLACK);
    display.drawLine(132, 49, 116, 58, GxEPD_BLACK);
    display.drawLine(116, 58, 100, 51, GxEPD_BLACK);
    display.drawCircle(78, 101, 7, GxEPD_BLACK);
    display.drawCircle(78, 101, 4, GxEPD_WHITE);

  } while (display.nextPage());
}

void drawRottenApple() {
  display.setFullWindow();
  display.firstPage();
  do {
    display.fillScreen(GxEPD_WHITE);
    display.setTextColor(GxEPD_BLACK);
    display.setTextSize(2);
    display.setCursor(58, 18);
    display.print(F("SPOILED"));

    display.drawCircle(82, 112, 43, GxEPD_BLACK);
    display.drawCircle(118, 112, 43, GxEPD_BLACK);
    display.fillCircle(100, 125, 35, GxEPD_WHITE);
    display.drawLine(100, 69, 100, 48, GxEPD_BLACK);
    display.drawLine(100, 51, 116, 42, GxEPD_BLACK);
    display.drawLine(116, 42, 132, 49, GxEPD_BLACK);
    display.drawLine(132, 49, 116, 58, GxEPD_BLACK);
    display.drawLine(116, 58, 100, 51, GxEPD_BLACK);

    display.fillCircle(76, 101, 8, GxEPD_BLACK);
    display.fillCircle(126, 126, 10, GxEPD_BLACK);
    display.fillCircle(94, 143, 6, GxEPD_BLACK);
    display.drawLine(105, 86, 96, 106, GxEPD_BLACK);
    display.drawLine(96, 106, 107, 119, GxEPD_BLACK);
    display.drawLine(107, 119, 98, 136, GxEPD_BLACK);
    display.drawLine(59, 151, 74, 158, GxEPD_BLACK);
    display.drawLine(74, 158, 87, 151, GxEPD_BLACK);

  } while (display.nextPage());
}