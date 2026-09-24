#include "freshness_snn_model.h"
#include <GxEPD2_BW.h>
#include <Adafruit_GFX.h>
#include <avr/pgmspace.h>
#include <math.h>

const uint8_t MQ3_PIN = A3;
const uint8_t MQ5_PIN = A5;
const uint8_t MQ135_PIN = A0;
const uint8_t NUM_SAMPLES = 20;
const float ADC_REFERENCE_VOLTAGE = 5.0f;

GxEPD2_BW<GxEPD2_154_D67, 8> display(
  GxEPD2_154_D67(/*CS=*/ 10, /*DC=*/ 9, /*RST=*/ 8, /*BUSY=*/ 7));

static int8_t lastState = -1;

int16_t quantizeSigned(float scaled, int16_t limit) {
  if (scaled > limit) return limit;
  if (scaled < -limit) return -limit;
  return static_cast<int16_t>(scaled + (scaled >= 0.0f ? 0.5f : -0.5f));
}

void buildFeatures(float mq3, float mq5, float mq135, int16_t output[SNN_INPUTS]) {
  float mean = (mq3 + mq5 + mq135) / 3.0f;
  float d3 = mq3 - mean;
  float d5 = mq5 - mean;
  float d135 = mq135 - mean;
  float values[SNN_INPUTS] = {
    mq3 / (mq135 + 0.000001f),
    mq3 / (mq5 + 0.000001f),
    mq3 - mq5,
    mq3 - mq135,
    mq5 / (mq135 + 0.000001f),
    sqrtf((d3 * d3 + d5 * d5 + d135 * d135) / 2.0f),
    mq5 - mq135,
    mq5 + mq135,
    mq3,
  };
  for (uint8_t i = 0; i < SNN_INPUTS; ++i) {
    float center = pgm_read_float(&kSnnCenter[i]);
    float invScale = pgm_read_float(&kSnnInvScale[i]);
    output[i] = quantizeSigned((values[i] - center) * invScale * 256.0f, 1024);
  }
}

uint8_t predictFreshness(const int16_t features[SNN_INPUTS]) {
  int32_t score = kSnnOutputBiasQ;
  for (uint8_t neuron = 0; neuron < SNN_NEURONS; ++neuron) {
    // z in [-4, 4] maps to a current in [0, half threshold].
    int32_t drive = (static_cast<int32_t>(features[neuron]) + 1024) / 4;
    int32_t membrane = 0;
    uint8_t spikes = 0;
    for (uint8_t step = 0; step < SNN_STEPS; ++step) {
      membrane = (membrane * 192) / 256 + drive;
      if (membrane >= 1024) {
        membrane -= 1024;
        ++spikes;
      }
    }
    score += static_cast<int32_t>(spikes) *
        static_cast<int16_t>(pgm_read_word(&kSnnReadout[neuron]));
  }
  return score >= 0 ? 1 : 0;
}

void drawState(uint8_t state) {
  display.setFullWindow();
  display.firstPage();
  do {
    display.fillScreen(GxEPD_WHITE);
    display.setTextColor(GxEPD_BLACK);
    display.setTextSize(state == 1 ? 3 : 2);
    display.setCursor(state == 1 ? 55 : 45, 95);
    if (state == 2) display.print(F("NO FRUIT"));
    else if (state == 1) display.print(F("FRESH"));
    else display.print(F("SPOILED"));
  } while (display.nextPage());
}

void setup() {
  Serial.begin(115200);
  display.init(115200);
  display.setRotation(1);
  Serial.println(F("Banana freshness SNN ready"));
}

void loop() {
  float sums[3] = {0.0f, 0.0f, 0.0f};
  for (uint8_t i = 0; i < NUM_SAMPLES; ++i) {
    sums[0] += analogRead(MQ3_PIN);
    sums[1] += analogRead(MQ5_PIN);
    sums[2] += analogRead(MQ135_PIN);
    delay(50);
  }

  float raw[3];
  float mapped[3];
  for (uint8_t i = 0; i < 3; ++i) {
    raw[i] = sums[i] / NUM_SAMPLES;
    float gain = pgm_read_float(&kCustomToOpenGain[i]);
    float offset = pgm_read_float(&kCustomToOpenOffset[i]);
    mapped[i] = raw[i] * gain + offset;
  }
  uint8_t state = 2;
  if (quantizeSigned(raw[0] * 8.0f, 32767) >= kNoFruitRawMq3ThresholdQ8) {
    int16_t features[SNN_INPUTS];
    buildFeatures(mapped[0], mapped[1], mapped[2], features);
    state = predictFreshness(features);
  }

  Serial.print(F("ADC[MQ3="));
  Serial.print(raw[0], 2);
  Serial.print(F(", MQ5="));
  Serial.print(raw[1], 2);
  Serial.print(F(", MQ135="));
  Serial.print(raw[2], 2);
  Serial.print(F("] V[MQ3="));
  Serial.print(raw[0] * ADC_REFERENCE_VOLTAGE / 1023.0f, 3);
  Serial.print(F(", MQ5="));
  Serial.print(raw[1] * ADC_REFERENCE_VOLTAGE / 1023.0f, 3);
  Serial.print(F(", MQ135="));
  Serial.print(raw[2] * ADC_REFERENCE_VOLTAGE / 1023.0f, 3);
  Serial.print(F("] RESULT["));
  if (state == 2) Serial.print(F("NO_FRUIT"));
  else if (state == 1) Serial.print(F("FRESH"));
  else Serial.print(F("SPOILED"));
  Serial.println(F("]"));

  if (lastState != static_cast<int8_t>(state)) {
    drawState(state);
    lastState = static_cast<int8_t>(state);
  }
}
