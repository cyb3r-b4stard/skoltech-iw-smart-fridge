# Freshness TinyML for Arduino Uno

This sketch contains the final TinyML export trained from the three gas sensors:

- MQ-3: `A5`
- MQ-5: `A4`
- MQ-135: `A3`

It uses the selected nine engineered features, in this exact order:

1. `MQ3 / MQ135`
2. `MQ3 / MQ5`
3. `MQ3 - MQ5`
4. `MQ3 - MQ135`
5. `MQ5 / MQ135`
6. sample standard deviation of `MQ3`, `MQ5`, `MQ135`
7. `MQ5 - MQ135`
8. `MQ5 + MQ135`
9. `MQ3`

The sketch contains two exported MLP heads:

- Freshness: `9 -> 40 -> 28 -> 14 -> 1`
- Fruit type: `9 -> 40 -> 28 -> 14 -> 8`

Weights are int8 and stored in flash with `PROGMEM`; biases are int32 and the model uses small float activation buffers. The slightly smaller generated header plus an 8-row e-paper page buffer and in-place feature normalization reduce SRAM pressure while keeping the model within the Arduino Uno's 32 KB flash and 2 KB SRAM limits.

Open `freshness_tinyml_uno.ino` in the Arduino IDE, select Arduino Uno, connect the sensors, and upload. Predictions are printed to the serial monitor at 115200 baud.

The sketch also drives a 1.54-inch monochrome GxEPD2 display using `CS=D10`, `DC=D9`, `RST=D8`, and `BUSY=D7`. A fresh prediction shows only `FRESH` and a clean apple illustration; a non-fresh prediction shows only `SPOILED` and a spotted, cracked apple illustration. The e-paper refreshes only when the predicted freshness state changes.

The serial monitor runs at 115200 baud and prints labeled raw ADC values, sensor voltages in volts, freshness, and fruit prediction. Example:

```text
ADC[MQ3=512, MQ5=430, MQ135=601] V[MQ3=2.502, MQ5=2.102, MQ135=2.942] RESULT[FRESH, fruit=apple]
```

Voltages are calculated from the Uno's 5 V ADC reference and 10-bit ADC range. Set the serial monitor to **115200 baud**; other baud rates will display unreadable characters.

To regenerate the export after retraining:

```text
.venv/bin/python export_arduino_model.py
```

The exporter trains on the full sampled dataset. The notebook remains the source for validation metrics; the generated C model is intended for embedded inference.

The exporter updates only `freshness_tinyml_model.h`; it intentionally does not overwrite this display-integrated sketch.

## Quantization validation

The exporter emulates the generated int8-weight/int32-bias inference path before writing the final model. On the held-out temporal-segment split, the quantized models achieved:

- Slightly smaller model: freshness **91.14%**, fruit type **90.76%**, joint **83.77%**.

For the final smaller models trained on all sampled rows, quantized predictions agreed with the original float models on **99.60%** of freshness predictions and **98.28%** of fruit predictions. This validates the quantization numerically in Python; an Arduino IDE or Arduino CLI build/upload is still required for hardware-level verification.
