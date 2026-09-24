# Banana Freshness TinyML for Arduino Uno

This sketch predicts banana freshness as `FRESH`, `SPOILED`, or `NO_FRUIT`.

Sensor inputs are MQ3 on `A3`, MQ5 on `A5`, and MQ135 on `A0`, matching the confirmed collection wiring. The sketch averages 20 ADC readings and checks the no-fruit gate in the raw MQ3 domain. Banana readings use the per-sensor affine mapping `open = custom × gain + offset` stored in `freshness_tinyml_model.h`, then enter a depth-seven entropy decision tree trained on open banana fresh/rotten data sorted by `Ticks` and summarized once per observed second within each recording burst. The first 20% of custom fresh and custom spoiled recordings supplies the two calibration anchors. The legacy MQ5/MQ135 column swap used in training should be verified against the connected hardware.

The sketch computes all nine engineered features: MQ3/MQ135, MQ3/MQ5, MQ3−MQ5, MQ3−MQ135, MQ5/MQ135, sample sensor standard deviation, MQ5−MQ135, MQ5+MQ135, and MQ3. The fitted tree receives all nine inputs and selects a subset for its actual splits. It stores 9 packed five-byte nodes (45 bytes) in `PROGMEM`, with int16 thresholds quantized at per-feature precision and byte-sized indexes and labels. Inference uses integer comparisons and no neural-network activation buffers.

The 1.54-inch monochrome GxEPD2 display uses `CS=D10`, `DC=D9`, `RST=D8`, and `BUSY=D7`. It changes only when the predicted state changes. The serial monitor must use **115200 baud**. Example:

```text
ADC[MQ3=92, MQ5=855, MQ135=70] V[MQ3=0.450, MQ5=4.179, MQ135=0.342] RESULT[FRESH]
```

Regenerate the header after changing data or training code with `.venv/bin/python export_arduino_model.py`. The exporter updates only `freshness_tinyml_model.h`; it does not overwrite the display sketch.

The exporter reports blocked-split balanced accuracy, macro F1, spoiled recall, float/integer agreement, and custom-recording predictions. With one-second open data, the tree reached **0.9889 mean blocked balanced accuracy** and **0.9778 spoiled recall**. Holding out the entire last burst of each open class gave only **0.5084 balanced accuracy**. On held-out custom data it reached **0.8502 fresh accuracy**, **1.0000 spoiled accuracy**, and **0.9251 balanced accuracy**. A local AVR build used **17,432 bytes of program memory** and **665 bytes of static RAM** on the Uno target. `starts_to_spoil` has no corresponding open training label and is diagnostic only.
