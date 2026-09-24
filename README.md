# Banana Freshness Detector

Gas-sensor project for predicting banana freshness, with `NO_FRUIT` as a separate state.

## Analysis

[freshness_detection_eda_model.ipynb](freshness_detection_eda_model.ipynb) evaluates the integer decision tree. [snn_freshness_eda.ipynb](snn_freshness_eda.ipynb) trains and evaluates the selected quantized spiking model. Both use the same nine engineered features. Open banana fresh/rotten data is sorted by `Ticks`, divided into recording bursts at gaps over 60 seconds, and reduced to one median reading per observed second within each burst. Empty seconds and multi-hour gaps are never filled. The first 20% of both custom fresh and custom spoiled recordings fits a per-sensor affine custom-to-open mapping; the remaining 80% of both recordings is held out. `NO_FRUIT` uses a raw MQ3 gate before mapping, and `starts_to_spoil` remains diagnostic. Shared loading, calibration, and features are in [freshness_tree_model.py](freshness_tree_model.py); the SNN implementation is in [freshness_snn_model.py](freshness_snn_model.py).

Run the notebook with the repository environment:

```text
.venv/bin/jupyter nbconvert --to notebook --execute freshness_detection_eda_model.ipynb --output /tmp/freshness_detection.executed.ipynb
```

## Arduino Uno

The tree exporter fits a depth-seven entropy decision tree on the one-second open readings, quantizes its feature thresholds, and regenerates its flash-resident header:

```text
.venv/bin/python export_arduino_model.py
```

The SNN exporter fits one fixed rate-coded LIF neuron per engineered feature and a quantized linear readout. Its training-only threshold reserves margin for sensor drift between recording sessions:

```text
.venv/bin/python export_arduino_snn.py
```

Both sketches apply the affine calibration, compute the nine features, and report `FRESH`, `SPOILED`, or `NO_FRUIT` at 115200 baud while updating an e-paper display. They use the confirmed collection wiring: MQ3=`A3`, MQ5=`A5`, MQ135=`A0`. Local AVR builds with the installed GxEPD2 and Adafruit GFX libraries used **17,432 bytes of flash / 665 bytes of static RAM** for the [tree sketch](arduino_sketches/freshness_tinyml_uno/) and **17,660 bytes of flash / 665 bytes of static RAM** for the selected [SNN sketch](arduino_sketches/freshness_snn_uno/), each within the Uno's 32,256-byte program and 2,048-byte RAM limits.

The selected SNN reached **0.9525 mean blocked balanced accuracy** with **0.8790 on the worst blocked split**. Pooled leave-one-complete-session-out predictions reached **0.8123 balanced accuracy**. Training on the earlier sessions and validating on each class's complete final session gave **1.0000 balanced accuracy**. On the held-out custom recordings, it also classified **48,197/48,197 fresh**, **4,630/4,630 spoiled**, and **2,485/2,485 no-fruit** rows correctly. The tree reached **0.9251 balanced accuracy** on held-out custom fresh/spoiled data. The SNN model occupies **94 bytes** and all nine readout weights are nonzero. The open source still contains only two fresh and three spoiled sessions, so more independently collected bananas are needed to measure deployment accuracy. The legacy open-data MQ5/MQ135 column swap and physical deployment still need verification.
