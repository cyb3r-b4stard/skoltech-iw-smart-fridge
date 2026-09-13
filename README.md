# Fruit Freshness Detector

Gas-sensor machine-learning project for classifying fruit freshness and fruit type.

## Notebook

The main analysis is in [freshness_detection_eda_model.ipynb](freshness_detection_eda_model.ipynb). It covers EDA, CatBoost baselines, three-sensor feature engineering, TinyML comparisons, sensor ablation, feature selection, inference timing, and quantization-size estimates.

Run the notebook with the repository environment:

```text
.venv/bin/jupyter nbconvert --to notebook --execute freshness_detection_eda_model.ipynb --output /tmp/freshness_detection.executed.ipynb
```

## Arduino Uno export

The final embedded model uses `MQ3`, `MQ5`, and `MQ135` with nine engineered features. The exporter trains on the sampled dataset, validates the quantized model, and regenerates the flash-resident header:

```text
.venv/bin/python export_arduino_model.py
```

The display-integrated Uno sketch is in [arduino_sketches/freshness_tinyml_uno](arduino_sketches/freshness_tinyml_uno/). It reports sensor values at **115200 baud** and shows `FRESH` or `SPOILED` on the e-paper display.

The exporter does not overwrite the hand-integrated Arduino sketch. Hardware compilation and upload still require Arduino IDE or Arduino CLI with the GxEPD2 and Adafruit GFX libraries installed.
