from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_DIR = Path('datasets')
OUTPUT_DIR = Path('arduino_sketches/freshness_tinyml_uno')
BASE_SENSORS = ['MQ3', 'MQ5', 'MQ135']
SELECTED_FEATURES = [
    'MQ3_ratio_MQ135', 'MQ3_ratio_MQ5', 'MQ3_minus_MQ5',
    'MQ3_minus_MQ135', 'MQ5_ratio_MQ135', 'sensor_std',
    'MQ5_minus_MQ135', 'MQ5_plus_MQ135', 'MQ3'
]
FRUIT_LABELS = ['apple', 'banana', 'blueberry', 'grape', 'kiwi', 'pear', 'strawberry', 'tomato']


def make_features(frame):
    raw = frame[BASE_SENSORS].astype(float)
    features = pd.DataFrame(index=frame.index)
    features['MQ3_ratio_MQ135'] = raw['MQ3'] / (raw['MQ135'] + 1e-6)
    features['MQ3_ratio_MQ5'] = raw['MQ3'] / (raw['MQ5'] + 1e-6)
    features['MQ3_minus_MQ5'] = raw['MQ3'] - raw['MQ5']
    features['MQ3_minus_MQ135'] = raw['MQ3'] - raw['MQ135']
    features['MQ5_ratio_MQ135'] = raw['MQ5'] / (raw['MQ135'] + 1e-6)
    features['sensor_std'] = raw.std(axis=1)
    features['MQ5_minus_MQ135'] = raw['MQ5'] - raw['MQ135']
    features['MQ5_plus_MQ135'] = raw['MQ5'] + raw['MQ135']
    features['MQ3'] = raw['MQ3']
    return features.replace([np.inf, -np.inf], np.nan).fillna(0.0)[SELECTED_FEATURES]


def load_data():
    frames = []
    for path in sorted(DATA_DIR.glob('*.csv')):
        frame = pd.read_csv(path)
        frame['fruit'] = path.stem.rsplit('_', 1)[0]
        frame['freshness'] = path.stem.rsplit('_', 1)[1]
        frame['is_fresh'] = (frame['freshness'] == 'fresh').astype(int)
        step = max(1, len(frame) // 3000)
        sampled = frame.iloc[::step].head(3000).copy()
        frames.append(sampled)
    return pd.concat(frames, ignore_index=True)


def make_model(hidden_layers):
    return Pipeline([
        ('scale', StandardScaler()),
        ('model', MLPClassifier(
            hidden_layer_sizes=hidden_layers,
            activation='relu', alpha=0.001,
            batch_size=256, max_iter=300,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15, random_state=42
        ))
    ])


def c_float(value):
    return f'{float(value):.9g}f'


def c_array(values, per_line=12):
    flat = np.asarray(values).reshape(-1)
    lines = []
    for start in range(0, len(flat), per_line):
        lines.append('  ' + ', '.join(str(int(value)) for value in flat[start:start + per_line]))
    return ',\n'.join(lines)


def quantize_model(model):
    scaler = model.named_steps['scale']
    network = model.named_steps['model']
    weights = []
    weight_scales = []
    biases = []
    for matrix, bias in zip(network.coefs_, network.intercepts_):
        scale = max(float(np.max(np.abs(matrix))) / 127.0, 1e-8)
        weights.append(np.clip(np.rint(matrix / scale), -127, 127).astype(np.int8))
        weight_scales.append(scale)
        biases.append(np.rint(bias / scale).astype(np.int32))
    return {
        'mean': scaler.mean_,
        'scale': scaler.scale_,
        'weights': weights,
        'weight_scales': weight_scales,
        'biases': biases,
        'layers': [weights[0].shape[0]] + [matrix.shape[1] for matrix in weights]
    }


def quantized_predict(model_data, features):
    values = (features.to_numpy(dtype=np.float32) - model_data['mean']) / model_data['scale']
    for layer_index, (weights, biases, weight_scale) in enumerate(zip(
            model_data['weights'], model_data['biases'], model_data['weight_scales'])):
        values = values @ weights.astype(np.float32) + biases.astype(np.float32)
        values *= weight_scale
        if layer_index < len(model_data['weights']) - 1:
            values = np.maximum(values, 0.0)
    return values


def quantized_predictions(model_data, features, target_type):
    outputs = quantized_predict(model_data, features)
    if target_type == 'binary':
        return (outputs[:, 0] >= 0.0).astype(np.int8)
    return np.asarray(FRUIT_LABELS)[np.argmax(outputs, axis=1)]


def validate_quantization(data, features):
    recording = data['fruit'] + '_' + data['freshness']
    segment = data.groupby(recording, sort=False).cumcount() // 300
    groups = recording + '_segment_' + segment.astype(str)
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(splitter.split(features, data['is_fresh'], groups=groups))

    freshness = make_model((40, 28, 14))
    fruit = make_model((40, 28, 14))
    freshness.fit(features.iloc[train_idx], data['is_fresh'].iloc[train_idx])
    fruit.fit(features.iloc[train_idx], data['fruit'].iloc[train_idx])
    quantized_freshness = quantize_model(freshness)
    quantized_fruit = quantize_model(fruit)

    quantized_freshness_predictions = quantized_predictions(
        quantized_freshness, features.iloc[test_idx], 'binary')
    quantized_fruit_predictions = quantized_predictions(
        quantized_fruit, features.iloc[test_idx], 'multiclass')
    print(f'held-out quantized freshness accuracy: '
          f'{accuracy_score(data["is_fresh"].iloc[test_idx], quantized_freshness_predictions):.4f}')
    print(f'held-out quantized fruit accuracy: '
          f'{accuracy_score(data["fruit"].iloc[test_idx], quantized_fruit_predictions):.4f}')
    held_out_joint = (
        (quantized_freshness_predictions == data['is_fresh'].iloc[test_idx].to_numpy())
        & (quantized_fruit_predictions == data['fruit'].iloc[test_idx].to_numpy())
    ).mean()
    print(f'held-out quantized joint accuracy: {held_out_joint:.4f}')

    full_freshness = make_model((40, 28, 14)).fit(features, data['is_fresh'])
    full_fruit = make_model((40, 28, 14)).fit(features, data['fruit'])
    full_quantized_freshness = quantize_model(full_freshness)
    full_quantized_fruit = quantize_model(full_fruit)
    float_freshness = full_freshness.predict(features)
    float_fruit = full_fruit.predict(features)
    quantized_freshness = quantized_predictions(full_quantized_freshness, features, 'binary')
    quantized_fruit = quantized_predictions(full_quantized_fruit, features, 'multiclass')
    print(f'full-data float/quantized freshness agreement: '
          f'{accuracy_score(float_freshness, quantized_freshness):.4f}')
    print(f'full-data float/quantized fruit agreement: '
          f'{accuracy_score(float_fruit, quantized_fruit):.4f}')


def write_header(freshness, fruit):
    lines = [
        '#pragma once',
        '#include <Arduino.h>',
        '#include <avr/pgmspace.h>',
        '',
        '// Generated by export_arduino_model.py. Do not edit by hand.',
        '// Inputs: MQ3, MQ5, MQ135. All model constants live in flash.',
        'constexpr uint8_t TINYML_INPUTS = 9;',
        'constexpr uint8_t TINYML_FRESHNESS_OUTPUTS = 1;',
        'constexpr uint8_t TINYML_FRUIT_OUTPUTS = 8;',
        'constexpr float TINYML_EPSILON = 0.000001f;',
        '',
        'struct TinyMlModel {',
        '  const int8_t* weights0;',
        '  const int32_t* biases0;',
        '  const int8_t* weights1;',
        '  const int32_t* biases1;',
        '  const int8_t* weights2;',
        '  const int32_t* biases2;',
        '  const int8_t* weights3;',
        '  const int32_t* biases3;',
        '  float scale0;',
        '  float scale1;',
        '  float scale2;',
        '  float scale3;',
        '  uint8_t hidden0;',
        '  uint8_t hidden1;',
        '  uint8_t hidden2;',
        '  uint8_t outputs;',
        '};',
        '',
        'const float kFeatureMean[TINYML_INPUTS] PROGMEM = {',
        '  ' + ', '.join(c_float(value) for value in freshness['mean']),
        '};',
        'const float kFeatureScale[TINYML_INPUTS] PROGMEM = {',
        '  ' + ', '.join(c_float(value) for value in freshness['scale']),
        '};',
        '',
    ]
    model_declarations = []
    for prefix, model in [('Freshness', freshness), ('Fruit', fruit)]:
        for index, (weights, biases, scale) in enumerate(zip(model['weights'], model['biases'], model['weight_scales'])):
            lines.extend([
                f'const int8_t k{prefix}Weights{index}[] PROGMEM = {{',
                c_array(weights),
                '};',
                f'const int32_t k{prefix}Biases{index}[] PROGMEM = {{',
                c_array(biases),
                '};',
                f'constexpr float k{prefix}WeightScale{index} = {c_float(scale)};',
                ''
            ])
        model_declarations.append(
               f'const TinyMlModel k{prefix}Model = {{'
            f'k{prefix}Weights0, k{prefix}Biases0, '
            f'k{prefix}Weights1, k{prefix}Biases1, '
            f'k{prefix}Weights2, k{prefix}Biases2, '
            f'k{prefix}Weights3, k{prefix}Biases3, '
            f'k{prefix}WeightScale0, k{prefix}WeightScale1, k{prefix}WeightScale2, k{prefix}WeightScale3, '
            f'{model["layers"][1]}, {model["layers"][2]}, {model["layers"][3]}, {model["layers"][4]}'
            '};'
        )
    lines.extend(model_declarations)
    lines.extend([
        '',
        'const char kFruitLabels[TINYML_FRUIT_OUTPUTS][12] PROGMEM = {',
        '  "apple", "banana", "blueberry", "grape",',
        '  "kiwi", "pear", "strawberry", "tomato"',
        '};',
        '',
    ])
    return '\n'.join(lines)


def write_sketch():
    """Legacy template retained for reference; main() does not use it."""
    return '''#include "freshness_tinyml_model.h"\n#include <math.h>\n#include <string.h>\n\nconst uint8_t MQ3_PIN = A5;\nconst uint8_t MQ5_PIN = A4;\nconst uint8_t MQ135_PIN = A3;\n\nstatic float feature_vector[TINYML_INPUTS];\nstatic float hidden0[24];\nstatic float hidden1[24];\nstatic float output[8];\n\nstatic float readFloat(const float* values, uint8_t index) {\n  return values[index];\n}\n\nvoid buildFeatures(float mq3, float mq5, float mq135) {\n  feature_vector[0] = mq3 / (mq135 + TINYML_EPSILON);\n  feature_vector[1] = mq3 / (mq5 + TINYML_EPSILON);\n  feature_vector[2] = mq3 - mq5;\n  feature_vector[3] = mq3 - mq135;\n  feature_vector[4] = mq5 / (mq135 + TINYML_EPSILON);\n  float mean = (mq3 + mq5 + mq135) / 3.0f;\n  float d0 = mq3 - mean;\n  float d1 = mq5 - mean;\n  float d2 = mq135 - mean;\n  feature_vector[5] = sqrt((d0 * d0 + d1 * d1 + d2 * d2) / 2.0f);\n  feature_vector[6] = mq5 - mq135;\n  feature_vector[7] = mq5 + mq135;\n  feature_vector[8] = mq3;\n}\n\nvoid runModel(const TinyMlModel& model, float* result) {\n  float input[TINYML_INPUTS];\n  for (uint8_t i = 0; i < TINYML_INPUTS; ++i) {\n    float mean = pgm_read_float(&kFeatureMean[i]);\n    float scale = pgm_read_float(&kFeatureScale[i]);\n    input[i] = (feature_vector[i] - mean) / scale;\n  }\n  for (uint8_t neuron = 0; neuron < model.hidden0; ++neuron) {\n    int32_t accumulator = pgm_read_dword(&model.biases0[neuron]);\n    for (uint8_t inputIndex = 0; inputIndex < TINYML_INPUTS; ++inputIndex) {\n      uint16_t offset = inputIndex * model.hidden0 + neuron;\n      accumulator += input[inputIndex] * pgm_read_byte(&model.weights0[offset]);\n    }\n    hidden0[neuron] = max(0.0f, accumulator * model.scale0);\n  }\n  for (uint8_t neuron = 0; neuron < model.hidden1; ++neuron) {\n    int32_t accumulator = pgm_read_dword(&model.biases1[neuron]);\n    for (uint8_t inputIndex = 0; inputIndex < model.hidden0; ++inputIndex) {\n      uint16_t offset = inputIndex * model.hidden1 + neuron;\n      accumulator += hidden0[inputIndex] * pgm_read_byte(&model.weights1[offset]);\n    }\n    hidden1[neuron] = max(0.0f, accumulator * model.scale1);\n  }\n  for (uint8_t neuron = 0; neuron < model.outputs; ++neuron) {\n    int32_t accumulator = pgm_read_dword(&model.biases2[neuron]);\n    for (uint8_t inputIndex = 0; inputIndex < model.hidden1; ++inputIndex) {\n      uint16_t offset = inputIndex * model.outputs + neuron;\n      accumulator += hidden1[inputIndex] * pgm_read_byte(&model.weights2[offset]);\n    }\n    result[neuron] = accumulator * model.scale2;\n  }\n}\n\nvoid setup() {\n  Serial.begin(9600);\n}\n\nvoid loop() {\n  float mq3 = analogRead(MQ3_PIN);\n  float mq5 = analogRead(MQ5_PIN);\n  float mq135 = analogRead(MQ135_PIN);\n  buildFeatures(mq3, mq5, mq135);\n  runModel(kFreshnessModel, output);\n  bool fresh = output[0] >= 0.0f;\n  runModel(kFruitModel, output);\n  uint8_t fruitIndex = 0;\n  for (uint8_t i = 1; i < TINYML_FRUIT_OUTPUTS; ++i) {\n    if (output[i] > output[fruitIndex]) fruitIndex = i;\n  }\n  char fruit[12];\n  strcpy_P(fruit, kFruitLabels[fruitIndex]);\n  Serial.print(F("Fruit: "));\n  Serial.print(fruit);\n  Serial.print(F(" | Status: "));\n  Serial.println(fresh ? F("fresh") : F("rotten"));\n  delay(1000);\n}\n'''


def main():
    data = load_data()
    features = make_features(data)
    validate_quantization(data, features)
    freshness = make_model((40, 28, 14))
    fruit = make_model((40, 28, 14))
    freshness.fit(features, data['is_fresh'])
    fruit.fit(features, data['fruit'])
    quantized_freshness = quantize_model(freshness)
    quantized_fruit = quantize_model(fruit)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / 'freshness_tinyml_model.h').write_text(
        write_header(quantized_freshness, quantized_fruit)
    )
    # Keep the hand-integrated display sketch intact when retraining the model.
    # The generated header is the only model artifact that must be replaced.
    print(f'rows used for final training: {len(data):,}')
    print(f'freshness parameters: {sum(w.size + b.size for w, b in zip(quantized_freshness["weights"], quantized_freshness["biases"])):,}')
    print(f'fruit parameters: {sum(w.size + b.size for w, b in zip(quantized_fruit["weights"], quantized_fruit["biases"])):,}')
    print(f'header bytes: {(OUTPUT_DIR / "freshness_tinyml_model.h").stat().st_size:,}')


if __name__ == '__main__':
    main()
