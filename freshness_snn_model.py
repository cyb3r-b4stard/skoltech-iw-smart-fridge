"""Deterministic nine-feature spiking banana-freshness model for Arduino Uno."""

from dataclasses import dataclass

import numpy as np
from sklearn.linear_model import LogisticRegression

from freshness_tree_model import FEATURES, GATE_Q, make_features

INPUT_Q = 256
READOUT_Q = 4096
DECAY_Q8 = 192
THRESHOLD_Q = 1024
TIME_STEPS = 8
DEFAULT_NEURONS = len(FEATURES)
DEFAULT_EPOCHS = 0
SPOILED_MARGIN_PERCENTILE = 80


@dataclass
class QuantizedSNN:
    center: np.ndarray
    inv_scale: np.ndarray
    readout: np.ndarray
    output_bias: int

    @property
    def neurons(self):
        return len(self.readout)


def _round_signed(values):
    values = np.asarray(values)
    return np.where(values >= 0, np.floor(values + .5), np.ceil(values - .5))


def _checked_int16(values, name):
    rounded = _round_signed(values)
    if (not np.isfinite(rounded).all() or (rounded < -32768).any()
            or (rounded > 32767).any()):
        raise ValueError(f'{name} exceeds signed int16 range')
    return rounded.astype(np.int16)


def _trunc_div(values, divisor):
    """Signed division with C/C++ truncation toward zero."""
    values = np.asarray(values)
    return np.where(values >= 0, values // divisor, -((-values) // divisor))


def _normalized_inputs(center, inv_scale, features):
    if list(features.columns) != list(FEATURES):
        raise ValueError('SNN requires the nine engineered features in fixed order')
    values = features.to_numpy(dtype=np.float32)
    normalized = np.clip((values - center) * inv_scale, -4, 4)
    return _checked_int16(normalized * INPUT_Q, 'inputs').astype(np.int32)


def _rate_counts_from_inputs(inputs):
    """Encode each feature with one fixed eight-step LIF neuron."""
    drive = _trunc_div(inputs + 4 * INPUT_Q, 4)
    membrane = np.zeros_like(drive, dtype=np.int32)
    counts = np.zeros_like(drive, dtype=np.uint8)
    for _ in range(TIME_STEPS):
        membrane = _trunc_div(membrane * DECAY_Q8, 256) + drive
        spikes = membrane >= THRESHOLD_Q
        counts += spikes.astype(np.uint8)
        membrane -= spikes.astype(np.int32) * THRESHOLD_Q
    return counts


def fit_snn(features, labels, *, neurons=DEFAULT_NEURONS, epochs=DEFAULT_EPOCHS,
            seed=42):
    """Fit a deterministic quantized readout with a recording-drift margin.

    The threshold is the 80th percentile of spoiled training scores. It is
    fitted without the held-out session and intentionally reserves margin for
    the large between-session sensor shift present in the open dataset.
    """
    del epochs, seed
    if neurons != len(FEATURES):
        raise ValueError(f'This architecture has exactly {len(FEATURES)} feature neurons')
    labels = np.asarray(labels, dtype=np.uint8)
    values = features.to_numpy(dtype=np.float32)
    center = values.mean(axis=0).astype(np.float32)
    scale = values.std(axis=0).astype(np.float32)
    scale[scale == 0] = 1
    inv_scale = np.float32(1) / scale
    counts = _rate_counts_from_inputs(_normalized_inputs(center, inv_scale, features))

    readout = LogisticRegression(
        C=.01, class_weight='balanced', max_iter=5000, random_state=42
    ).fit(counts, labels)
    scores = readout.decision_function(counts)
    spoiled_threshold = float(np.percentile(
        scores[labels == 0], SPOILED_MARGIN_PERCENTILE
    ))
    weights_q = _checked_int16(readout.coef_[0] * READOUT_Q, 'readout')
    bias_q = int(_round_signed(
        (readout.intercept_[0] - spoiled_threshold) * READOUT_Q
    ))
    return QuantizedSNN(center, inv_scale, weights_q, bias_q)


def spike_counts(model, features):
    inputs = _normalized_inputs(model.center, model.inv_scale, features)
    return _rate_counts_from_inputs(inputs)


def decision_scores(model, mapped_values):
    counts = spike_counts(model, make_features(mapped_values))
    return counts.astype(np.int32) @ model.readout.astype(np.int32) + model.output_bias


def predict_snn(model, mapped_values, *, no_fruit_threshold=None,
                raw_values=None):
    """Return 0=spoiled, 1=fresh, 2=no_fruit using integer SNN inference."""
    predictions = (decision_scores(model, mapped_values) >= 0).astype(np.uint8)
    if no_fruit_threshold is not None:
        if raw_values is None:
            raise ValueError('raw_values are required for the raw no-fruit gate')
        gate_q = int(np.rint(no_fruit_threshold * GATE_Q))
        mq3_q = _round_signed(raw_values['MQ3'].to_numpy(dtype=float) * GATE_Q)
        predictions[mq3_q < gate_q] = 2
    return predictions


def model_flash_bytes(model):
    return (model.readout.nbytes + model.center.nbytes
            + model.inv_scale.nbytes + 4)
