"""Banana freshness model and integer inference shared by the notebook and Uno export."""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

DATA_DIR = Path(__file__).resolve().parent / 'datasets'
SENSORS = ('MQ3', 'MQ5', 'MQ135')
FEATURES = (
    'MQ3_ratio_MQ135', 'MQ3_ratio_MQ5', 'MQ3_minus_MQ5',
    'MQ3_minus_MQ135', 'MQ5_ratio_MQ135', 'sensor_std',
    'MQ5_minus_MQ135', 'MQ5_plus_MQ135', 'MQ3',
)
# Per-feature fixed-point precision. The large ratio is limited to Q512;
# all observed training thresholds fit signed int16 values.
FEATURE_Q = (1024, 4096, 16, 16, 512, 16, 16, 16, 16)
GATE_Q = 8
FRESH_CAL_FRACTION = 0.20
ROTTEN_CAL_FRACTION = 0.20
NO_FRUIT_CAL_FRACTION = 0.60
SAMPLES_PER_OPEN_RECORDING = 5000
OPEN_BIN_MS = 1000
OPEN_SESSION_GAP_MS = 60000


def _load_open_raw(filename):
    raw = pd.read_csv(DATA_DIR / filename)
    # The legacy file order is not chronological. Keep equal timestamps stable.
    ticks = pd.to_numeric(raw['Ticks'], errors='raise')
    if ticks.isna().any():
        raise ValueError(f'{filename} contains missing Ticks')
    return raw.assign(Ticks=ticks).sort_values('Ticks', kind='stable').reset_index(drop=True)


def load_recording(filename, *, custom=False):
    if custom:
        raw = pd.read_csv(DATA_DIR / filename)
        columns = ['mq3_adc', 'mq5_adc', 'mq135_adc']
    else:
        raw = _load_open_raw(filename)
        # The legacy MQ5/MQ135 columns are interchanged relative to the logger.
        columns = ['MQ3', 'MQ135', 'MQ5']
    values = raw[columns].copy()
    values.columns = SENSORS
    values = values.apply(pd.to_numeric, errors='raise').reset_index(drop=True)
    if values.isna().any().any():
        raise ValueError(f'{filename} contains missing sensor readings')
    return values


def load_open_resampled(filename):
    """One median sensor reading per observed second, within each open burst.

    Gaps longer than 60 seconds start a new recording session. Empty seconds
    and multi-hour gaps are never filled or interpolated.
    """
    raw = _load_open_raw(filename)
    ticks = raw['Ticks']
    sessions = ticks.diff().gt(OPEN_SESSION_GAP_MS).cumsum()
    starts = ticks.groupby(sessions).transform('first')
    seconds = (ticks - starts) // OPEN_BIN_MS
    grouped = raw.groupby([sessions, seconds], sort=True)[['MQ3', 'MQ135', 'MQ5']]
    values = grouped.median().reset_index(drop=True)
    values.columns = SENSORS
    values.attrs['session_ids'] = tuple(
        grouped.size().index.get_level_values(0).to_numpy()
    )
    return values


def even_sample(frame, maximum=SAMPLES_PER_OPEN_RECORDING):
    return frame.iloc[::max(1, len(frame) // maximum)].head(maximum).copy()


def open_training_data():
    fresh = load_open_resampled('banana_fresh.csv')
    rotten = load_open_resampled('banana_rotten.csv')
    values = pd.concat([fresh, rotten], ignore_index=True)
    values.attrs['session_ids'] = (fresh.attrs['session_ids'] +
                                   rotten.attrs['session_ids'])
    labels = np.r_[np.ones(len(fresh), dtype=np.uint8),
                   np.zeros(len(rotten), dtype=np.uint8)]
    return values, labels, fresh


def open_block_groups(values, labels, *, block_seconds=15):
    """Group adjacent resampled seconds without crossing session/class edges."""
    sessions = values.attrs.get('session_ids')
    if sessions is None or len(sessions) != len(values):
        raise ValueError('Missing open session IDs for blocked validation')
    sessions = np.asarray(sessions)
    groups = np.empty(len(values), dtype=object)
    for label in np.unique(labels):
        for session in np.unique(sessions[labels == label]):
            indexes = np.flatnonzero((labels == label) & (sessions == session))
            groups[indexes] = [f'{label}_{session}_{i // block_seconds}'
                               for i in range(len(indexes))]
    return groups


def open_time_windows(labels, n_windows=5):
    """Five balanced elapsed-time folds within each class's observed bursts."""
    class_windows = [np.array_split(np.flatnonzero(labels == label), n_windows)
                     for label in (1, 0)]
    all_indexes = np.arange(len(labels))
    for window in range(n_windows):
        test = np.r_[class_windows[0][window], class_windows[1][window]]
        train = np.setdiff1d(all_indexes, test)
        yield train, test


def last_session_split(values, labels):
    """Train on earlier bursts, test the final burst of each open class."""
    sessions = values.attrs.get('session_ids')
    if sessions is None or len(sessions) != len(values):
        raise ValueError('Missing open session IDs for session holdout')
    sessions = np.asarray(sessions)
    test = np.zeros(len(values), dtype=bool)
    for label in (1, 0):
        selected = labels == label
        test |= selected & (sessions == np.max(sessions[selected]))
    return np.flatnonzero(~test), np.flatnonzero(test)


def calibration(open_fresh, open_rotten, custom_fresh, custom_rotten, no_fruit):
    """Fit a per-sensor affine custom-to-open mapping from class anchors.

    The first 20% of each custom banana recording supplies one fresh and one
    spoiled median anchor. The remaining rows stay available for validation.
    No-fruit is gated in the raw custom sensor domain before this mapping.
    """
    fresh_end = int(FRESH_CAL_FRACTION * len(custom_fresh))
    rotten_end = int(ROTTEN_CAL_FRACTION * len(custom_rotten))
    no_fruit_end = int(NO_FRUIT_CAL_FRACTION * len(no_fruit))
    custom_fresh_anchor = custom_fresh.iloc[:fresh_end].median()
    custom_rotten_anchor = custom_rotten.iloc[:rotten_end].median()
    denominator = custom_rotten_anchor - custom_fresh_anchor
    if (denominator.abs() < 1e-9).any():
        raise ValueError('Custom fresh and spoiled calibration anchors overlap')
    gains = (open_rotten.median() - open_fresh.median()) / denominator
    offsets = open_fresh.median() - gains * custom_fresh_anchor
    if not np.isfinite(gains).all() or not np.isfinite(offsets).all():
        raise ValueError('Invalid custom-to-open affine calibration')

    no_fruit_max = no_fruit.iloc[:no_fruit_end]['MQ3'].max()
    banana_min = min(custom_fresh.iloc[:fresh_end]['MQ3'].min(),
                     custom_rotten.iloc[:rotten_end]['MQ3'].min())
    if no_fruit_max >= banana_min:
        raise ValueError('No-fruit and custom banana MQ3 calibration ranges overlap')
    threshold = (no_fruit_max + banana_min) / 2
    return gains, offsets, threshold, fresh_end, rotten_end, no_fruit_end


def map_custom(frame, gains, offsets):
    return frame.mul(gains, axis='columns').add(offsets, axis='columns')


def make_features(frame):
    raw = frame.loc[:, SENSORS].astype(float)
    mq3, mq5, mq135 = (raw[name] for name in SENSORS)
    mean = (mq3 + mq5 + mq135) / 3.0
    features = pd.DataFrame(index=frame.index)
    features['MQ3_ratio_MQ135'] = mq3 / (mq135 + 1e-6)
    features['MQ3_ratio_MQ5'] = mq3 / (mq5 + 1e-6)
    features['MQ3_minus_MQ5'] = mq3 - mq5
    features['MQ3_minus_MQ135'] = mq3 - mq135
    features['MQ5_ratio_MQ135'] = mq5 / (mq135 + 1e-6)
    features['sensor_std'] = np.sqrt(
        ((mq3 - mean)**2 + (mq5 - mean)**2 + (mq135 - mean)**2) / 2.0
    )
    features['MQ5_minus_MQ135'] = mq5 - mq135
    features['MQ5_plus_MQ135'] = mq5 + mq135
    features['MQ3'] = mq3
    return features.loc[:, FEATURES]


def make_model():
    # Compact nine-input tree retained as the integer Uno baseline.
    return DecisionTreeClassifier(
        max_depth=7, min_samples_leaf=1, criterion='entropy', random_state=42
    )


def quantize_model(model):
    tree = model.tree_
    if tree.node_count > 255:
        raise ValueError('Tree is too large for uint8 node indexes')
    nodes = []
    for i in range(tree.node_count):
        feature = int(tree.feature[i])
        if feature < 0:
            label = int(model.classes_[np.argmax(tree.value[i, 0])])
            nodes.append((0, 0, 255, label))
        else:
            if tree.children_left[i] != i + 1:
                raise ValueError('Tree nodes are not in preorder')
            threshold = int(np.rint(tree.threshold[i] * FEATURE_Q[feature]))
            if not -32768 <= threshold <= 32767:
                raise ValueError('Threshold exceeds int16 range')
            nodes.append((threshold, int(tree.children_right[i]), feature, 0))
    return nodes


def quantize_features(features):
    scaled = features.loc[:, FEATURES].to_numpy(dtype=float) * np.asarray(FEATURE_Q)
    # Match the Uno's signed rounding and saturate readings outside training range.
    rounded = np.where(scaled >= 0, np.floor(scaled + 0.5), np.ceil(scaled - 0.5))
    return np.clip(rounded, -32768, 32767).astype(np.int16)


def predict_quantized(nodes, mapped_values, *, no_fruit_threshold=None,
                      raw_values=None):
    """Return 0=spoiled, 1=fresh, 2=no_fruit using the Uno's integer decisions."""
    values = quantize_features(make_features(mapped_values))
    result = np.empty(len(values), dtype=np.uint8)
    gate_q = None if no_fruit_threshold is None else int(np.rint(no_fruit_threshold * GATE_Q))
    if no_fruit_threshold is not None and raw_values is None:
        raise ValueError('raw_values are required for the raw no-fruit gate')
    gate_source = mapped_values if raw_values is None else raw_values
    gate_mq3_q = np.floor(gate_source['MQ3'].to_numpy(dtype=float) * GATE_Q + 0.5).astype(np.int32)
    for row_index, row in enumerate(values):
        if gate_q is not None and gate_mq3_q[row_index] < gate_q:
            result[row_index] = 2
            continue
        node_index = 0
        while True:
            threshold, right, feature, fresh = nodes[node_index]
            if feature == 255:
                result[row_index] = fresh
                break
            node_index = node_index + 1 if row[feature] <= threshold else right
    return result


def model_flash_bytes(nodes):
    # packed C node: int16 threshold, uint8 right, uint8 feature, uint8 label
    return len(nodes) * 5
