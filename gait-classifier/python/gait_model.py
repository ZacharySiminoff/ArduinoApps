"""Windowing, features, and k-nearest-neighbour classifiers for the gait app.

Two models are trained on the same windows:
  * Model 1 ("peak"): one feature, the peak |acceleration| along a single axis.
  * Model 2 ("full"): summary statistics of all six IMU channels plus the
    dominant stride/pedal frequency.
Both use plain kNN (majority vote of the k closest training windows) on
z-scored features. No neural nets, no scikit-learn (it isn't in the board image).
"""
import numpy as np

SAMPLE_RATE_HZ = 50       # must match SAMPLE_PERIOD_US in sketch.ino
WINDOW = 128              # 2.56 s: at least two strides / pedal turns
TRAIN_STEP = WINDOW // 2  # 50% overlap between training windows
CV_FOLDS = 5

LABELS = ["still", "walk", "run", "bike"]
CHANNELS = ["ax", "ay", "az", "gx", "gy", "gz"]
AXES = ["x", "y", "z"]


def make_windows(data: np.ndarray, step: int) -> np.ndarray:
    """Cut an (n_samples, 6) recording into (n_windows, WINDOW, 6)."""
    starts = range(0, len(data) - WINDOW + 1, step)
    return np.array([data[s:s + WINDOW] for s in starts]).reshape(-1, WINDOW, data.shape[1])


def peak_features(windows: np.ndarray, axis: int) -> np.ndarray:
    """Model 1: peak absolute acceleration along one axis, in g."""
    return np.abs(windows[:, :, axis]).max(axis=1, keepdims=True)


def _dominant_frequency(x: np.ndarray) -> np.ndarray:
    """Frequency (Hz) of the strongest periodic component, and its share of the power."""
    x = x - x.mean(axis=1, keepdims=True)
    spectrum = np.abs(np.fft.rfft(x * np.hanning(x.shape[1]), axis=1)) ** 2
    spectrum[:, 0] = 0
    freqs = np.fft.rfftfreq(x.shape[1], 1 / SAMPLE_RATE_HZ)
    peak = spectrum.argmax(axis=1)
    share = spectrum.max(axis=1) / (spectrum.sum(axis=1) + 1e-9)
    return np.column_stack([freqs[peak], share])


def full_features(windows: np.ndarray) -> np.ndarray:
    """Model 2: statistics of every IMU channel (accel x/y/z, gyro x/y/z)."""
    acc_mag = np.linalg.norm(windows[:, :, :3], axis=2)
    gyro_mag = np.linalg.norm(windows[:, :, 3:], axis=2)
    return np.column_stack([
        windows.mean(axis=1),   # orientation (gravity direction) and gyro bias
        windows.std(axis=1),    # how hard each channel is moving
        windows.min(axis=1),
        windows.max(axis=1),
        acc_mag.mean(axis=1), acc_mag.std(axis=1), acc_mag.max(axis=1),
        gyro_mag.mean(axis=1), gyro_mag.std(axis=1), gyro_mag.max(axis=1),
        _dominant_frequency(acc_mag),
        _dominant_frequency(gyro_mag),
    ])


FULL_FEATURE_COUNT = full_features(np.zeros((1, WINDOW, 6))).shape[1]


class KNN:
    """k-nearest-neighbour classifier with z-score scaling and majority vote."""

    def __init__(self, k: int = 5):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "KNN":
        self.mean = X.mean(axis=0)
        self.scale = X.std(axis=0)
        self.scale[self.scale < 1e-9] = 1.0
        self.X = (X - self.mean) / self.scale
        self.y = y
        return self

    def votes(self, X: np.ndarray) -> np.ndarray:
        """Fraction of the k neighbours voting for each label, shape (n, len(LABELS))."""
        Z = (np.atleast_2d(X) - self.mean) / self.scale
        dist = ((Z[:, None, :] - self.X[None, :, :]) ** 2).sum(axis=2)
        k = min(self.k, len(self.y))
        nearest = np.argpartition(dist, k - 1, axis=1)[:, :k]
        out = np.zeros((len(Z), len(LABELS)))
        for i, row in enumerate(self.y[nearest]):
            out[i] = np.bincount(row, minlength=len(LABELS)) / k
        return out

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.votes(X).argmax(axis=1)


def cross_validate(X: np.ndarray, y: np.ndarray, folds: np.ndarray, k: int) -> dict:
    """Blocked k-fold: each recording is split into CV_FOLDS contiguous time
    blocks, so test windows come from a different part of the recording than
    the training windows next to them."""
    pred = np.full(len(y), -1)
    for f in range(CV_FOLDS):
        test = folds == f
        if test.any() and (~test).any():
            pred[test] = KNN(k).fit(X[~test], y[~test]).predict(X[test])
    scored = pred >= 0
    confusion = np.zeros((len(LABELS), len(LABELS)), dtype=int)
    np.add.at(confusion, (y[scored], pred[scored]), 1)
    accuracy = float((pred[scored] == y[scored]).mean()) if scored.any() else 0.0
    return {"accuracy": accuracy, "confusion": confusion.tolist()}


class GaitModels:
    """Both models, trained from a list of (label, (n_samples, 6) array) recordings."""

    def __init__(self, k: int = 5, axis: str = "auto"):
        self.k = k
        self.axis_setting = axis
        self.trained = False
        self.report = {}

    def train(self, recordings: list[tuple[str, np.ndarray]]) -> dict:
        windows, labels, folds = [], [], []
        for label, data in recordings:
            w = make_windows(data, TRAIN_STEP)
            if len(w) == 0:
                continue
            windows.append(w)
            labels += [LABELS.index(label)] * len(w)
            folds.append((np.arange(len(w)) * CV_FOLDS) // len(w))

        present = sorted(set(labels))
        self.trained = len(present) >= 2
        self.report = {
            "trained": self.trained,
            "k": self.k,
            "axis_setting": self.axis_setting,
            "windows_per_label": {l: labels.count(i) for i, l in enumerate(LABELS)},
        }
        if not self.trained:
            return self.report

        W = np.concatenate(windows)
        y = np.array(labels)
        folds = np.concatenate(folds)

        # Model 1: evaluate every axis, use the chosen (or best) one
        axis_results = {a: cross_validate(peak_features(W, i), y, folds, self.k) for i, a in enumerate(AXES)}
        if self.axis_setting in AXES:
            self.axis = self.axis_setting
        else:
            self.axis = max(AXES, key=lambda a: axis_results[a]["accuracy"])
        self.peak_model = KNN(self.k).fit(peak_features(W, AXES.index(self.axis)), y)

        X2 = full_features(W)
        self.full_model = KNN(self.k).fit(X2, y)

        self.report.update({
            "axis": self.axis,
            "model1": {**axis_results[self.axis],
                       "axis_accuracy": {a: r["accuracy"] for a, r in axis_results.items()}},
            "model2": cross_validate(X2, y, folds, self.k),
            "peak_by_label": {
                LABELS[i]: _summary(peak_features(W[y == i], AXES.index(self.axis))[:, 0])
                for i in present
            },
        })
        return self.report

    def classify(self, window: np.ndarray) -> dict:
        """Votes from both models for one (WINDOW, 6) window."""
        W = window[None]
        peak = float(peak_features(W, AXES.index(self.axis))[0, 0])
        return {
            "peak": peak,
            "model1": self.peak_model.votes(np.array([[peak]]))[0],
            "model2": self.full_model.votes(full_features(W))[0],
        }


def _summary(x: np.ndarray) -> dict:
    return {"min": float(x.min()), "median": float(np.median(x)), "max": float(x.max())}
