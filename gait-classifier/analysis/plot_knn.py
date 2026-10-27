#!/usr/bin/env python3
"""Draw the kNN models as PNGs from the recorded training data.

Runs on your computer, not the board (needs numpy and matplotlib):

    python3 gait-classifier/analysis/plot_knn.py --fetch   # download recordings from the board, then plot
    python3 gait-classifier/analysis/plot_knn.py           # plot the recordings already downloaded

Recordings are kept in analysis/data/<activity>/*.csv and figures are
written to analysis/figures/.
"""
import argparse
import base64
import io
import subprocess
import sys
import tarfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import ListedColormap

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
sys.path.insert(0, str(HERE.parent / "python"))
from gait_model import (AXES, CHANNELS, CV_FOLDS, LABELS, TRAIN_STEP, KNN, GaitModels,  # noqa: E402
                        cross_validate, full_features, make_windows, peak_features)

DATA_DIR = HERE / "data"
OUT_DIR = HERE / "figures"
K = 5

# Validated (dataviz validate_palette.js, light surface, all pairs):
# worst normal-vision dE 19.6, worst protan/deutan dE 16.2. Shapes are a
# second cue so no activity is identified by color alone.
COLORS = {"still": "#6250d6", "walk": "#008300", "run": "#e87ba4", "bike": "#eda100"}
MARKERS = {"still": "o", "walk": "s", "run": "^", "bike": "D"}
SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": GRID, "axes.labelcolor": TEXT_2, "text.color": TEXT,
    "xtick.color": TEXT_2, "ytick.color": TEXT_2, "font.size": 11,
    "axes.titlesize": 14, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.spines.top": False, "axes.spines.right": False,
})


# ---------- data ----------

def fetch_from_board():
    """Copy the board's data/ folder here, through tools/deploy.py."""
    cmd = [sys.executable, str(ROOT / "tools" / "deploy.py"), "--cmd",
           "echo @@BEGIN@@; tar czf - -C /home/arduino/ArduinoApps/gait-classifier data | base64 -w0; echo; echo @@END@@"]
    out = subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines()
    if "@@BEGIN@@" not in out:
        sys.exit("Couldn't download the recordings from the board (is it reachable?)")
    payload = out[out.index("@@BEGIN@@") + 1]
    with tarfile.open(fileobj=io.BytesIO(base64.b64decode(payload))) as tar:
        tar.extractall(HERE, filter="data")
    print(f"Downloaded recordings to {DATA_DIR}")


def load_windows():
    """All training windows, labels, and CV folds, exactly as the app builds them."""
    recordings, windows, labels, folds = [], [], [], []
    for label in LABELS:
        for path in sorted((DATA_DIR / label).glob("*.csv")):
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            recordings.append((label, data))
            w = make_windows(data, TRAIN_STEP)
            if len(w):
                windows.append(w)
                labels += [LABELS.index(label)] * len(w)
                folds.append((np.arange(len(w)) * CV_FOLDS) // len(w))
    if not windows:
        sys.exit(f"No recordings in {DATA_DIR}. Run with --fetch first.")
    return recordings, np.concatenate(windows), np.array(labels), np.concatenate(folds)


# ---------- drawing helpers ----------

def present_labels(y):
    return [LABELS[i] for i in sorted(set(y))]


def region_cmap():
    """Very light tints of each activity's color for the decision-region background."""
    tints = []
    for label in LABELS:
        rgb = np.array(matplotlib.colors.to_rgb(COLORS[label]))
        tints.append(tuple(rgb * 0.16 + np.array(matplotlib.colors.to_rgb(SURFACE)) * 0.84))
    return ListedColormap(tints)


def scatter_classes(ax, P, y, size=46):
    for i, label in enumerate(LABELS):
        pts = P[y == i]
        if len(pts):
            ax.scatter(pts[:, 0], pts[:, 1], s=size, marker=MARKERS[label], c=COLORS[label],
                       edgecolors=SURFACE, linewidths=1.2, label=label, zorder=3)


def label_clusters(ax, P, y):
    """Direct label next to each activity's cluster, in text ink (not series color).

    Tries above / below / left / right of the cluster and uses the spot
    farthest from other activities' points, so labels don't sit on them."""
    span = P.max(axis=0) - P.min(axis=0)
    for i, label in enumerate(LABELS):
        pts = P[y == i]
        others = P[y != i]
        if not len(pts):
            continue
        cx, cy = np.median(pts, axis=0)
        (x0, y0), (x1, y1) = pts.min(axis=0), pts.max(axis=0)
        gap = 0.05 * span
        candidates = [  # (anchor point, horizontal alignment, vertical alignment)
            ((cx, y1 + gap[1]), "center", "bottom"),
            ((cx, y0 - gap[1]), "center", "top"),
            ((x1 + gap[0], cy), "left", "center"),
            ((x0 - gap[0], cy), "right", "center"),
        ]

        def clearance(c):
            if not len(others):
                return np.inf
            return np.min(np.linalg.norm((others - np.array(c[0])) / span, axis=1))

        (px, py), ha, va = max(candidates, key=clearance)
        ax.text(px, py, label, ha=ha, va=va, fontsize=11, fontweight="bold", color=TEXT, zorder=4)


def decision_regions(ax, P, y, pad=0.08, steps=400):
    """Shade the background by what kNN (trained on these 2 coordinates) predicts there."""
    lo, hi = P.min(axis=0), P.max(axis=0)
    span = hi - lo
    lo, hi = lo - span * pad, hi + span * pad
    gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], steps), np.linspace(lo[1], hi[1], steps))
    grid = np.column_stack([gx.ravel(), gy.ravel()])
    model = KNN(K).fit(P, y)
    pred = np.concatenate([model.predict(chunk) for chunk in np.array_split(grid, 40)])
    ax.imshow(pred.reshape(gx.shape), origin="lower", extent=(lo[0], hi[0], lo[1], hi[1]),
              cmap=region_cmap(), vmin=-0.5, vmax=len(LABELS) - 0.5, aspect="auto",
              interpolation="nearest", zorder=0)
    ax.set_xlim(lo[0], hi[0])
    ax.set_ylim(lo[1], hi[1])


def legend(ax, **kw):
    ax.legend(frameon=False, labelcolor=TEXT, handletextpad=0.3, **kw)


def subtitle(ax, text):
    ax.text(0, 1.015, text, transform=ax.transAxes, fontsize=10, color=TEXT_2, va="bottom")


# ---------- figures ----------

def plot_model1(ax, W, y, folds):
    """Model 1 is one-dimensional, so it can be drawn exactly: a number line."""
    models = GaitModels(k=K)
    models.train(RECORDINGS)
    axis = models.axis
    peaks = peak_features(W, AXES.index(axis))[:, 0]
    acc = cross_validate(peaks[:, None], y, folds, K)["accuracy"]

    # decision regions along the line (kNN on the single feature)
    lo, hi = peaks.min(), peaks.max()
    pad = (hi - lo) * 0.05
    xs = np.linspace(lo - pad, hi + pad, 1200)
    pred = KNN(K).fit(peaks[:, None], y).predict(xs[:, None])
    labels = present_labels(y)
    rows = {label: r for r, label in enumerate(reversed(labels))}
    ax.imshow(pred[None, :], extent=(xs[0], xs[-1], -0.6, len(labels) - 0.4), aspect="auto",
              cmap=region_cmap(), vmin=-0.5, vmax=len(LABELS) - 0.5, interpolation="nearest", zorder=0)

    rng = np.random.default_rng(0)
    for i, label in enumerate(LABELS):
        pts = peaks[y == i]
        if len(pts):
            jitter = rng.uniform(-0.22, 0.22, len(pts))
            ax.scatter(pts, rows[label] + jitter, s=46, marker=MARKERS[label], c=COLORS[label],
                       edgecolors=SURFACE, linewidths=1.2, zorder=3)

    # the walk/bike overlap that causes most of Model 1's mistakes
    if "walk" in labels and "bike" in labels:
        walk, bike = peaks[y == LABELS.index("walk")], peaks[y == LABELS.index("bike")]
        o_lo, o_hi = max(walk.min(), bike.min()), min(walk.max(), bike.max())
        if o_lo < o_hi:
            ax.vlines([o_lo, o_hi], -0.6, len(labels) - 0.4, colors=TEXT_2,
                      linestyles=(0, (4, 3)), linewidth=1.2, zorder=2)
            ax.annotate(f"walk / bike overlap {o_lo:.2f}–{o_hi:.2f} g", ((o_lo + o_hi) / 2, len(labels) - 0.4),
                        xytext=(0, 6), textcoords="offset points", ha="center", va="bottom",
                        fontsize=9.5, color=TEXT_2)

    ax.set_yticks(range(len(labels)), [l for l in reversed(labels)])
    ax.tick_params(axis="y", length=0, labelsize=11.5)
    for t in ax.get_yticklabels():
        t.set_color(TEXT)
    ax.set_ylim(-0.6, len(labels) - 0.4 + 0.5)
    ax.set_xlim(xs[0], xs[-1])
    along = " (along the shin)" if axis == "x" else ""
    ax.set_xlabel(f"hardest jolt in the window: peak acceleration on the {axis} axis{along}, in g")
    ax.spines["left"].set_visible(False)
    ax.set_title("Model 1: peak acceleration on one axis", pad=22)
    subtitle(ax, f"1 feature, drawn exactly · shading = kNN (k={K}) prediction · CV accuracy {acc:.0%}")
    return acc


def pca_2d(X):
    Z = (X - X.mean(axis=0)) / np.where(X.std(axis=0) < 1e-9, 1, X.std(axis=0))
    Z = Z - Z.mean(axis=0)
    _, s, Vt = np.linalg.svd(Z, full_matrices=False)
    explained = s ** 2 / (s ** 2).sum()
    # PCA picks each axis's sign arbitrarily; pin them so the plain-language
    # labels stay true: right = more movement, up = steadier rhythm.
    # (On our data PC1 loads on every shaking/rotation feature and PC2 mostly
    # on how much of the motion is one repeating beat.)
    names = feature_names()
    if Vt[0, names.index("acc_mag_std")] < 0:
        Vt[0] *= -1
    if Vt[1, names.index("acc_freq_share")] < 0:
        Vt[1] *= -1
    return Z @ Vt[:2].T, explained[:2]


def plot_model2_pca(ax, W, y, folds):
    X = full_features(W)
    P, explained = pca_2d(X)
    acc_full = cross_validate(X, y, folds, K)["accuracy"]
    acc_2d = cross_validate(P, y, folds, K)["accuracy"]

    decision_regions(ax, P, y)
    scatter_classes(ax, P, y)
    label_clusters(ax, P, y)
    ax.set_xlabel("how much the leg is moving  →\n(shaking + rotation combined)")
    ax.set_ylabel("how steady the rhythm is  →\n(share of motion in one repeating beat)")
    ax.set_title("Model 2: all 34 IMU features, squeezed into 2 summary axes", pad=22)
    subtitle(ax, f"axes found by PCA, keeping {explained.sum():.0%} of the information · "
                 f"kNN on these 2 axes {acc_2d:.0%}, real 34-D model {acc_full:.0%}")
    ax.tick_params(labelbottom=False, labelleft=False, length=0)  # PCA units have no physical meaning
    legend(ax, loc="lower right", fontsize=10)
    return explained, acc_2d, acc_full


def plot_distance_heatmap(W, y):
    X = full_features(W)
    model = KNN(K).fit(X, y)
    Z = model.X  # z-scored features, exactly what kNN measures distance on
    D = np.sqrt(((Z[:, None, :] - Z[None, :, :]) ** 2).sum(axis=2))

    fig, ax = plt.subplots(figsize=(8.6, 7.6))
    # dark = close: a dark block on the diagonal means that activity's windows sit together
    img = ax.imshow(D, cmap="Blues_r", interpolation="nearest")
    bounds = [0] + list(np.cumsum([np.sum(y == i) for i in range(len(LABELS)) if np.any(y == i)]))
    labels = present_labels(y)
    for b in bounds[1:-1]:
        ax.axhline(b - 0.5, color=SURFACE, linewidth=2)
        ax.axvline(b - 0.5, color=SURFACE, linewidth=2)
    centers = [(a + b - 1) / 2 for a, b in zip(bounds[:-1], bounds[1:])]
    ax.set_xticks(centers, labels)
    ax.set_yticks(centers, labels)
    ax.tick_params(length=0, labelsize=11.5)
    for t in ax.get_xticklabels() + ax.get_yticklabels():
        t.set_color(TEXT)
    for side in ax.spines.values():
        side.set_visible(False)
    cbar = fig.colorbar(img, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("how different two windows are (darker = more alike)", color=TEXT_2)
    cbar.outline.set_visible(False)
    ax.set_title("Model 2: distance between every pair of windows", pad=22)
    subtitle(ax, f"{len(y)} windows sorted by activity · the exact distances kNN uses, no projection")
    fig.tight_layout()
    return fig


def plot_two_features(W, y, folds):
    """Walk vs. bike only: the pair Model 1 confuses, on the two features that split them best."""
    keep = np.isin(y, [LABELS.index("walk"), LABELS.index("bike")])
    X = full_features(W[keep])
    y, folds = y[keep], folds[keep]
    names = feature_names()
    P = X[:, [names.index("gyro_freq"), names.index("mean_az")]]
    acc = cross_validate(P, y, folds, K)["accuracy"]

    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    decision_regions(ax, P, y, pad=0.15)   # KNN z-scores internally, so real units are fine here
    scatter_classes(ax, P, y)
    label_clusters(ax, P, y)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: f"{v + 0:.2f}"))
    ax.set_xlabel("how often the leg's rotation repeats (times per second)")
    ax.set_ylabel("how the shin is tilted\n(average gravity on the z axis, g)")
    ax.set_title("Why Model 2 can tell walking from biking", pad=22)
    subtitle(ax, f"walk and bike windows only · just 2 of the 34 features · CV accuracy {acc:.0%}")
    legend(ax, loc="lower right", fontsize=10)
    fig.tight_layout()
    return fig, acc


def feature_names():
    """Names of the 34 Model 2 features, in the order full_features() returns them."""
    names = [f"{stat}_{c}" for stat in ("mean", "std", "min", "max") for c in CHANNELS]
    names += ["acc_mag_mean", "acc_mag_std", "acc_mag_max", "gyro_mag_mean", "gyro_mag_std", "gyro_mag_max",
              "acc_freq", "acc_freq_share", "gyro_freq", "gyro_freq_share"]
    return names


def main():
    global RECORDINGS
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fetch", action="store_true", help="download the recordings from the board first")
    args = parser.parse_args()
    if args.fetch:
        fetch_from_board()

    RECORDINGS, W, y, folds = load_windows()
    OUT_DIR.mkdir(exist_ok=True)
    print(f"{len(RECORDINGS)} recordings, {len(y)} windows")

    fig, ax = plt.subplots(figsize=(9, 5))
    acc1 = plot_model1(ax, W, y, folds)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "1_model1_number_line.png", dpi=200)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.6, 6.8))
    explained, acc_2d, acc2 = plot_model2_pca(ax, W, y, folds)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "2_model2_pca.png", dpi=200)
    plt.close(fig)

    fig = plot_distance_heatmap(W, y)
    fig.savefig(OUT_DIR / "3_model2_distance_heatmap.png", dpi=200)
    plt.close(fig)

    fig, acc_pair = plot_two_features(W, y, folds)
    fig.savefig(OUT_DIR / "4_two_features.png", dpi=200)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(17, 6.6), gridspec_kw={"width_ratios": [1, 1]})
    plot_model1(a1, W, y, folds)
    plot_model2_pca(a2, W, y, folds)
    fig.tight_layout(w_pad=4)
    fig.savefig(OUT_DIR / "0_models_side_by_side.png", dpi=200)
    plt.close(fig)

    print(f"Model 1 CV accuracy {acc1:.1%}")
    print(f"Model 2 CV accuracy {acc2:.1%} (PCA keeps {explained.sum():.1%}, kNN on 2 PCs {acc_2d:.1%})")
    print(f"Walk vs. bike on 2 features {acc_pair:.1%}")
    print(f"Figures written to {OUT_DIR}")


if __name__ == "__main__":
    main()
