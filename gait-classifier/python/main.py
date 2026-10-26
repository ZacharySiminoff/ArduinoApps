import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

import numpy as np
from fastapi import HTTPException
from fastapi.responses import FileResponse
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge, Logger

from gait_model import AXES, CHANNELS, LABELS, SAMPLE_RATE_HZ, WINDOW, GaitModels

logger = Logger("gait-classifier")

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
LIVE_STEP = SAMPLE_RATE_HZ // 2   # classify twice a second
SMOOTHING = 3                     # average votes over the last 3 windows (~1.5 s)
LOOP_SECONDS = 0.1
UI_SAMPLE_DECIMATION = 2          # send every 2nd sample (25 Hz) to the live plot

lock = threading.Lock()
live = deque(maxlen=WINDOW)       # newest samples for live classification
new_since_classify = 0
ui_samples = []                   # samples waiting to be sent to the plot
sample_count = 0
imu_address = None                # I2C address reported by the sketch, 0 = not found
rate_hz = 0.0                     # measured sample rate

recording = {"state": "idle"}     # idle | countdown | recording
models = GaitModels()
model_lock = threading.Lock()    # retraining runs on web callback threads
recent_votes = {"model1": deque(maxlen=SMOOTHING), "model2": deque(maxlen=SMOOTHING)}


# ---------- recordings on disk ----------

def list_recordings() -> list[dict]:
    out = []
    for path in sorted(DATA_DIR.glob("*/*.csv")):
        if path.parent.name in LABELS:
            n = sum(1 for _ in path.open()) - 1
            out.append({"name": path.name, "label": path.parent.name, "seconds": round(n / SAMPLE_RATE_HZ, 1)})
    return out


def load_recordings() -> list[tuple[str, np.ndarray]]:
    out = []
    for label in LABELS:
        for path in sorted((DATA_DIR / label).glob("*.csv")):
            data = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
            out.append((label, data))
    return out


def save_recording(label: str, samples: list) -> str:
    folder = DATA_DIR / label
    folder.mkdir(parents=True, exist_ok=True)
    name = f"{label}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.csv"
    np.savetxt(folder / name, np.array(samples), delimiter=",", fmt="%.4f",
               header=",".join(CHANNELS), comments="")
    return name


def find_recording(name: str) -> Path:
    if not re.fullmatch(r"[a-z]+_[0-9-]+\.csv", name):
        raise HTTPException(404)
    path = DATA_DIR / name.split("_")[0] / name
    if not path.exists():
        raise HTTPException(404)
    return path


def retrain():
    recordings = load_recordings()
    with model_lock:
        report = models.train(recordings)
        for v in recent_votes.values():
            v.clear()
    logger.info(f"Trained: model1 {report.get('model1', {}).get('accuracy')}, "
                f"model2 {report.get('model2', {}).get('accuracy')}")


# ---------- web UI ----------

web_ui = WebUI()


def status() -> dict:
    rec = dict(recording)
    rec.pop("samples", None)
    if recording["state"] == "countdown":
        rec["seconds_left"] = max(0.0, recording["starts_at"] - time.time())
    elif recording["state"] == "recording":
        rec["progress"] = len(recording["samples"]) / recording["needed"]
    return {
        "labels": LABELS,
        "axes": AXES,
        "recording": rec,
        "recordings": list_recordings(),
        "model": models.report,
        "imu_address": imu_address,
        "rate_hz": rate_hz,
    }


def broadcast_status():
    web_ui.send_message("status", status())


def on_start_recording(sid, data):
    label = data.get("label")
    if label not in LABELS or recording["state"] != "idle":
        return
    seconds = min(max(float(data.get("seconds", 30)), 5), 300)
    countdown = min(max(float(data.get("countdown", 5)), 0), 60)
    with lock:
        recording.clear()
        recording.update(state="countdown", label=label, seconds=seconds,
                         needed=int(seconds * SAMPLE_RATE_HZ), starts_at=time.time() + countdown, samples=[])
    broadcast_status()


def on_cancel_recording(sid, data):
    with lock:
        recording.clear()
        recording["state"] = "idle"
    broadcast_status()


def on_delete_recording(sid, data):
    find_recording(data.get("name", "")).unlink()
    retrain()
    broadcast_status()


def on_settings(sid, data):
    if "k" in data:
        models.k = min(max(int(data["k"]), 1), 25)
    if data.get("axis") in AXES + ["auto"]:
        models.axis_setting = data["axis"]
    retrain()
    broadcast_status()


web_ui.on_message("start_recording", on_start_recording)
web_ui.on_message("cancel_recording", on_cancel_recording)
web_ui.on_message("delete_recording", on_delete_recording)
web_ui.on_message("settings", on_settings)
web_ui.on_connect(lambda sid: broadcast_status())
web_ui.expose_api("GET", "/api/status", status)
web_ui.expose_api("GET", "/api/download/{name}", lambda name: FileResponse(find_recording(name), filename=name))


# ---------- sensor input from the sketch ----------

def on_imu_sample(ax, ay, az, gx, gy, gz):
    global new_since_classify, sample_count
    s = (ax, ay, az, gx, gy, gz)
    with lock:
        live.append(s)
        new_since_classify += 1
        sample_count += 1
        if sample_count % UI_SAMPLE_DECIMATION == 0:
            ui_samples.append(s)
        if recording["state"] == "recording" and len(recording["samples"]) < recording["needed"]:
            recording["samples"].append(s)


def on_imu_status(address):
    global imu_address
    if address == imu_address:
        return
    imu_address = address
    if address == 0:
        logger.warning("MPU-6050 not found on I2C (check SDA/SCL/3.3V/GND wiring)")
    else:
        logger.info(f"MPU-6050 found at I2C address 0x{address:02x}")


Bridge.provide("imu_sample", on_imu_sample)
Bridge.provide("imu_status", on_imu_status)


# ---------- main loop ----------

def classify_window(window: np.ndarray):
    with model_lock:
        if not models.trained:
            return
        result = models.classify(window)
        pred = {"peak": result["peak"], "axis": models.axis}
        for name in ("model1", "model2"):
            recent_votes[name].append(result[name])
            votes = np.mean(recent_votes[name], axis=0)
            pred[name] = {"label": LABELS[int(votes.argmax())],
                          "votes": {l: float(v) for l, v in zip(LABELS, votes)}}
    web_ui.send_message("prediction", pred)


last_rate_check = (time.time(), 0)
last_status_push = 0.0


def loop():
    global new_since_classify, ui_samples, last_rate_check, last_status_push, rate_hz
    now = time.time()
    finished = None
    window = None

    with lock:
        if ui_samples:
            web_ui.send_message("samples", ui_samples)
            ui_samples = []

        if recording["state"] == "countdown" and now >= recording["starts_at"]:
            recording["state"] = "recording"
            recording["samples"] = []
        elif recording["state"] == "recording" and len(recording["samples"]) >= recording["needed"]:
            finished = (recording["label"], recording["samples"])
            recording.clear()
            recording["state"] = "idle"

        if models.trained and len(live) == WINDOW and new_since_classify >= LIVE_STEP:
            new_since_classify = 0
            window = np.array(live)

    if finished:
        name = save_recording(*finished)
        logger.info(f"Saved {name}")
        retrain()
        broadcast_status()
    elif recording["state"] != "idle" and now - last_status_push > 0.5:
        broadcast_status()   # countdown / progress updates
        last_status_push = now

    if window is not None:
        classify_window(window)

    t0, n0 = last_rate_check
    if now - t0 >= 2:
        rate_hz = (sample_count - n0) / (now - t0)
        web_ui.send_message("rate", {"hz": rate_hz, "imu_address": imu_address})
        last_rate_check = (now, sample_count)

    time.sleep(LOOP_SECONDS)


DATA_DIR.mkdir(parents=True, exist_ok=True)
retrain()
App.run(user_loop=loop)
