# 🏃 Gait Classifier

A wearable built from an Arduino UNO Q and an MPU-6050 IMU. It classifies
**standing still, walking, running, and biking** (biking can also be
air-biking on your back). It uses two k-nearest-neighbour models and no
neural nets.

## Hardware

| MPU-6050 | UNO Q          |
|----------|----------------|
| VCC      | 3.3V           |
| GND      | GND            |
| SCL      | SCL (header)   |
| SDA      | SDA (header)   |

Strap the board and sensor to the **outside of the ankle**, with the sensor
held firmly against the leg so it can't flop around. Use the same placement
and orientation for training and testing. Power the board from a USB power
bank and open the web page on a phone on the same Wi-Fi.

## How it works

**Sampling (sketch):** the STM32 reads the MPU-6050 over I2C at **50 Hz**, with
±16 g and ±2000 °/s ranges and the sensor's ~21 Hz low-pass filter. Walking,
running, and pedalling all fall below about 5 Hz (impact harmonics go up to
about 15–20 Hz), so 50 Hz captures them without aliasing. Each sample goes to
Python over the Bridge.

**Windows:** the signal is cut into **2.56 s windows (128 samples)**, which is
long enough to hold at least two strides or pedal turns. Training windows
overlap by 50%. The live classifier runs every 0.5 s and averages the kNN
votes over the last 3 windows.

**Model 1 (peak, one direction):** a single feature, the peak |acceleration|
along one axis in the window. The axis is chosen automatically by
cross-validation, or you can pick it on the web page.

**Model 2 (all IMU outputs):** 34 features: mean, std, min, and max of all six
channels (accel x/y/z, gyro x/y/z); mean, std, and max of |a| and |ω|; and the
dominant frequency of |a| and |ω| with its share of the spectral power.

Both models use **kNN (k = 5 by default)** on z-scored features with a majority
vote, written in numpy (`python/gait_model.py`).

**Evaluation:** blocked 5-fold cross-validation. Each recording is split into
5 contiguous time blocks, and each block is predicted by a model trained on
everything else. The web page shows accuracy and a confusion matrix for both
models.

## Using it

1. Deploy: `python3 tools/deploy.py gait-classifier`, then open
   `http://<board-ip>:7000`.
2. Under **Record training data**, choose a length (30 s default) and a
   countdown. Tap **Record walk**, put the phone away, and walk until the
   progress bar finishes. Do the same for still, run, and bike. More than one
   recording per activity (for example, on different days) makes the models
   more robust.
3. The models retrain automatically after every recording. **Current gait**
   shows both models' live predictions.

Recordings are saved on the board as
`data/<activity>/<activity>_<timestamp>.csv` with columns
`ax,ay,az,gx,gy,gz` (g and °/s, 50 Hz). Download them from the web page or
delete bad takes there.
