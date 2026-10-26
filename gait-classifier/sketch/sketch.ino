// Reads an MPU-6050 on the header I2C pins (SDA/SCL, 3.3V, GND) at a fixed
// 50 Hz and streams every sample to Python over the Bridge.
// Units sent: acceleration in g, angular rate in degrees/second.

#include <Arduino_RouterBridge.h>
#include <Wire.h>

const unsigned long SAMPLE_PERIOD_US = 20000;  // 50 Hz, must match SAMPLE_RATE_HZ in main.py

// MPU-6050 registers
const uint8_t REG_SMPLRT_DIV = 0x19;
const uint8_t REG_CONFIG = 0x1A;
const uint8_t REG_GYRO_CONFIG = 0x1B;
const uint8_t REG_ACCEL_CONFIG = 0x1C;
const uint8_t REG_ACCEL_XOUT_H = 0x3B;
const uint8_t REG_PWR_MGMT_1 = 0x6B;

// +-16 g and +-2000 dps: running impacts at the ankle can pass 8 g
const float ACCEL_LSB_PER_G = 2048.0;
const float GYRO_LSB_PER_DPS = 16.4;

uint8_t mpuAddr = 0;
unsigned long nextSampleUs = 0;

void writeReg(uint8_t reg, uint8_t value) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(reg);
  Wire.write(value);
  Wire.endTransmission();
}

bool findMpu() {
  // AD0 low -> 0x68, AD0 high -> 0x69
  const uint8_t candidates[] = {0x68, 0x69};
  for (uint8_t addr : candidates) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      mpuAddr = addr;
      return true;
    }
  }
  return false;
}

void setupMpu() {
  writeReg(REG_PWR_MGMT_1, 0x01);    // wake up, clock from X gyro PLL
  delay(50);
  writeReg(REG_SMPLRT_DIV, 0x00);
  writeReg(REG_CONFIG, 0x04);        // low-pass ~21 Hz (anti-aliasing for 50 Hz sampling)
  writeReg(REG_GYRO_CONFIG, 0x18);   // +-2000 dps
  writeReg(REG_ACCEL_CONFIG, 0x18);  // +-16 g
}

bool readMpu(float out[6]) {
  Wire.beginTransmission(mpuAddr);
  Wire.write(REG_ACCEL_XOUT_H);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom(mpuAddr, (uint8_t)14) != 14) return false;

  int16_t raw[7];  // ax ay az temp gx gy gz
  for (int i = 0; i < 7; i++) {
    uint8_t hi = Wire.read();
    uint8_t lo = Wire.read();
    raw[i] = (int16_t)((hi << 8) | lo);
  }
  out[0] = raw[0] / ACCEL_LSB_PER_G;
  out[1] = raw[1] / ACCEL_LSB_PER_G;
  out[2] = raw[2] / ACCEL_LSB_PER_G;
  out[3] = raw[4] / GYRO_LSB_PER_DPS;
  out[4] = raw[5] / GYRO_LSB_PER_DPS;
  out[5] = raw[6] / GYRO_LSB_PER_DPS;
  return true;
}

void setup() {
  Bridge.begin();
  Wire.begin();
  Wire.setClock(400000);

  while (!findMpu()) {
    Bridge.notify("imu_status", 0);
    delay(1000);
  }
  setupMpu();
  Bridge.notify("imu_status", (int)mpuAddr);
  nextSampleUs = micros();
}

void loop() {
  unsigned long now = micros();
  if ((long)(now - nextSampleUs) < 0) return;

  nextSampleUs += SAMPLE_PERIOD_US;
  // If we fell far behind (e.g. a slow Bridge call), resync instead of bursting
  if ((long)(now - nextSampleUs) > (long)(5 * SAMPLE_PERIOD_US)) {
    nextSampleUs = now + SAMPLE_PERIOD_US;
  }

  // Re-announce the sensor every few seconds so Python learns it after a restart
  static unsigned long lastStatusMs = 0;
  if (millis() - lastStatusMs > 3000) {
    lastStatusMs = millis();
    Bridge.notify("imu_status", (int)mpuAddr);
  }

  float s[6];
  if (readMpu(s)) {
    Bridge.notify("imu_sample", s[0], s[1], s[2], s[3], s[4], s[5]);
  } else {
    Bridge.notify("imu_status", 0);
    if (findMpu()) setupMpu();
  }
}
