# AI-Powered Exercise Recognition Glove

An embedded TinyML project that uses motion sensor data and machine learning to recognize exercise gestures in real time using the Arduino Nano 33 BLE Sense.

## Project Overview

This project combines wearable technology, embedded systems, and machine learning to create an intelligent exercise-recognition glove.

The glove captures motion data from an onboard IMU sensor, processes the data using a TensorFlow Lite model, and classifies predefined exercise gestures.

The system was designed as part of the AIRE325 course project.

---

## Features

* Real-time gesture recognition
* TinyML deployment on Arduino Nano 33 BLE Sense
* Accelerometer and gyroscope data acquisition
* TensorFlow neural network training
* TensorFlow Lite inference
* State-machine controlled operation
* Visual feedback using onboard LEDs
* Low-power embedded implementation

---

## Hardware

* Arduino Nano 33 BLE Sense
* LSM9DS1 IMU
* Breadboard
* Push Button
* Buzzer
* USB Power

---

## Recognized Gestures

* Punch
* Down
* Left Hook
* Right Hook
* Uppercut

---

## Dataset Collection

Motion data is collected from the onboard IMU.

Each sample contains:

* aX
* aY
* aZ
* gX
* gY
* gZ

119 samples are recorded for every gesture instance.

---

## Machine Learning Pipeline

1. Collect IMU data
2. Export sensor readings to CSV
3. Normalize sensor values
4. One-hot encode gesture labels
5. Train TensorFlow model
6. Convert model to TensorFlow Lite
7. Deploy on Arduino

---

## Neural Network Architecture

Input Layer

Dense(50, ReLU)

Dense(15, ReLU)

Dense(15, ReLU)

Output Layer (Softmax)

---

## Training Configuration

* Optimizer: RMSProp
* Loss: Categorical Crossentropy
* Epochs: 1000
* Train Split: 60%
* Validation Split: 20%
* Test Split: 20%

---

## Results

Validation accuracy reached approximately 82%.

The system successfully performed gesture classification using IMU sensor data.

---

## Future Improvements

* More gesture classes
* Better hardware platform
* Audio feedback support
* Mobile application integration
* Higher accuracy neural networks

---

## Author

Mohammad Hasan Razi

Khaled Abdualmajed Al Faqeeh

Abu Dhabi University
