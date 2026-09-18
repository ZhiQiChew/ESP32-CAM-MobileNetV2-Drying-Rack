# ESP32-CAM-MobileNetV2-Drying-Rack
An AIoT-enabled automated drying rack system integrating computer vision, environmental sensing, and IoT control.

# AI Smart Drying Rack System

An AIoT-based smart drying rack system that integrates ESP32-CAM, MobileNetV2 image classification, environmental sensing, and automated decision-making for intelligent clothes drying management.

---

## Overview

This repository contains the local server-side implementation of an AI-based smart drying rack system.

The system uses ESP32-CAM to capture outdoor images, processes the images using computer vision techniques, performs weather condition classification using a trained MobileNetV2 model, and generates automated drying rack movement decisions.

The system combines AI-based image classification, environmental sensor information, and decision logic to automatically control the extension and retraction of the drying rack.

---

## Main Features

- ESP32-CAM image acquisition and upload
- MobileNetV2-based weather condition classification
- AI-based extend/retract decision making
- Image preprocessing and enhancement
  - Exposure correction
  - Colour correction
  - White balance adjustment
  - Overexposure detection
- FIFO-based prediction stabilisation
- Camera crop calibration system
- Local Flask server implementation
- Web-based monitoring dashboard
- Manual rack control interface
- Automatic image storage management
- Local network server discovery

---

# System Architecture

The system consists of three main modules:

## 1. AI Vision Module

The AI vision module uses ESP32-CAM to capture outdoor environmental images.

The captured images are:

1. Uploaded to the local server
2. Preprocessed and corrected
3. Passed into the MobileNetV2 model
4. Classified into drying rack conditions

The model generates a retract probability value which is used for decision making.

---

## 2. Environmental Monitoring Module

The environmental monitoring module collects information from external sensors.

The sensor information is used together with AI model prediction to determine the final rack movement decision.

The system considers:

- Rain detection
- Temperature
- Humidity
- Atmospheric condition

The rain sensor provides the highest priority safety decision for rack retraction.

---

## 3. Rack Control Module

The final decision is processed by the controller to operate the motor mechanism.

Possible rack actions:

- Extend
- Retract
- Hold current position

---
