
# 📂 Dataset

## Overview

The `data` folder contains the datasets used for training, validation, and testing the proposed **DWU-ODBN (Deep Weighted Unsupervised Optimized Deep Belief Network)** model for anomaly detection in **IoT-enabled multimedia communication systems**.

The datasets represent both normal and anomalous network traffic generated from IoT devices and multimedia communication environments.

---

## Dataset Contents

```
data/
│
├── train/
├── test/
├── validation/
└── README.md
```

> The folder structure may vary depending on the preprocessing pipeline and experimental setup.

---

## Dataset Description

The datasets contain network traffic records collected from IoT communication environments. Each record consists of multiple network features extracted from packet-level and flow-level information.

Typical attributes include:

- Source IP Address
- Destination IP Address
- Source Port
- Destination Port
- Protocol
- Packet Length
- Flow Duration
- Packet Count
- Byte Count
- Time Stamp
- Network Flags
- Traffic Labels (Normal / Anomaly)

---

## Data Preprocessing

Before model training, the following preprocessing steps are performed:

- Data Cleaning
- Missing Value Handling
- Duplicate Removal
- Feature Selection
- Label Encoding
- Feature Scaling
- Data Normalization

---

## Supported Platforms

The datasets are compatible with:

- Internet of Things (IoT) Networks
- Multimedia Communication Systems
- Windows Operating System
- Linux Operating System

---

## Purpose

The datasets are used to:

- Train the DWU-ODBN model
- Detect anomalous network activities
- Evaluate model performance
- Improve cybersecurity in IoT environments

---

## Notes

Large datasets are not included in this repository due to GitHub file size limitations. Users can place the required datasets inside this directory before running the project.

---

## Keywords

IoT • Network Security • Cybersecurity • Anomaly Detection • Deep Learning • DWU-ODBN • Multimedia Communication • Machine Learning
