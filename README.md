# 💧 Water Leak Detection – Rule-Based Method Comparison

[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/)
[![PostgreSQL](https://img.shields.io/badge/PostgreSQL-TimescaleDB-green.svg)](https://www.timescale.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A comparative study of **PyAerial** and **FP-Growth + CBA** for **rule-based classification** in water leak detection across multiple hydraulic scenarios.  
This project evaluates classification performance, rule interpretability, and robustness in **8 designed leak scenarios**.

---

## 📌 Project Overview

Water distribution network leaks can cause significant water loss and infrastructure damage.  
In this project, we:

1. Extract **flow**, **pressure**, and **demand** features from **TimescaleDB**.
2. Parse **EPANET INP** files to merge semantic metadata (pipe diameter, length, junction attributes).
3. Apply **two rule-based classification methods**:
   - **PyAerial** (Autoencoder-based rule extraction)
   - **FP-Growth + CBA** (Frequent Pattern Mining with Class-Based Association)
4. Compare their performance under controlled scenarios.
---

## 📂 Directory Structure

water_leak_detection_comparison/
```
│
├── discretization/ # Discretized datasets
├── rules_learned/ # Extracted rules (CSV format)
├── rules_stats/ # Rule statistics
├── classification_reports/ # Classification metrics (Precision, Recall, F1-score)
├── scripts/ # Python scripts for scenarios and pipelines
│ ├── scenario1_pyaerial.py
│ ├── scenario1_fpgrowth.py
│ └── ...
├── utils/ # Utility functions (database connection, INP parsing)
├── README.md # Project documentation
└── requirements.txt # Python dependencies
```

---

## ⚙️ Installation

1️⃣ **Clone the repository**
```bash
git clone https://github.com/yourusername/water_leak_detection_comparison.git
cd water_leak_detection_comparison

2️⃣ Create a virtual environment
python3 -m venv .venv
source .venv/bin/activate  # macOS/Linux
.venv\Scripts\activate     # Windows

3️⃣ Setup TimescaleDB connection
DB_USER = "xxx"
DB_PASS = "xxx"
DB_NAME = "xxx"
DB_HOST = "xxx"
DB_PORT = "xxx"
```

---

## 🚀 Running Experiments

Run PyAerial for a given scenario:
```
python scripts/scenario3_pyaerial.py
```
Results will be saved to:
```
classification_reports/
rules_learned/
rules_stats/
```

## 🛠 Technologies Used
Python 3.9+

TimescaleDB (PostgreSQL extension)

PyAerial – Autoencoder-based rule learning

FP-Growth (mlxtend) – Frequent Pattern Mining

CBA (pyarc) – Classification Based on Associations

