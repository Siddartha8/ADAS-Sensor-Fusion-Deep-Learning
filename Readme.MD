# 🚗 Sensor Fusion and Deep Learning for Advanced Driver Assistance System (ADAS) Perception

![Python](https://img.shields.io/badge/Python-3.12-blue?logo=python)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.18.0-orange?logo=tensorflow)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-1.6.1-yellow?logo=scikit-learn)
![Redis](https://img.shields.io/badge/Redis-6.4.0-red?logo=redis)
![License](https://img.shields.io/badge/license-MIT-green)
[![GitHub](https://img.shields.io/badge/GitHub-Siddartha8-black?logo=github)](https://github.com/Siddartha8/ADAS-Sensor-Fusion-Deep-Learning)

> A **Major Project** implementing a hybrid **DualStream-ConvRF** model that fuses sensor data from electric vehicles (EV) with deep learning (CNN-1D) and classical machine learning (Random Forest) to classify ADAS driving actions in real time.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [Dataset Description](#-dataset-description)
- [Features & Workflow](#-features--workflow)
- [Prerequisites](#-prerequisites)
- [Installation Guide (Step-by-Step)](#-installation-guide-step-by-step)
  - [Step 1 – Install Python](#step-1--install-python)
  - [Step 2 – Install & Start Redis](#step-2--install--start-redis)
  - [Step 3 – Clone the Repository](#step-3--clone-the-repository)
  - [Step 4 – Create a Virtual Environment](#step-4--create-a-virtual-environment)
  - [Step 5 – Install Dependencies](#step-5--install-dependencies)
  - [Step 6 – Add Required Files](#step-6--add-required-files)
  - [Step 7 – Run the Application](#step-7--run-the-application)
- [Using the Application](#-using-the-application)
  - [Admin Workflow](#admin-workflow)
  - [User Workflow](#user-workflow)
- [Models Implemented](#-models-implemented)
- [Output & Results](#-output--results)
- [Troubleshooting](#-troubleshooting)
- [Contributing](#-contributing)
- [License](#-license)

---

## 📖 Project Overview

This project addresses the challenge of intelligent ADAS decision-making by fusing multiple sensor streams — speed, acceleration, braking, lane deviation, obstacle distance, weather, road type, and more — from an Electric Vehicle (EV) dataset. A novel **DualStream-ConvRF** hybrid architecture is proposed and benchmarked against classical ML models (SVM, KNN, Random Forest).

**ADAS Output Classes:**
| Label | Description |
|---|---|
| `Brake` | Apply brakes based on sensor inputs |
| `Lane Correct` | Perform lane correction manoeuvre |
| `Maintain Speed` | Cruise at current speed |
| `Accelerate` | Increase vehicle speed |

---

## 🏗 System Architecture

```
Raw EV Sensor CSV
       │
       ▼
┌─────────────────────────────┐
│  Preprocessing & EDA        │  ← Label Encoding, Timestamp parsing,
│                             │     Missing value imputation
└─────────────────────────────┘
       │
       ▼
┌─────────────────────────────┐
│  SMOTE Balancing & Splitting │  ← 80% Train / 20% Test
└─────────────────────────────┘
       │
       ├──────────────────┬─────────────────┬──────────────────────────────────┐
       ▼                  ▼                 ▼                                  ▼
   SVM Model         KNN Model         RFC Model          DualStream-ConvRF (Proposed)
                                                          ┌───────────────────────────┐
                                                          │ CNN-1D Feature Extractor  │
                                                          │   + Raw Sensor Features   │
                                                          │   → Random Forest (300t)  │
                                                          └───────────────────────────┘
       │
       ▼
Accuracy / Precision / Recall / F1 / Confusion Matrix / ROC Curve
```

---

## 🛠 Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.10+ |
| **GUI** | Tkinter |
| **Deep Learning** | TensorFlow 2.18 / Keras |
| **Machine Learning** | Scikit-Learn 1.6 |
| **Data Balancing** | imbalanced-learn (SMOTE) |
| **Data Processing** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn |
| **Model Persistence** | Joblib |
| **Authentication** | Redis (User/Admin login with SHA-256 hashed passwords) |
| **Image Handling** | Pillow |

---

## 📁 Project Structure

```
ADAS-Project/
│
├── Main.py                        # Main application entry point (GUI + ML pipeline)
├── requirements.txt               # Python dependencies
├── run.bat                        # Windows one-click launcher
├── background.jpg                 # Background image for the GUI (you must provide this)
│
├── ADAS_EV_Dataset.csv            # Training dataset (10,000+ EV sensor records)
├── TestData1.csv                  # Test dataset for prediction
│
├── models/                        # Auto-created — saved models stored here
│   ├── svm_classifier.joblib
│   ├── knn_classifier.joblib
│   ├── random_forest_classifier.joblib
│   ├── hybrid_cnn1d_model.h5
│   ├── hybrid_cnn1d_random_forest.joblib
│   └── scaler.joblib
│
└── results/                       # Auto-created — plots and CSVs saved here
    ├── *_confusion_matrix.png
    ├── *_roc_curve.png
    └── TestData1_HybridPredictions_CNN1D_clean.csv
```

---

## 📊 Dataset Description

**File:** `ADAS_EV_Dataset.csv`  
**Records:** ~10,000 rows

| Column | Type | Description |
|---|---|---|
| `timestamp` | datetime | Date and time of sensor reading |
| `speed_kmh` | float | Vehicle speed in km/h |
| `acceleration_mps2` | float | Acceleration in m/s² |
| `brake_intensity` | float | Brake pedal intensity (0–1) |
| `battery_level` | float | EV battery level (%) |
| `energy_consumption` | float | Power draw (kWh) |
| `regen_braking_usage` | float | Regenerative braking usage (%) |
| `lane_deviation` | float | Lane deviation in meters |
| `obstacle_distance` | float | Distance to nearest obstacle (m) |
| `traffic_density` | int | Traffic density score |
| `weather_condition` | string | Sunny / Rainy / Foggy |
| `road_type` | string | Highway / Urban / Rural |
| `steering_angle` | float | Steering angle in degrees |
| `reaction_time` | float | Driver reaction time (s) |
| `ADAS_output` | string | **Target class** (Brake / Lane Correct / Maintain Speed / Accelerate) |

---

## ✨ Features & Workflow

- 🔐 **Role-based Authentication** via Redis (Admin & User with hashed passwords)
- 📂 **Dataset Upload** via GUI file dialog
- 🔍 **Preprocessing & EDA** — encoding, imputation, class distribution plot
- ⚖️ **SMOTE Balancing** — handles class imbalance before training
- 🤖 **Multiple ML Models** — SVM, KNN, Random Forest
- 🧠 **Proposed DualStream-ConvRF** — CNN-1D feature extraction fused with Random Forest
- 📈 **Metrics Dashboard** — Accuracy, Precision, Recall, F1-Score per model
- 🗺 **Confusion Matrix & ROC Curves** — saved to `results/` folder
- 🔮 **Prediction on New Data** — upload test CSV and get ADAS predictions

---

## ✅ Prerequisites

Before you begin, ensure the following are installed on your system:

- Python **3.12** (Microsoft Store or official installer)
- `pip` package manager
- Redis server (for authentication)
- Git

---

## 🚀 Installation Guide (Step-by-Step)

### Step 1 – Install Python

1. Go to [https://www.python.org/downloads/](https://www.python.org/downloads/)
2. Download **Python 3.12.x** (or install via Microsoft Store — search "Python 3.12")
3. During installation, **check ✅ "Add Python to PATH"**
4. Verify installation:
   ```bash
   python --version
   # Expected: Python 3.12.x
   ```

---

### Step 2 – Install & Start Redis

Redis is used for storing user credentials securely.

**Option A – Windows (Recommended: using Memurai or WSL)**

**Using Memurai (native Windows Redis):**
1. Download from [https://www.memurai.com/get-memurai](https://www.memurai.com/get-memurai)
2. Install and it starts automatically as a Windows service on port `6379`

**Using Windows Subsystem for Linux (WSL):**
```bash
# In WSL terminal:
sudo apt update
sudo apt install redis-server -y
sudo service redis-server start
```

**Option B – macOS**
```bash
brew install redis
brew services start redis
```

**Option C – Linux (Ubuntu/Debian)**
```bash
sudo apt update
sudo apt install redis-server -y
sudo systemctl enable redis-server
sudo systemctl start redis-server
```

**Verify Redis is running:**
```bash
redis-cli ping
# Expected output: PONG
```

> ⚠️ **Important:** Redis must be running on `localhost:6379` before launching the application. This is the most common cause of the startup error shown in the screenshot.

---

### Step 3 – Clone the Repository

```bash
git clone https://github.com/Siddartha8/ADAS-Sensor-Fusion-Deep-Learning.git
cd ADAS-Sensor-Fusion-Deep-Learning
```

Or download the ZIP and extract it.

---

### Step 4 – Create a Virtual Environment

Creating a virtual environment keeps your project dependencies isolated.

```bash
# Create virtual environment
python -m venv venv

# Activate it — Windows
venv\Scripts\activate

# Activate it — macOS/Linux
source venv/bin/activate
```

You should see `(venv)` at the start of your terminal prompt.

---

### Step 5 – Install Dependencies

```bash
pip install -r requirements.txt
```

This installs all required packages:

```
numpy==1.26.4
pandas==2.3.1
matplotlib==3.10.3
seaborn==0.13.2
scikit-learn==1.6.1
imbalanced-learn==0.14.0
tensorflow==2.18.0
joblib==1.4.2
Pillow==11.3.0
redis==6.4.0
```

> 💡 TensorFlow installation may take a few minutes. If you encounter issues on Windows, ensure you have [Microsoft Visual C++ Redistributable](https://aka.ms/vs/17/release/vc_redist.x64.exe) installed.

---

### Step 6 – Add Required Files

Ensure the following files are in the **root project directory** (same folder as `Main.py`):

| File | Notes |
|---|---|
| `ADAS_EV_Dataset.csv` | Main training dataset |
| `TestData1.csv` | Test/prediction dataset |
| `background.jpg` | Any JPEG image for the GUI background (rename your image to `background.jpg`) |

> The `models/` and `results/` folders will be **created automatically** the first time you run the application.

---

### Step 7 – Run the Application

**Windows (double-click):**
```
run.bat
```

**Windows (terminal):**
```bash
python Main.py
```

**macOS / Linux:**
```bash
python3 Main.py
```

The GUI window will open full screen.

---

## 🖥 Using the Application

### Admin Workflow

Admins can train models and run the full ML pipeline.

1. **Admin Signup** → Click `Admin Signup` → Enter username and password → Click `Signup`
2. **Admin Login** → Click `Admin Login` → Enter credentials → Click `Login`
3. After login, the following buttons appear:

| Button | Action |
|---|---|
| `Upload Dataset (EV)` | Open file dialog → Select `ADAS_EV_Dataset.csv` |
| `Preprocessing & EDA` | Encodes features, imputes nulls, shows class distribution chart |
| `Data Balancing & Splitting` | Applies SMOTE, splits 80/20 into train/test |
| `Build & Train ML Models` | Trains SVM, KNN, Random Forest — shows metrics & plots |
| `Build & Train Proposed DualStream-ConvRF Model` | Trains the CNN-1D + RF hybrid model |

> 💡 **Run buttons in order (top-left to right)** for first-time use. Trained models are saved in `models/` and reloaded on subsequent runs — no need to retrain.

---

### User Workflow

Users can make predictions using the trained models.

1. **User Signup** → Click `User Signup` → Register credentials
2. **User Login** → Click `User Login` → Log in
3. Click `Prediction on Test Data` → Select `TestData1.csv` via file dialog
4. Predictions appear in the output window and are saved to:
   ```
   results/TestData1_HybridPredictions_CNN1D_clean.csv
   ```

> ⚠️ The Prediction feature requires Admin to have already **trained all models** (Step 4 & 5 of Admin Workflow) at least once.

---

## 🤖 Models Implemented

| Model | Description | Key Parameters |
|---|---|---|
| **SVM** | Support Vector Machine with RBF kernel | `probability=True`, `kernel='rbf'` |
| **KNN** | K-Nearest Neighbours | `n_neighbors=5` |
| **RFC** | Random Forest Classifier (baseline) | `n_estimators=10`, `max_depth=2` |
| **DualStream-ConvRF** *(Proposed)* | CNN-1D feature extractor + Random Forest (300 trees) | `epochs=200`, `early_stopping`, `n_estimators=300`, `max_depth=30` |

The proposed model uses **feature fusion**: raw normalized features are concatenated with the 128-dimensional deep feature embeddings from the CNN penultimate layer before feeding into the Random Forest.

---

## 📈 Output & Results

All results are automatically saved to the `results/` directory:

- `SVM_Model_confusion_matrix.png`
- `SVM_Model_roc_curve.png`
- `KNN_Classifier_confusion_matrix.png`
- `KNN_Classifier_roc_curve.png`
- `RFC_Model_confusion_matrix.png`
- `RFC_Model_roc_curve.png`
- `DualStream-ConvRF_Model_confusion_matrix.png`
- `DualStream-ConvRF_Model_roc_curve.png`
- `TestData1_HybridPredictions_CNN1D_clean.csv`

Metrics displayed in the GUI output panel include:
- Accuracy (%)
- Precision (macro, %)
- Recall (macro, %)
- F1-Score (macro, %)
- Full classification report per class

---

## 🔧 Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `Redis Error: Error 10061 connecting to localhost:6379` | Redis server is not running | Start Redis (see Step 2) |
| `ModuleNotFoundError: No module named 'tensorflow'` | Dependencies not installed | Run `pip install -r requirements.txt` in the activated virtual environment |
| `FileNotFoundError: background.jpg` | Missing background image | Add any `.jpg` image renamed as `background.jpg` to the project root |
| `ValueError: Missing encoder for column` | Predicting before training | Complete Admin Workflow Steps 1–5 first |
| `OSError: No such file or directory: 'models/scaler.joblib'` | Scaler not saved yet | Train the DualStream-ConvRF model first (Admin → Step 5) |
| GUI opens but buttons don't respond | Admin not logged in | Log in as Admin to see training buttons |
| TensorFlow not installing | Version incompatibility | Ensure you are using Python 3.12 and `tensorflow==2.18.0` |

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a new branch: `git checkout -b feature/your-feature-name`
3. Make your changes and commit: `git commit -m "Add: your feature"`
4. Push to your branch: `git push origin feature/your-feature-name`
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👨‍💻 Author

**Kandadi Siddartha**  
Major Project — B.Tech  
Department of Computer Science & Engineering  
Sree Dattha Institution of Engineering and Science  

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Kandadi%20Siddartha-blue?logo=linkedin)](https://www.linkedin.com/in/siddarthakandadi)

---

> ⭐ If you found this project helpful, please give it a star on GitHub!