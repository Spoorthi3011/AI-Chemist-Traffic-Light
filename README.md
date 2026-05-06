# 🧪 AI-Chemist Traffic Light

An autonomous robotic chemistry system that monitors and controls a three-phase oscillating colour reaction (GREEN → RED → YELLOW → GREEN) using a UR robot arm, YOLO-based vial detection, real-time colour analysis, and a live web dashboard.

> **Full cycle confirmed** — experiment run 2026-05-06, all three phases completed successfully.

---

## 🎬 What It Does

The system automates the classic **traffic light reaction** — a redox oscillator that cycles through three distinct colours. A UR robot arm picks up a vial, places it on an IKA stirrer, and the AI monitors the colour transition in real time using a camera + YOLO model. Each phase is confirmed by stable colour detection before moving to the next.

```
GREEN  ──►  RED  ──►  YELLOW  ──►  GREEN
(start)  (oxidation)  (reduction)  (regeneration)
```

---

## 📊 Example Results

### Reaction Profile
![AI-Chemist Reaction Profile](results/ai_chemist_report.png)

| Phase | Transition | Success | Yield | Duration | Dominant |
|-------|-----------|---------|-------|----------|----------|
| 1 | GREEN → RED | ✅ YES | 66.8% | 2.8s | RED |
| 2 | RED → YELLOW | ✅ YES | 46.4% | 0.2s | YELLOW |
| 3 | YELLOW → GREEN | ✅ YES | 47.4% | 0.0s | GREEN |

**Full Cycle Complete: ✅ YES** — timestamp `2026-05-06 11:33:20`

See [`results/SUMMARY.txt`](results/SUMMARY.txt) for full metrics.

---

## 🗂️ Repository Structure

```
ai-chemist-traffic-light/
│
├── main_novel.py                  # Main entry point — run this
│
├── handlers/                      # All hardware & AI modules
│   ├── __init__.py
│   ├── robot_handler.py           # UR robot control (URControl)
│   ├── camera_handler_2.py        # Camera capture & recording
│   ├── colour_detection.py        # HSV-based colour detector
│   ├── ika_stirrer.py             # IKA stirrer serial control
│   ├── ai_chemist.py              # RPM recommendations & AI logic
│   ├── digital_twin.py            # Data logging & twin state
│   ├── reaction_analyzer.py       # Phase metrics & final report
│   └── vlm_monitor.py             # VLM overlay & status
│
├── examples/
│   └── utils/
│       └── robotiq/
│           └── robotiq_gripper.py # Robotiq gripper driver
│
├── models/
│   └── best.pt                    # YOLO vial detection weights
│                                  # (download separately — see below)
│
├── results/                       # Example experiment output (committed)
│   ├── ai_chemist_report.png      # Reaction profile graph
│   └── SUMMARY.txt                # Phase metrics from example run
│
├── docs/
│   ├── experiment_overview.md     # Chemistry background
│   ├── colour_sequence.md         # Detection logic explained
│   └── hardware_setup.md          # Wiring, ports, network config
│
├── OUTPUT/                        # Runtime output — gitignored
│   └── {timestamp}_AI_CHEMIST/
│       ├── images/                # Phase stills + confirmations
│       ├── videos/                # Annotated .avi recordings
│       └── reports/               # SUMMARY.txt + plots
│
├── requirements.txt
├── .gitignore
└── README.md
```

---

## 🔧 Hardware Requirements

| Component | Details |
|-----------|---------|
| **Robot arm** | Universal Robots UR (any model), IP `192.168.0.1` |
| **Gripper** | Robotiq 2F gripper, port `63352` |
| **Camera** | USB camera, device index `2`, rotated 180° |
| **Stirrer** | IKA stirrer, serial port `/dev/ttyACM0` |
| **Host machine** | IP `192.168.0.2`, Ubuntu 24, mambaforge Python 3.10 |

Network: robot and host must be on the same subnet (`192.168.0.x`).

---

## 🚀 Quick Start

### 1. Clone the repo

```bash
git clone https://github.com/YOUR_USERNAME/ai-chemist-traffic-light.git
cd ai-chemist-traffic-light
```

### 2. Set up environment

```bash
# Using mambaforge (recommended — matches deployment environment)
conda create -n ai-chemist python=3.10
conda activate ai-chemist
pip install -r requirements.txt
```

### 3. Download YOLO weights

Place `best.pt` in the `models/` folder. The model is trained to detect the reaction vial within the camera frame. Contact the team or see the release assets for the weights file.

### 4. Check hardware connections

```bash
# Verify camera index
python -c "import cv2; cap = cv2.VideoCapture(2); print(cap.isOpened())"

# Verify stirrer port
ls /dev/ttyACM*

# Free camera if already in use
fuser -k /dev/video2
```

### 5. Run the experiment

```bash
python main_novel.py
```

### 6. Open the live dashboard

```
http://192.168.0.2:8765
```

The dashboard shows the live MJPEG camera feed with YOLO + colour annotations, phase progress, pixel intensity bars, stirrer RPM, and a live log — no display or GUI needed on the robot PC.

---

## 🧠 System Architecture

```
main_novel.py
    │
    ├── StreamServer          → MJPEG web dashboard (port 8765)
    ├── enhanced_detection()  → YOLO + colour detection loop (thread)
    │       ├── YOLO (best.pt)       vial bounding box
    │       ├── ColourDetector       HSV pixel classification
    │       └── VLMMonitor           overlay annotations
    │
    ├── phase1_oxidation()    → pick vial, insert, wait for RED
    ├── phase2_reduction()    → reposition, wait for YELLOW  
    ├── phase3_regeneration() → stir 25s, wait for GREEN
    │
    ├── DigitalTwin           → data logging throughout
    ├── ReactionAnalyzer      → per-phase metrics
    └── VideoRecorder         → annotated .avi output
```

### Colour detection logic

Detection runs every 0.5s. A colour is **confirmed** only when:
- Detected in **≥2 consecutive frames**
- Held stable for **≥1.5 seconds**
- With tolerance for up to **5 flicker frames** before resetting

This prevents noise from vial reflections or stirrer motion triggering false phase transitions.

---

## 📦 Requirements

```
ultralytics>=8.0
opencv-python>=4.8
numpy>=1.24
pyserial>=3.5
```

Install with:

```bash
pip install -r requirements.txt
```

> **Note:** The deployment environment uses mambaforge with PyTorch from conda and system OpenCV. If you encounter GPU/CUDA issues, install torch separately via conda before running pip.

---

## 📁 Output Files

Every experiment run creates a timestamped folder under `OUTPUT/`:

```
OUTPUT/20260506_112717_AI_CHEMIST/
├── images/
│   ├── phase1_start.jpg
│   ├── phase1_pick.jpg
│   ├── CONFIRMED_RED_20260506_113320.jpg   ← confirmation snapshots
│   ├── CONFIRMED_YELLOW_...jpg
│   ├── CONFIRMED_GREEN_...jpg
│   └── LIVE_...jpg                          ← periodic live frames
├── videos/
│   └── FULL_EXPERIMENT_20260506_112717.avi
└── reports/
    ├── ai_chemist_report.png               ← reaction profile graph
    └── SUMMARY.txt                         ← phase metrics
```

---

## 🌐 Live Dashboard

Open `http://192.168.0.2:8765` in any browser on the local network during an experiment run.

Features:
- Live MJPEG stream with YOLO bounding boxes and colour labels
- Phase progress tracker (waiting / running / done)
- Colour swatch + RED / GREEN / YELLOW pixel intensity bars
- Stirrer RPM and YOLO confidence readout
- Scrolling log of key events

---

## ⚙️ Configuration

Key constants in `main_novel.py`:

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST` | `192.168.0.2` | Robot/host IP |
| `PORT` | `30003` | UR robot port |
| `StreamServer port` | `8765` | Dashboard port |
| `stirrer port` | `/dev/ttyACM0` | IKA serial port |
| Camera index | `2` | `cv2.VideoCapture(2)` |
| Phase 1 timeout | `180s` | Max wait for RED |
| Phase 2 timeout | `120s` | Max wait for YELLOW |
| Phase 3 timeout | `200s` | Max wait for GREEN |
| Stirrer RPM | `1500` | Set in phases 1 & 3 |

Robot joint positions (`YOUR_POSITIONS`) are hardcoded for the specific lab setup — update these if the robot or stirrer position changes.

---

## 🛑 Stopping the Experiment

Press `Ctrl+C` at any time. The `finally` block will:
1. Stop the stirrer
2. Stop all threads cleanly
3. Stop camera recording
4. Save annotated video
5. Shut down the web server

All files captured so far are saved to the `OUTPUT/` folder.

---

## 📜 License

MIT License — see [`LICENSE`](LICENSE) for details.

---

## 👥 Authors

Group C — Robotics & AI Chemistry Lab  
University project, 2025–2026
