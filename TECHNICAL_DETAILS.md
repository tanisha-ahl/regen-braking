# AI-Controlled Hybrid Energy Storage for Regenerative Braking
**ECS Mini Project | Python Simulation Module**

---

## Project Structure

```
regen_braking/
├── storage_model.py     # Battery (Thevenin) + Supercapacitor (RC) models
├── drive_cycle.py       # Vehicle dynamics + synthetic drive cycle generator
├── train_agent.py       # Dataset generation + Random Forest ML training
├── simulate.py          # Full closed-loop simulation + results dashboard
├── requirements.txt     # Python dependencies
├── data/                # Generated training CSV (auto-created)
├── model/               # Saved RF model pickle (auto-created)
└── results/             # Output plots and CSVs (auto-created)
```

---

## How to Run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run the full simulation
```bash
python simulate.py
```

On first run this will:
- Generate a labelled training dataset from 6 drive cycle simulations
- Train and save the Random Forest classifier
- Run the test simulation (ML agent vs threshold baseline)
- Print a results summary table
- Display and save the 6-panel results dashboard

### 3. Train the model separately (optional)
```bash
python train_agent.py
```

---

## Module Overview

### `storage_model.py`
| Class | Description |
|---|---|
| `BatteryModel` | 1st-order Thevenin equivalent. Tracks SoC via Coulomb counting, V_terminal, cycle count |
| `SupercapacitorModel` | RC equivalent. Energy-based SoC between V_min and V_max |
| `HybridStorageSystem` | Combines both; applies split ratio to distribute braking power |

### `drive_cycle.py`
| Class | Description |
|---|---|
| `VehicleDynamics` | Computes recoverable braking power from velocity + deceleration |
| `DriveCycleGenerator` | Generates synthetic urban / suburban / highway drive cycles |

### `train_agent.py`
| Function | Description |
|---|---|
| `generate_dataset()` | Runs simulations, labels data using physics-based oracle |
| `train_model()` | Trains Random Forest + cross-validation + feature importance |
| `predict_split()` | Inference interface: returns split class, ratio, probabilities |

### `simulate.py`
- Runs full closed-loop simulation
- Compares ML agent vs threshold-based baseline
- Outputs: energy recovered (Wh), battery cycles, SoC curves, power split plots

---

## Split Classes
| Class | Ratio | When Used |
|---|---|---|
| 0 — Supercap-heavy | 10% battery / 90% SC | High jerk, battery near full |
| 1 — Balanced | 50% / 50% | Moderate braking, normal SoC |
| 2 — Battery-heavy | 80% battery / 20% SC | Supercap near full, battery has headroom |

---

## Next Steps
- Connect to MATLAB/Simulink via `matlab.engine` (co-simulation bridge)
- Replace synthetic drive cycle with real UDDS/WLTP data
- Upgrade ML agent to Deep Q-Network (DQN) for full RL control
