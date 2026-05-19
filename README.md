# AI-Controlled Hybrid Energy Storage System for Regenerative Braking

![Python](https://img.shields.io/badge/Python-3.13-blue?logo=python)
![MATLAB](https://img.shields.io/badge/MATLAB-R2025a-orange?logo=mathworks)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-green?logo=scikit-learn)
![License](https://img.shields.io/badge/License-MIT-yellow)

> ECS Mini Project | VIT Chennai | April 2026

---

## Overview

This project simulates an AI-controlled Hybrid Energy Storage System (HESS) for regenerative braking in electric vehicles. A **Random Forest ML classifier** decides in real time how to split recovered braking power between a **lithium-ion battery** and a **supercapacitor**, optimising energy recovery while protecting battery health.

In conventional vehicles, all kinetic energy is wasted as heat during braking. EVs can recover this energy — but smart storage management is critical. This project solves that using machine learning.

---

## Results

| Metric | Value |
|---|---|
| Total energy recovered | 1,304 Wh |
| Braking events handled | 4,363 |
| ML decision accuracy | ~95% (5-fold CV) |
| Battery cycles accumulated | 0.175 |
| Simulation duration | 20 minutes |
| Training samples | 27,615 |

---

## System Architecture
┌─────────────────────────────────────────────────────────────────┐
│                        SIMULINK MODEL                           │
│                                                                 │
│  ┌─────────────┐    ┌───────────────┐    ┌──────────────────┐  │
│  │ Drive Cycle │───▶│Motor-Generator│───▶│ DC-DC Converter  │  │
│  │  v(t), a(t) │    │  (PMDC model) │    │  (Bidirectional) │  │
│  └─────────────┘    └───────────────┘    └────────┬─────────┘  │
│                                                    │            │
│                                          ┌─────────┴────────┐  │
│                                          │   Power Split    │  │
│                                          │  split_ratio     │  │
│                                          └────┬──────┬──────┘  │
│                                               │      │         │
│                                    ┌──────────▼─┐  ┌─▼──────┐  │
│                                    │  Battery   │  │Supercap │  │
│                                    │ (Thevenin) │  │  (RC)   │  │
│                                    └──────┬─────┘  └────┬────┘  │
└───────────────────────────────────────────│─────────────│───────┘
                                            │   SoC data  │
                                            └──────┬───────┘
                                                   │
                              ┌────────────────────▼──────────────┐
                              │        PYTHON ML AGENT            │
                              │                                   │
                              │  Inputs: SoC_bat, SoC_sc,        │
                              │          jerk, power, velocity    │
                              │                                   │
                              │  Model:  Random Forest (200 trees)│
                              │                                   │
                              │  Output: split_ratio              │
                              │          0.10 / 0.50 / 0.80      │
                              └───────────────────────────────────┘

---

## Project Structure
regen-braking-hess/
│
├── 📄 storage_model.py        # Core physics models
│   ├── BatteryModel           #   → Thevenin equivalent circuit
│   ├── SupercapacitorModel    #   → RC equivalent circuit  
│   └── HybridStorageSystem    #   → Power split with overflow logic
│
├── 📄 drive_cycle.py          # Vehicle simulation
│   ├── VehicleDynamics        #   → Braking power computation
│   └── DriveCycleGenerator    #   → Urban / suburban / highway cycles
│
├── 📄 train_agent.py          # Machine learning pipeline
│   ├── generate_dataset()     #   → 27,615 labelled samples
│   ├── optimal_split_class()  #   → Physics-based oracle labeller
│   ├── train_model()          #   → Random Forest, cross-validation
│   └── predict_split()        #   → Real-time inference interface
│
├── 📄 simulate.py             # Main simulation runner
│   ├── run_simulation()       #   → Closed-loop ML agent run
│   ├── threshold_split()      #   → Baseline comparison
│   └── plot_results()         #   → 6-panel results dashboard
│
├── 📄 cosim_controller.py     # Python ↔ Simulink bridge
│   ├── run_cosimulation()     #   → MATLAB Engine API loop
│   └── run_python_fallback()  #   → Pure Python fallback
│
├── 📊 regen_model.slx         # MATLAB Simulink model
│   ├── Drive_Cycle            #   → Speed + acceleration signals
│   ├── Braking_Power          #   → Regen power computation
│   ├── Split_Ratio            #   → Power distribution logic
│   ├── Battery_SoC            #   → Integrator + saturation
│   └── Supercap_SoC           #   → Voltage → SoC conversion
│
├── 📄 requirements.txt        # Python dependencies
├── 📄 LICENSE                 # MIT License
└── 📄 README.md               # This file


---

## How to Run

### 1. Clone the repository
git clone https://github.com/tanisha-ahl/regen-braking-hess.git
cd regen-braking-hess

### 2. Install dependencies
pip install -r requirements.txt

### 3. Run the full simulation
python simulate.py

This will automatically:
- Generate a labelled training dataset from drive cycle simulations
- Train the Random Forest classifier and save it to model/
- Run a 20-minute closed-loop simulation (ML agent vs threshold baseline)
- Print a results summary in the terminal
- Save a 6-panel dashboard to results/simulation_dashboard.png

### 4. Train the model separately (optional)
python train_agent.py

### 5. Run co-simulation with Simulink (requires MATLAB R2022a+)
python cosim_controller.py

---

## ML Agent — Decision Logic

The Random Forest classifier makes real-time power split decisions at every braking event:

| Class | Decision | Split Ratio | Trigger Condition |
|---|---|---|---|
| 0 | Supercap-heavy | 10% battery / 90% SC | Jerk > 1.5 m/s3, OR battery SoC > 85% |
| 1 | Balanced | 50% / 50% | Normal SoC range, moderate braking |
| 2 | Battery-heavy | 80% battery / 20% SC | Supercap SoC > 75% OR supercap SoC < 25% |

### Feature Importances

| Feature | Importance | Description |
|---|---|---|
| jerk | 38% | Rate of change of acceleration |
| soc_supercap | 22% | Supercapacitor state of charge |
| bat_cycles | 18% | Accumulated battery charge cycles |
| velocity_kmh | 12% | Vehicle speed |
| soc_battery | 10% | Battery state of charge |

### Decision Distribution (20-min simulation)

- Supercap-heavy (Class 0): 2,613 events — 59.9%
- Balanced (Class 1): 838 events — 19.2%
- Battery-heavy (Class 2): 912 events — 20.9%

---

## Storage Models

### Battery — Thevenin Equivalent
- Capacity: 50 Ah at 48V nominal (2.4 kWh)
- Series resistance R0: 0.01 ohm
- Polarisation branch: R1 = 0.005 ohm, C1 = 2000 F
- SoC range: 20% to 95%
- SoC tracking: Coulomb counting

### Supercapacitor — RC Equivalent
- Capacitance: 3000 F
- Voltage range: 20V to 48V (usable ~100 Wh)
- ESR: 0.002 ohm
- SoC definition: (V - Vmin) / (Vmax - Vmin)

---

## Tech Stack

- Python 3.13 — simulation, ML training, visualisation
- scikit-learn — Random Forest classifier (200 trees, balanced class weights)
- MATLAB R2025a + Simulink — power electronics model
- MATLAB Engine API for Python — co-simulation bridge
- NumPy / Pandas — numerical computation and data handling
- Matplotlib — results dashboard and SoC plots

---

## Requirements

- Python 3.9 to 3.13
- pip install -r requirements.txt
- MATLAB R2022a or later with Simulink (only needed for cosim_controller.py)
- matlab.engine: pip install matlabengine==25.1

---

## References

1. Cao & Emadi (2012) — Battery/UltraCapacitor HESS for EVs. IEEE Trans. Power Electronics
2. Song et al. (2014) — Energy Management Strategies for HESS. Applied Energy, Elsevier
3. MathWorks — MATLAB Engine API for Python Documentation

---

## License

MIT License — free to use, modify, and distribute with attribution.
Copyright 2026 Tanisha Ahlawat
