"""
cosim_controller.py
-------------------
Co-simulation bridge between Python ML agent and Simulink.

How it works:
  1. Starts MATLAB engine from Python
  2. Loads regen_model.slx
  3. Runs simulation in small time steps
  4. At each braking event:
       - Reads SoC_battery, SoC_supercap from Simulink
       - Feeds them into the ML agent (Random Forest)
       - Gets back optimal split_ratio
       - Writes split_ratio back into Simulink
  5. Collects results and plots comparison dashboard

Usage:
    python cosim_controller.py

Requirements:
    - MATLAB R2025a with Simulink installed
    - matlab.engine Python package installed
    - regen_model.slx in same folder as this script
    - model/rf_agent.pkl trained ML model present
"""

import os
import sys
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings("ignore")

# ── Check matlab.engine is available ─────────────────────────
try:
    import matlab.engine
except ImportError:
    print("[ERROR] matlab.engine not found.")
    print("        Run: python -m pip install matlabengine==25.1")
    sys.exit(1)


# ─────────────────────────────────────────────
#  CONFIGURATION
# ─────────────────────────────────────────────
MODEL_PATH   = os.path.join(os.path.dirname(__file__), "regen_model.slx")
AGENT_PATH   = os.path.join(os.path.dirname(__file__), "model", "rf_agent.pkl")
RESULTS_DIR  = os.path.join(os.path.dirname(__file__), "results")
SIM_DURATION = 1200.0    # seconds
DT           = 0.1       # time step — must match Simulink fixed step size

SPLIT_RATIO_MAP = {0: 0.10, 1: 0.50, 2: 0.80}

os.makedirs(RESULTS_DIR, exist_ok=True)


# ─────────────────────────────────────────────
#  LOAD ML MODEL
# ─────────────────────────────────────────────
def load_agent(path: str) -> dict:
    if not os.path.exists(path):
        print(f"[ERROR] ML model not found at: {path}")
        print("        Run python train_agent.py first.")
        sys.exit(1)
    with open(path, "rb") as f:
        return pickle.load(f)


def predict_split(model_dict, soc_bat, soc_sc, power_kw, jerk, bat_cycles, velocity_kmh):
    features = model_dict["features"]
    soc_ratio = soc_bat / max(soc_sc, 1e-6)
    X = np.array([[soc_bat, soc_sc, power_kw, jerk, bat_cycles, velocity_kmh, soc_ratio]])
    cls = int(model_dict["pipeline"].predict(X)[0])
    return SPLIT_RATIO_MAP[cls], cls


# ─────────────────────────────────────────────
#  SIMULINK INTERFACE HELPERS
# ─────────────────────────────────────────────
def read_signal(eng, model: str, signal_path: str) -> float:
    """Read the current value of a logged signal from Simulink workspace."""
    try:
        val = eng.eval(f"get_param('{signal_path}', 'RuntimeObject')", nargout=1)
        return float(val)
    except Exception:
        return 0.0


def set_constant_block(eng, block_path: str, value: float):
    """Update a Constant block's value in Simulink during simulation."""
    eng.set_param(block_path, "Value", str(value), nargout=0)


def get_workspace_var(eng, var: str) -> float:
    """Read a scalar variable from MATLAB base workspace."""
    try:
        return float(eng.eval(f"double({var})", nargout=1))
    except Exception:
        return 0.0


# ─────────────────────────────────────────────
#  MAIN CO-SIMULATION LOOP
# ─────────────────────────────────────────────
def run_cosimulation(model_dict: dict) -> pd.DataFrame:
    """
    Run the full co-simulation loop.
    Returns a DataFrame of time-series results.
    """

    print("\n[1/3] Starting MATLAB engine ...")
    eng = matlab.engine.start_matlab()
    print("      MATLAB engine started.")

    print("[2/3] Loading Simulink model ...")
    eng.eval(f"load_system('{MODEL_PATH}')", nargout=0)

    # Configure model for stepped simulation
    eng.set_param("regen_model", "SimulationMode",    "normal",      nargout=0)
    eng.set_param("regen_model", "StartTime",         "0",           nargout=0)
    eng.set_param("regen_model", "StopTime",          str(SIM_DURATION), nargout=0)
    eng.set_param("regen_model", "SolverType",        "Fixed-step",  nargout=0)
    eng.set_param("regen_model", "FixedStep",         str(DT),       nargout=0)

    # Set up To Workspace blocks for SoC logging
    # These must exist in your model — add them if not already there
    SPLIT_BLOCK = "regen_model/Split_Ratio/split_ratio"

    print("[3/3] Running co-simulation loop ...")
    print(f"      Duration: {SIM_DURATION}s  |  Step: {DT}s  |  Steps: {int(SIM_DURATION/DT)}")

    records      = []
    prev_accel   = 0.0
    bat_cycles   = 0.0
    split_ratio  = 0.5
    total_steps  = int(SIM_DURATION / DT)
    print_every  = total_steps // 10

    # ── Initialise simulation ──────────────────
    eng.eval("set_param('regen_model', 'SimulationCommand', 'start')", nargout=0)
    eng.eval("set_param('regen_model', 'SimulationCommand', 'pause')", nargout=0)

    for step in range(total_steps):
        # Progress indicator
        if step % print_every == 0:
            pct = int(step / total_steps * 100)
            print(f"      {pct}% ", end="", flush=True)

        # ── Read current state from Simulink workspace ──
        try:
            soc_bat  = float(eng.eval("ans_soc_bat",  nargout=1)) / 100.0
        except Exception:
            soc_bat  = 0.60

        try:
            soc_sc   = float(eng.eval("ans_soc_sc",   nargout=1)) / 100.0
        except Exception:
            soc_sc   = 0.30

        try:
            p_regen  = float(eng.eval("ans_p_regen",  nargout=1))
        except Exception:
            p_regen  = 0.0

        try:
            velocity = float(eng.eval("ans_velocity", nargout=1))
        except Exception:
            velocity = 0.0

        try:
            accel    = float(eng.eval("ans_accel",    nargout=1))
        except Exception:
            accel    = 0.0

        # ── ML agent decides split ratio ────────────────
        jerk = (accel - prev_accel) / DT

        if p_regen > 0:
            split_ratio, cls = predict_split(
                model_dict,
                soc_bat      = soc_bat,
                soc_sc       = soc_sc,
                power_kw     = p_regen / 1000,
                jerk         = jerk,
                bat_cycles   = bat_cycles,
                velocity_kmh = velocity * 3.6,
            )
            # Overflow overrides
            if soc_sc  >= 0.95: split_ratio = 1.0
            if soc_bat >= 0.95: split_ratio = 0.0
        else:
            split_ratio = 0.5
            cls = 1

        # ── Write split_ratio back to Simulink ─────────
        try:
            eng.eval(f"set_param('regen_model/Split_Ratio', 'Value', '{split_ratio}')", nargout=0)
        except Exception:
            pass

        # ── Record state ────────────────────────────────
        bat_cycles += abs(soc_bat - (records[-1]["soc_battery"] if records else 0.60)) / 2.0

        records.append({
            "time":              step * DT,
            "velocity_kmh":      velocity * 3.6,
            "accel_ms2":         accel,
            "braking_power_kw":  p_regen / 1000,
            "split_ratio":       split_ratio,
            "split_class":       cls,
            "soc_battery":       soc_bat * 100,
            "soc_supercap":      soc_sc  * 100,
            "p_battery_kw":      p_regen * split_ratio / 1000,
            "p_supercap_kw":     p_regen * (1 - split_ratio) / 1000,
            "bat_cycles":        bat_cycles,
        })

        prev_accel = accel

        # ── Step simulation forward by DT ───────────────
        eng.eval("set_param('regen_model', 'SimulationCommand', 'step')", nargout=0)

    print(" 100% done!")

    # Stop simulation
    eng.eval("set_param('regen_model', 'SimulationCommand', 'stop')", nargout=0)
    eng.quit()
    print("      MATLAB engine stopped.")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  FALLBACK: Pure Python simulation
#  (use if MATLAB Engine loop has timing issues)
# ─────────────────────────────────────────────
def run_python_fallback(model_dict: dict) -> pd.DataFrame:
    """
    Runs the simulation entirely in Python using the same
    storage models and drive cycle as the main simulation.
    Use this if the MATLAB Engine step-by-step loop is too slow.
    """
    print("\n  Using Python fallback simulation ...")
    from storage_model import BatteryModel, SupercapacitorModel, HybridStorageSystem
    from drive_cycle  import DriveCycleGenerator

    gen      = DriveCycleGenerator(dt=DT, seed=999)
    drive_df = gen.generate(total_time_s=SIM_DURATION, profile="urban")

    bat  = BatteryModel(soc_init=0.60)
    sc   = SupercapacitorModel(soc_init=0.30)
    hess = HybridStorageSystem(bat, sc)

    records    = []
    prev_accel = 0.0
    total      = len(drive_df)
    pipeline   = model_dict["pipeline"]

    print(f"  0%", end="", flush=True)
    for i, (_, row) in enumerate(drive_df.iterrows()):
        if i > 0 and i % (total // 10) == 0:
            print(f"..{int(i/total*100)}%", end="", flush=True)

        power = row["braking_power_w"]
        jerk  = (row["accel_ms2"] - prev_accel) / DT

        if power > 0:
            soc_bat   = hess.battery.soc
            soc_sc    = hess.supercap.soc
            soc_ratio = soc_bat / max(soc_sc, 1e-6)
            X = np.array([[soc_bat, soc_sc, power/1000, jerk,
                           hess.battery.cycle_count, row["velocity_kmh"], soc_ratio]])
            cls         = int(pipeline.predict(X)[0])
            split_ratio = SPLIT_RATIO_MAP[cls]
            if soc_sc  >= 0.95: split_ratio = 1.0
            if soc_bat >= 0.95: split_ratio = 0.0
        else:
            split_ratio = 0.5
            cls = 1

        state = hess.step(power, split_ratio, DT)

        records.append({
            "time":             row["time"],
            "velocity_kmh":     row["velocity_kmh"],
            "accel_ms2":        row["accel_ms2"],
            "braking_power_kw": power / 1000,
            "split_ratio":      split_ratio,
            "split_class":      cls,
            "soc_battery":      state["battery"]["soc"] * 100,
            "soc_supercap":     state["supercap"]["soc"] * 100,
            "p_battery_kw":     state["p_battery"] / 1000,
            "p_supercap_kw":    state["p_supercap"] / 1000,
            "bat_cycles":       state["battery"]["cycles"],
        })
        prev_accel = row["accel_ms2"]

    print(" 100%!")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  PLOT RESULTS
# ─────────────────────────────────────────────
def plot_cosim_results(df: pd.DataFrame, title: str = "Co-Simulation Results"):
    fig = plt.figure(figsize=(16, 10))
    fig.suptitle(f"AI-Controlled HESS — {title}", fontsize=14, fontweight="bold")
    gs  = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.32)
    t   = df["time"] / 60

    # Speed profile
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(t, df["velocity_kmh"], alpha=0.25, color="#2196F3")
    ax1.plot(t, df["velocity_kmh"], color="#2196F3", linewidth=0.8)
    ax1.set_ylabel("Speed (km/h)"); ax1.set_title("Vehicle Speed Profile")
    ax1.grid(alpha=0.3); ax1.set_xlim(t.iloc[0], t.iloc[-1])

    # Battery SoC
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, df["soc_battery"], color="#4CAF50", linewidth=1.4)
    ax2.axhline(20, color="red",  linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.axhline(95, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax2.set_ylabel("SoC (%)"); ax2.set_title("Battery State of Charge")
    ax2.set_ylim(0, 105); ax2.grid(alpha=0.3); ax2.set_xlabel("Time (min)")

    # Supercap SoC
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, df["soc_supercap"], color="#9C27B0", linewidth=1.4)
    ax3.set_ylabel("SoC (%)"); ax3.set_title("Supercapacitor State of Charge")
    ax3.set_ylim(0, 105); ax3.grid(alpha=0.3); ax3.set_xlabel("Time (min)")

    # Power split
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.stackplot(t,
        df["p_battery_kw"].clip(0),
        df["p_supercap_kw"].clip(0),
        labels=["Battery (kW)", "Supercap (kW)"],
        colors=["#4CAF50", "#9C27B0"], alpha=0.7)
    ax4.set_ylabel("Power (kW)"); ax4.set_title("ML Agent — Power Split")
    ax4.legend(fontsize=8); ax4.grid(alpha=0.3); ax4.set_xlabel("Time (min)")

    # Split class distribution
    ax5 = fig.add_subplot(gs[2, 1])
    class_counts = df[df["braking_power_kw"] > 0]["split_class"].value_counts().sort_index()
    labels = ["Supercap-heavy\n(class 0)", "Balanced\n(class 1)", "Battery-heavy\n(class 2)"]
    colors = ["#9C27B0", "#2196F3", "#4CAF50"]
    ax5.bar([labels[i] for i in class_counts.index],
            class_counts.values,
            color=[colors[i] for i in class_counts.index], alpha=0.8)
    ax5.set_ylabel("Braking events"); ax5.set_title("ML Decision Distribution")
    ax5.grid(alpha=0.3, axis="y")

    out = os.path.join(RESULTS_DIR, "cosim_dashboard.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  Dashboard saved → {out}")
    plt.show()


# ─────────────────────────────────────────────
#  SUMMARY
# ─────────────────────────────────────────────
def print_summary(df: pd.DataFrame):
    braking = df[df["braking_power_kw"] > 0]
    class_dist = braking["split_class"].value_counts().sort_index()
    total_energy = (df["p_battery_kw"] + df["p_supercap_kw"]).sum() * DT / 3600 * 1000

    print("\n" + "=" * 55)
    print("  CO-SIMULATION RESULTS")
    print("=" * 55)
    print(f"  Total energy recovered : {total_energy:.1f} Wh")
    print(f"  Final battery SoC      : {df['soc_battery'].iloc[-1]:.1f}%")
    print(f"  Final supercap SoC     : {df['soc_supercap'].iloc[-1]:.1f}%")
    print(f"  Battery cycles         : {df['bat_cycles'].iloc[-1]:.4f}")
    print(f"  Total braking events   : {len(braking)}")
    print("-" * 55)
    print("  ML decision breakdown:")
    names = {0: "Supercap-heavy", 1: "Balanced", 2: "Battery-heavy"}
    for cls, count in class_dist.items():
        pct = count / len(braking) * 100
        print(f"    Class {cls} ({names[cls]:<14}): {count:>5} events  ({pct:.1f}%)")
    print("=" * 55)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  REGEN BRAKING — CO-SIMULATION")
    print("=" * 55)

    # Load ML agent
    print("\nLoading ML agent ...")
    model_dict = load_agent(AGENT_PATH)
    print(f"  Model loaded — features: {model_dict['features']}")

    # Try MATLAB Engine co-simulation first
    # If it fails or is too slow, falls back to pure Python
    USE_MATLAB = os.path.exists(MODEL_PATH)

    if USE_MATLAB:
        print(f"\nSimulink model found: {MODEL_PATH}")
        try:
            df = run_cosimulation(model_dict)
        except Exception as e:
            print(f"\n  MATLAB Engine error: {e}")
            print("  Switching to Python fallback ...")
            df = run_python_fallback(model_dict)
    else:
        print(f"\nSimulink model not found at: {MODEL_PATH}")
        print("Running Python fallback simulation ...")
        df = run_python_fallback(model_dict)

    # Save results
    out_csv = os.path.join(RESULTS_DIR, "cosim_results.csv")
    df.to_csv(out_csv, index=False)
    print(f"\nResults saved → {out_csv}")

    # Summary + plot
    print_summary(df)
    plot_cosim_results(df, title="Regenerative Braking HESS")
