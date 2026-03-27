"""
simulate.py
-----------
Full closed-loop simulation of the AI-controlled HESS
for regenerative braking. Runs entirely in Python
(no Simulink required to test the logic).

Compares:
    1. ML Agent (Random Forest)  — intelligent split
    2. Threshold Baseline        — simple rule: supercap first, then battery

Run:
    python simulate.py
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pickle
import os

from storage_model import BatteryModel, SupercapacitorModel, HybridStorageSystem
from drive_cycle import DriveCycleGenerator
from train_agent import (
    load_model, predict_split, generate_dataset, train_model,
    save_model, SPLIT_RATIO_MAP, optimal_split_class
)


# ─────────────────────────────────────────────
#  BASELINE CONTROLLER  (threshold-based)
# ─────────────────────────────────────────────
def threshold_split(soc_bat: float, soc_sc: float) -> float:
    """
    Simple rule: fill supercap first (high power density),
    then overflow to battery.
    """
    if soc_sc < 0.85:
        return 0.1   # Supercap-heavy
    elif soc_bat < 0.90:
        return 0.8   # Battery-heavy
    else:
        return 0.5   # Balanced (both near full)


# ─────────────────────────────────────────────
#  SINGLE SIMULATION RUN
# ─────────────────────────────────────────────
def run_simulation(
    drive_df: pd.DataFrame,
    model_dict: dict = None,
    use_ml: bool = True,
    dt: float = 0.1,
    label: str = "",
) -> pd.DataFrame:
    bat  = BatteryModel(soc_init=0.60)
    sc   = SupercapacitorModel(soc_init=0.30)
    hess = HybridStorageSystem(bat, sc)

    records    = []
    prev_accel = 0.0
    total      = len(drive_df)
    pipeline   = model_dict["pipeline"] if (use_ml and model_dict) else None
    features   = model_dict["features"] if (use_ml and model_dict) else None

    print(f"      0% ", end="", flush=True)

    for i, (_, row) in enumerate(drive_df.iterrows()):
        # Progress bar every 10%
        if i > 0 and i % (total // 10) == 0:
            pct = int(i / total * 100)
            print(f".. {pct}%", end="", flush=True)

        power = row["braking_power_w"]
        jerk  = (row["accel_ms2"] - prev_accel) / dt

        if power > 0:
            if pipeline is not None:
                soc_bat = hess.battery.soc
                soc_sc  = hess.supercap.soc
                soc_ratio = soc_bat / max(soc_sc, 1e-6)
                X = np.array([[
                    soc_bat, soc_sc, power / 1000,
                    jerk, hess.battery.cycle_count,
                    row["velocity_kmh"], soc_ratio
                ]])
                split_class = int(pipeline.predict(X)[0])
                split_ratio = SPLIT_RATIO_MAP[split_class]
            else:
                split_ratio = threshold_split(hess.battery.soc, hess.supercap.soc)
        else:
            split_ratio = 0.5

        state = hess.step(power, split_ratio, dt)

        records.append({
            "time":                row["time"],
            "velocity_kmh":        row["velocity_kmh"],
            "accel_ms2":           row["accel_ms2"],
            "braking_power_kw":    power / 1000,
            "split_ratio":         split_ratio,
            "soc_battery":         state["battery"]["soc"],
            "soc_supercap":        state["supercap"]["soc"],
            "p_battery_kw":        state["p_battery"] / 1000,
            "p_supercap_kw":       state["p_supercap"] / 1000,
            "bat_v_terminal":      state["battery"]["v_terminal"],
            "bat_cycles":          state["battery"]["cycles"],
            "energy_recovered_wh": state["total_energy_recovered"] / 3600,
        })

        prev_accel = row["accel_ms2"]

    print(" .. 100% ✓")
    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  RESULTS DASHBOARD
# ─────────────────────────────────────────────
def plot_results(ml_df: pd.DataFrame, base_df: pd.DataFrame, drive_df: pd.DataFrame):
    """
    Generate a comprehensive 6-panel results dashboard.
    """
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(
        "AI-Controlled Hybrid Energy Storage — Regenerative Braking Simulation",
        fontsize=14, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 2, hspace=0.45, wspace=0.32)

    t = ml_df["time"] / 60  # Convert to minutes

    # ── Panel 1: Vehicle speed profile ──────────
    ax1 = fig.add_subplot(gs[0, :])
    ax1.fill_between(t, drive_df["velocity_kmh"].values[:len(t)], alpha=0.25, color="#2196F3")
    ax1.plot(t, drive_df["velocity_kmh"].values[:len(t)], color="#2196F3", linewidth=0.8)
    ax1.set_ylabel("Speed (km/h)")
    ax1.set_title("Vehicle Speed Profile (Drive Cycle)", fontsize=10)
    ax1.set_xlim(t.iloc[0], t.iloc[-1])
    ax1.grid(alpha=0.3)

    # ── Panel 2: Battery SoC comparison ─────────
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(t, ml_df["soc_battery"] * 100,   color="#4CAF50", linewidth=1.2, label="ML Agent")
    ax2.plot(t, base_df["soc_battery"] * 100, color="#F44336", linewidth=1.2, linestyle="--", label="Threshold")
    ax2.axhline(20, color="red",  linestyle=":", linewidth=0.8, alpha=0.5, label="SoC min (20%)")
    ax2.axhline(95, color="gray", linestyle=":", linewidth=0.8, alpha=0.5, label="SoC max (95%)")
    ax2.set_ylabel("Battery SoC (%)")
    ax2.set_title("Battery State of Charge", fontsize=10)
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 105)
    ax2.grid(alpha=0.3)
    ax2.set_xlabel("Time (min)")

    # ── Panel 3: Supercapacitor SoC comparison ───
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(t, ml_df["soc_supercap"] * 100,   color="#9C27B0", linewidth=1.2, label="ML Agent")
    ax3.plot(t, base_df["soc_supercap"] * 100, color="#FF9800", linewidth=1.2, linestyle="--", label="Threshold")
    ax3.set_ylabel("Supercapacitor SoC (%)")
    ax3.set_title("Supercapacitor State of Charge", fontsize=10)
    ax3.legend(fontsize=8)
    ax3.set_ylim(0, 105)
    ax3.grid(alpha=0.3)
    ax3.set_xlabel("Time (min)")

    # ── Panel 4: Power split (ML only) ──────────
    ax4 = fig.add_subplot(gs[2, 0])
    ax4.stackplot(
        t,
        ml_df["p_battery_kw"].clip(0),
        ml_df["p_supercap_kw"].clip(0),
        labels=["Battery power (kW)", "Supercap power (kW)"],
        colors=["#4CAF50", "#9C27B0"],
        alpha=0.7,
    )
    ax4.set_ylabel("Power (kW)")
    ax4.set_title("ML Agent — Power Split During Braking", fontsize=10)
    ax4.legend(fontsize=8)
    ax4.grid(alpha=0.3)
    ax4.set_xlabel("Time (min)")

    # ── Panel 5: Energy recovered comparison ────
    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(t, ml_df["energy_recovered_wh"],   color="#4CAF50", linewidth=1.4, label="ML Agent")
    ax5.plot(t, base_df["energy_recovered_wh"], color="#F44336", linewidth=1.4, linestyle="--", label="Threshold")
    ax5.set_ylabel("Energy Recovered (Wh)")
    ax5.set_title("Cumulative Energy Recovery", fontsize=10)
    ax5.legend(fontsize=8)
    ax5.grid(alpha=0.3)
    ax5.set_xlabel("Time (min)")

    plt.savefig("results/simulation_dashboard.png", dpi=150, bbox_inches="tight")
    print("  Dashboard saved → results/simulation_dashboard.png")
    plt.show()


# ─────────────────────────────────────────────
#  SUMMARY METRICS
# ─────────────────────────────────────────────
def print_summary(ml_df: pd.DataFrame, base_df: pd.DataFrame):
    def metrics(df, name):
        e_rec  = df["energy_recovered_wh"].iloc[-1]
        cycles = df["bat_cycles"].iloc[-1]
        soc_end_bat = df["soc_battery"].iloc[-1]
        soc_end_sc  = df["soc_supercap"].iloc[-1]
        return {
            "Controller":           name,
            "Energy Recovered (Wh)":f"{e_rec:.2f}",
            "Battery Cycles":       f"{cycles:.4f}",
            "Final Battery SoC":    f"{soc_end_bat*100:.1f}%",
            "Final Supercap SoC":   f"{soc_end_sc*100:.1f}%",
        }

    ml_m   = metrics(ml_df,   "ML Agent (RF)")
    base_m = metrics(base_df, "Threshold")

    ml_e   = float(ml_df["energy_recovered_wh"].iloc[-1])
    base_e = float(base_df["energy_recovered_wh"].iloc[-1])
    improvement = (ml_e - base_e) / max(base_e, 1e-6) * 100

    ml_cyc   = float(ml_df["bat_cycles"].iloc[-1])
    base_cyc = float(base_df["bat_cycles"].iloc[-1])
    cycle_reduction = (base_cyc - ml_cyc) / max(base_cyc, 1e-6) * 100

    print("\n" + "=" * 60)
    print("  SIMULATION RESULTS SUMMARY")
    print("=" * 60)
    for key in ml_m:
        print(f"  {key:<28} ML: {ml_m[key]:<12}  Base: {base_m[key]}")
    print("-" * 60)
    print(f"  Energy improvement (ML vs Baseline): +{improvement:.1f}%")
    print(f"  Battery cycle reduction:              {cycle_reduction:.1f}%")
    print("=" * 60)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────
if __name__ == "__main__":
    os.makedirs("data",    exist_ok=True)
    os.makedirs("model",   exist_ok=True)
    os.makedirs("results", exist_ok=True)

    MODEL_PATH = "model/rf_agent.pkl"

    # ── Train model if not already saved ────────
    if not os.path.exists(MODEL_PATH):
        print("[1/4] Training ML model (first run) ...")
        df_train = generate_dataset(n_cycles=6)
        result   = train_model(df_train)
        save_model(result, MODEL_PATH)
    else:
        print("[1/4] Loading existing ML model ...")

    model_dict = load_model(MODEL_PATH)

    # ── Generate test drive cycle ────────────────
    print("[2/4] Generating test drive cycle ...")
    gen      = DriveCycleGenerator(dt=0.1, seed=999)
    drive_df = gen.generate(total_time_s=1200, profile="urban")

    # ── Run both simulations ─────────────────────
    print("[3/4] Running ML agent simulation ...")
    ml_results = run_simulation(drive_df, model_dict, use_ml=True,  label="ML Agent")

    print("      Running baseline simulation ...")
    base_results = run_simulation(drive_df, model_dict, use_ml=False, label="Threshold")

    # ── Save results ─────────────────────────────
    ml_results.to_csv("results/ml_results.csv",       index=False)
    base_results.to_csv("results/baseline_results.csv", index=False)

    # ── Print summary ────────────────────────────
    print_summary(ml_results, base_results)

    # ── Plot dashboard ───────────────────────────
    print("[4/4] Generating results dashboard ...")
    plot_results(ml_results, base_results, drive_df)
