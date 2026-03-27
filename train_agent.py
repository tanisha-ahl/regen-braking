"""
train_agent.py
--------------
Generates a labelled dataset from drive cycle simulations and
trains a Random Forest classifier to determine the optimal
power split ratio between battery and supercapacitor.

Split classes:
    0 → Supercap-heavy   (split_ratio = 0.10)  — high jerk, supercap protects battery
    1 → Balanced         (split_ratio = 0.50)  — moderate braking
    2 → Battery-heavy    (split_ratio = 0.80)  — supercap near full, battery has headroom
"""

import numpy as np
import pandas as pd
import pickle
import os
from typing import Tuple as Tuple_hint
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

from storage_model import BatteryModel, SupercapacitorModel, HybridStorageSystem
from drive_cycle import DriveCycleGenerator


# ─────────────────────────────────────────────
#  OPTIMAL SPLIT RULE  (physics-based heuristic)
# ─────────────────────────────────────────────
def optimal_split_class(
    soc_bat: float,
    soc_sc: float,
    braking_power_w: float,
    jerk: float,
    bat_cycles: float,
) -> int:
    """
    Rule-based oracle to label training data.
    Produces balanced classes across real operating conditions.

    Class 0 — Supercap-heavy (split_ratio = 0.10):
        High jerk bursts, or battery near full, or supercap has room + low power

    Class 1 — Balanced (split_ratio = 0.50):
        Moderate SoC on both, moderate power and jerk

    Class 2 — Battery-heavy (split_ratio = 0.80):
        Supercap near full, or sustained high power, or supercap nearly empty
    """
    HIGH_JERK    = 1.5     # m/s³  — reduced threshold → more class 0 events
    MED_JERK     = 0.5     # m/s³
    HIGH_POWER   = 20.0    # kW
    MED_POWER    = 8.0     # kW

    SC_HIGH      = 0.75    # supercap getting full → push to battery
    SC_LOW       = 0.25    # supercap low → use battery instead
    BAT_HIGH     = 0.85    # battery getting full → push to supercap
    BAT_LOW      = 0.30    # battery low → preserve it, use supercap

    power_kw = braking_power_w / 1000.0

    # ── Class 0: Supercap-heavy ──────────────────────────────
    # High-jerk spike → supercap absorbs it
    if abs(jerk) > HIGH_JERK:
        return 0
    # Battery nearly full → avoid charging it more
    if soc_bat > BAT_HIGH:
        return 0
    # Both have room and power is low → supercap is ideal
    if soc_sc < SC_HIGH and power_kw < MED_POWER and abs(jerk) > MED_JERK:
        return 0
    # Battery is low on charge → protect it, use supercap
    if soc_bat < BAT_LOW and soc_sc > SC_LOW:
        return 0

    # ── Class 2: Battery-heavy ───────────────────────────────
    # Supercap is nearly full → overflow to battery
    if soc_sc > SC_HIGH:
        return 2
    # Supercap too low to contribute meaningfully
    if soc_sc < SC_LOW:
        return 2
    # High sustained power → battery can handle it better
    if power_kw > HIGH_POWER and abs(jerk) < MED_JERK:
        return 2

    # ── Class 1: Balanced ────────────────────────────────────
    # Both devices in mid-range, moderate conditions
    return 1


SPLIT_RATIO_MAP = {0: 0.10, 1: 0.50, 2: 0.80}


# ─────────────────────────────────────────────
#  DATASET GENERATION
# ─────────────────────────────────────────────
def generate_dataset(
    n_cycles: int = 8,
    profiles: list = None,
    dt: float = 0.1,
) -> pd.DataFrame:
    """
    Run multiple drive cycle simulations and collect
    labelled state snapshots for ML training.

    Features collected at each braking event:
        soc_battery, soc_supercap, braking_power_kw,
        jerk, bat_cycles, velocity_kmh, soc_ratio

    Label:
        split_class (0 / 1 / 2)
    """
    if profiles is None:
        profiles = ["urban", "suburban", "highway"]

    gen     = DriveCycleGenerator(dt=dt)
    records = []

    for i in range(n_cycles):
        profile = profiles[i % len(profiles)]
        seed    = 42 + i * 7
        gen.rng = np.random.default_rng(seed)

        df = gen.generate(total_time_s=1200, profile=profile)

        # Fresh HESS for each cycle with varied initial conditions
        bat = BatteryModel(soc_init=np.random.uniform(0.3, 0.8))
        sc  = SupercapacitorModel(soc_init=np.random.uniform(0.2, 0.7))
        hess = HybridStorageSystem(bat, sc)

        prev_accel = 0.0

        for idx, row in df.iterrows():
            power = row["braking_power_w"]

            if power <= 0:
                # Not braking — advance with zero power, no label
                hess.step(0.0, 0.5, dt)
                prev_accel = row["accel_ms2"]
                continue

            # Jerk = rate of change of acceleration
            jerk = (row["accel_ms2"] - prev_accel) / dt

            soc_bat = hess.battery.soc
            soc_sc  = hess.supercap.soc
            bat_cyc = hess.battery.cycle_count

            label = optimal_split_class(soc_bat, soc_sc, power, jerk, bat_cyc)
            split = SPLIT_RATIO_MAP[label]

            records.append({
                "soc_battery":      soc_bat,
                "soc_supercap":     soc_sc,
                "braking_power_kw": power / 1000,
                "jerk":             jerk,
                "bat_cycles":       bat_cyc,
                "velocity_kmh":     row["velocity_kmh"],
                "soc_ratio":        soc_bat / max(soc_sc, 1e-6),
                "split_class":      label,
            })

            hess.step(power, split, dt)
            prev_accel = row["accel_ms2"]

        print(f"  Cycle {i+1}/{n_cycles} ({profile}) — {len(records)} samples so far")

    return pd.DataFrame(records)


# ─────────────────────────────────────────────
#  MODEL TRAINING
# ─────────────────────────────────────────────
def train_model(df: pd.DataFrame) -> dict:
    """
    Train a Random Forest classifier on the labelled dataset.

    Returns a dict containing:
        pipeline   : trained sklearn Pipeline (scaler + classifier)
        report     : classification report string
        cv_scores  : cross-validation accuracy scores
        feature_importance : dict of feature → importance
    """
    FEATURES = [
        "soc_battery", "soc_supercap", "braking_power_kw",
        "jerk", "bat_cycles", "velocity_kmh", "soc_ratio"
    ]
    TARGET = "split_class"

    X = df[FEATURES].values
    y = df[TARGET].values

    print(f"\nDataset size: {len(df)} samples")
    print(f"Class distribution:\n{pd.Series(y).value_counts().sort_index()}\n")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Pipeline: StandardScaler + Random Forest
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(pipeline, X, y, cv=cv, scoring="accuracy")

    report = classification_report(
        y_test, y_pred,
        target_names=["Supercap-heavy", "Balanced", "Battery-heavy"]
    )

    # Feature importances
    importances = pipeline.named_steps["clf"].feature_importances_
    feat_imp = dict(zip(FEATURES, importances))

    print("Classification Report:")
    print(report)
    print(f"Cross-validation accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
    print("\nFeature Importances:")
    for k, v in sorted(feat_imp.items(), key=lambda x: -x[1]):
        print(f"  {k:<22}: {v:.4f}")

    return {
        "pipeline":           pipeline,
        "report":             report,
        "cv_scores":          cv_scores,
        "feature_importance": feat_imp,
        "features":           FEATURES,
    }


# ─────────────────────────────────────────────
#  SAVE / LOAD
# ─────────────────────────────────────────────
def save_model(model_dict: dict, path: str = "model/rf_agent.pkl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model_dict, f)
    print(f"\nModel saved → {path}")


def load_model(path: str = "model/rf_agent.pkl") -> dict:
    with open(path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
#  PREDICTION INTERFACE
# ─────────────────────────────────────────────
def predict_split(
    model_dict: dict,
    soc_battery: float,
    soc_supercap: float,
    braking_power_kw: float,
    jerk: float,
    bat_cycles: float,
    velocity_kmh: float,
) -> Tuple_hint:
    """
    Predict the optimal split class and return the split ratio.

    Returns (split_class, split_ratio, class_probabilities)
    """
    soc_ratio = soc_battery / max(soc_supercap, 1e-6)
    X = np.array([[
        soc_battery, soc_supercap, braking_power_kw,
        jerk, bat_cycles, velocity_kmh, soc_ratio
    ]])
    pipeline   = model_dict["pipeline"]
    split_class = int(pipeline.predict(X)[0])
    probs       = pipeline.predict_proba(X)[0]
    split_ratio = SPLIT_RATIO_MAP[split_class]
    return split_class, split_ratio, probs


# ─────────────────────────────────────────────
#  MAIN — Run training
# ─────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  REGEN BRAKING — ML AGENT TRAINING")
    print("=" * 55)

    print("\n[1/3] Generating drive cycle dataset ...")
    df = generate_dataset(n_cycles=8)
    df.to_csv("data/training_data.csv", index=False)
    print(f"      Saved {len(df)} samples → data/training_data.csv")

    print("\n[2/3] Training Random Forest classifier ...")
    result = train_model(df)

    print("\n[3/3] Saving model ...")
    save_model(result, "model/rf_agent.pkl")

    print("\nDone. Model ready for co-simulation.")
