"""
drive_cycle.py
--------------
Vehicle dynamics model and synthetic drive cycle generator.
Computes braking power from vehicle kinematics and produces
labelled datasets for ML training.
"""

import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Tuple


# ─────────────────────────────────────────────
#  VEHICLE PARAMETERS
# ─────────────────────────────────────────────
@dataclass
class VehicleParams:
    mass_kg:       float = 1500.0    # Vehicle mass (kg)
    cd:            float = 0.30      # Drag coefficient
    frontal_area:  float = 2.2       # Frontal area (m²)
    crr:           float = 0.012     # Rolling resistance coefficient
    regen_eff:     float = 0.85      # Regenerative braking efficiency
    motor_eff:     float = 0.92      # Motor/generator efficiency
    max_regen_kw:  float = 50.0      # Maximum regen power (kW)
    air_density:   float = 1.225     # kg/m³


# ─────────────────────────────────────────────
#  VEHICLE DYNAMICS
# ─────────────────────────────────────────────
class VehicleDynamics:
    """
    Computes instantaneous braking power from vehicle state.

    Forces acting on the vehicle:
        F_drag      = 0.5 * rho * Cd * A * v²
        F_rolling   = Crr * m * g
        F_braking   = m * a  (deceleration force)

    Regenerated power = F_braking * v * η_regen (when decelerating)
    """

    def __init__(self, params: VehicleParams = None):
        self.p = params or VehicleParams()
        self.g = 9.81

    def braking_power(self, velocity_ms: float, accel_ms2: float) -> float:
        """
        Compute recoverable braking power (W).

        Returns positive value during braking (deceleration),
        zero during acceleration or cruise.

        Parameters
        ----------
        velocity_ms : Vehicle speed (m/s)
        accel_ms2   : Acceleration (m/s²). Negative = braking.
        """
        if accel_ms2 >= 0 or velocity_ms <= 0:
            return 0.0

        # Kinetic braking force
        f_brake = self.p.mass_kg * abs(accel_ms2)

        # Parasitic losses (drag + rolling) — these are not recoverable
        f_drag    = 0.5 * self.p.air_density * self.p.cd * self.p.frontal_area * velocity_ms**2
        f_rolling = self.p.crr * self.p.mass_kg * self.g

        # Net recoverable force (subtract parasitic from available braking)
        f_regen = max(0.0, f_brake - f_drag - f_rolling)

        # Power with efficiency
        power = f_regen * velocity_ms * self.p.regen_eff * self.p.motor_eff

        # Cap at motor limit
        power = min(power, self.p.max_regen_kw * 1000)

        return power


# ─────────────────────────────────────────────
#  DRIVE CYCLE GENERATOR
# ─────────────────────────────────────────────
class DriveCycleGenerator:
    """
    Generates synthetic drive cycles that mimic real-world
    urban, suburban, and highway driving patterns.

    Each cycle consists of segments:
        ACCEL → CRUISE → BRAKE → IDLE  (repeated)
    """

    def __init__(self, dt: float = 0.1, seed: int = 42):
        self.dt   = dt      # Time step (seconds)
        self.rng  = np.random.default_rng(seed)

    def _segment(
        self,
        v_start: float,
        v_target: float,
        duration: float,
        mode: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate velocity and acceleration arrays for one segment."""
        n   = int(duration / self.dt)
        t   = np.linspace(0, duration, n)

        if mode == "accel":
            # Smooth acceleration using sigmoid-like ramp
            v = v_start + (v_target - v_start) * (t / duration)
            v = np.clip(v, 0, None)
        elif mode == "cruise":
            v = np.full(n, v_target) + self.rng.normal(0, 0.2, n)
            v = np.clip(v, 0, None)
        elif mode == "brake":
            v = v_start * np.maximum(0, 1 - (t / duration) ** 0.8)
        elif mode == "idle":
            v = np.zeros(n)
        else:
            v = np.zeros(n)

        # Compute acceleration by finite difference
        a = np.gradient(v, self.dt)
        return v, a

    def generate(
        self,
        total_time_s: float = 1200.0,
        profile: str = "urban",
    ) -> pd.DataFrame:
        """
        Generate a full drive cycle.

        Parameters
        ----------
        total_time_s : Total simulation duration (seconds)
        profile      : 'urban' | 'suburban' | 'highway'

        Returns a DataFrame with columns:
            time, velocity_ms, velocity_kmh, accel_ms2,
            braking_power_w, braking_power_kw, is_braking
        """
        profiles = {
            "urban":    dict(v_range=(5, 14),  accel_t=(4, 8),   cruise_t=(5, 15),  brake_t=(3, 7),  idle_t=(3, 8)),
            "suburban": dict(v_range=(10, 22), accel_t=(5, 10),  cruise_t=(8, 20),  brake_t=(4, 8),  idle_t=(2, 5)),
            "highway":  dict(v_range=(20, 33), accel_t=(8, 15),  cruise_t=(15, 40), brake_t=(5, 12), idle_t=(1, 3)),
        }
        cfg = profiles.get(profile, profiles["urban"])
        dyn = VehicleDynamics()

        velocities   = []
        accelerations = []
        current_v    = 0.0
        elapsed      = 0.0

        while elapsed < total_time_s:
            # Random segment parameters
            v_target   = self.rng.uniform(*cfg["v_range"])
            t_accel    = self.rng.uniform(*cfg["accel_t"])
            t_cruise   = self.rng.uniform(*cfg["cruise_t"])
            t_brake    = self.rng.uniform(*cfg["brake_t"])
            t_idle     = self.rng.uniform(*cfg["idle_t"])

            # Acceleration segment
            v_seg, a_seg = self._segment(current_v, v_target, t_accel, "accel")
            velocities.append(v_seg); accelerations.append(a_seg)

            # Cruise segment
            v_seg, a_seg = self._segment(v_target, v_target, t_cruise, "cruise")
            velocities.append(v_seg); accelerations.append(a_seg)

            # Braking segment (back to 0)
            v_seg, a_seg = self._segment(v_target, 0.0, t_brake, "brake")
            velocities.append(v_seg); accelerations.append(a_seg)

            # Idle segment
            v_seg, a_seg = self._segment(0.0, 0.0, t_idle, "idle")
            velocities.append(v_seg); accelerations.append(a_seg)

            current_v  = 0.0
            elapsed   += t_accel + t_cruise + t_brake + t_idle

        v_arr = np.concatenate(velocities)
        a_arr = np.concatenate(accelerations)
        n     = len(v_arr)
        t_arr = np.arange(n) * self.dt

        # Braking power at each step
        bp = np.array([
            dyn.braking_power(v_arr[i], a_arr[i]) for i in range(n)
        ])

        df = pd.DataFrame({
            "time":            t_arr,
            "velocity_ms":     v_arr,
            "velocity_kmh":    v_arr * 3.6,
            "accel_ms2":       a_arr,
            "braking_power_w": bp,
            "braking_power_kw":bp / 1000,
            "is_braking":      (a_arr < -0.1).astype(int),
        })

        return df.iloc[:int(total_time_s / self.dt)].reset_index(drop=True)
