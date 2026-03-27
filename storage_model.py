"""
storage_model.py
----------------
Thevenin equivalent battery model and RC supercapacitor model.
Tracks State of Charge (SoC), voltage, current, and power for both
storage elements in a hybrid energy storage system (HESS).
"""

import numpy as np


# ─────────────────────────────────────────────
#  BATTERY MODEL  (1st-order Thevenin)
# ─────────────────────────────────────────────
class BatteryModel:
    """
    1st-order Thevenin equivalent battery model.

    Equivalent circuit:
        OCV(SoC) ── R0 ── [ R1 ║ C1 ] ── terminal

    Parameters
    ----------
    capacity_ah   : Nominal capacity in Ah
    v_nominal     : Nominal voltage (V)
    r0            : Series internal resistance (Ω)
    r1            : Polarization resistance (Ω)
    c1            : Polarization capacitance (F)
    soc_init      : Initial SoC (0–1)
    soc_min       : Minimum allowable SoC
    soc_max       : Maximum allowable SoC
    """

    def __init__(
        self,
        capacity_ah: float = 50.0,
        v_nominal: float   = 48.0,
        r0: float          = 0.01,
        r1: float          = 0.005,
        c1: float          = 2000.0,
        soc_init: float    = 0.6,
        soc_min: float     = 0.2,
        soc_max: float     = 0.95,
    ):
        self.capacity_ah = capacity_ah
        self.capacity_as = capacity_ah * 3600          # Convert to Coulombs
        self.v_nominal   = v_nominal
        self.r0, self.r1, self.c1 = r0, r1, c1
        self.soc_min     = soc_min
        self.soc_max     = soc_max

        # State variables
        self.soc         = soc_init
        self.v_rc        = 0.0                         # Voltage across R1‖C1
        self.v_terminal  = self._ocv(soc_init)
        self.current     = 0.0
        self.power       = 0.0
        self.energy_in   = 0.0                         # Total energy charged (J)
        self.energy_out  = 0.0                         # Total energy discharged (J)
        self.cycle_count = 0.0                         # Accumulated half-cycles

    # ── Open-circuit voltage vs SoC (polynomial fit, typical Li-ion) ──
    def _ocv(self, soc: float) -> float:
        return (
            self.v_nominal * (
                0.7 +
                0.18 * soc +
                0.05 * soc**2 -
                0.02 * np.exp(-10 * soc) +
                0.02 * np.exp(10 * (soc - 1))
            )
        )

    def step(self, power_w: float, dt: float) -> dict:
        """
        Advance the battery model by one time step.

        Parameters
        ----------
        power_w : Power demand (W).
                  Positive  = charging (regenerative braking input)
                  Negative  = discharging (traction)
        dt      : Time step (s)

        Returns a dict of current state variables.
        """
        # Clamp SoC limits
        if self.soc >= self.soc_max and power_w > 0:
            power_w = 0.0
        if self.soc <= self.soc_min and power_w < 0:
            power_w = 0.0

        ocv = self._ocv(self.soc)

        # Solve for current: P = I * V_terminal = I * (OCV - I*R0 - V_rc)
        # Quadratic:  R0 * I^2 - (OCV - V_rc) * I + P = 0
        a = self.r0
        b = -(ocv - self.v_rc)
        c_coef = power_w
        discriminant = b**2 - 4 * a * c_coef

        if discriminant < 0:
            discriminant = 0.0

        # Choose the physically meaningful root (smaller magnitude current)
        i1 = (-b + np.sqrt(discriminant)) / (2 * a)
        i2 = (-b - np.sqrt(discriminant)) / (2 * a)
        current = i1 if abs(i1) < abs(i2) else i2

        # Update RC network voltage (1st-order Euler)
        dv_rc = (current / self.c1 - self.v_rc / (self.r1 * self.c1)) * dt
        self.v_rc += dv_rc

        # Update SoC using Coulomb counting
        # Convention: positive current charges the battery
        delta_soc = (current * dt) / self.capacity_as
        self.soc = np.clip(self.soc + delta_soc, 0.0, 1.0)

        # Terminal voltage
        self.v_terminal = ocv - current * self.r0 - self.v_rc

        # Energy accounting
        actual_power = current * self.v_terminal
        if actual_power > 0:
            self.energy_in  += actual_power * dt
        else:
            self.energy_out += abs(actual_power) * dt

        # Cycle counting (incremental, based on SoC change)
        self.cycle_count += abs(delta_soc) / 2.0

        self.current = current
        self.power   = actual_power

        return self._state()

    def _state(self) -> dict:
        return {
            "soc":        self.soc,
            "v_terminal": self.v_terminal,
            "v_ocv":      self._ocv(self.soc),
            "current":    self.current,
            "power":      self.power,
            "energy_in":  self.energy_in,
            "energy_out": self.energy_out,
            "cycles":     self.cycle_count,
        }

    def reset(self, soc: float = 0.6):
        self.__init__(
            self.capacity_ah, self.v_nominal, self.r0, self.r1,
            self.c1, soc, self.soc_min, self.soc_max
        )


# ─────────────────────────────────────────────
#  SUPERCAPACITOR MODEL  (RC equivalent)
# ─────────────────────────────────────────────
class SupercapacitorModel:
    """
    RC equivalent supercapacitor model.

    Equivalent circuit:
        V_cap ── ESR ── terminal

    SoC is defined as energy-based: SoC = (V / V_max)^2

    Parameters
    ----------
    capacitance_f : Capacitance in Farads
    v_max         : Maximum voltage (V)
    v_min         : Minimum voltage (V)  — defines SoC = 0
    esr           : Equivalent series resistance (Ω)
    soc_init      : Initial SoC (0–1)
    """

    def __init__(
        self,
        capacitance_f: float = 3000.0,  # Realistic EV supercap bank (~100 Wh usable)
        v_max: float         = 48.0,
        v_min: float         = 20.0,
        esr: float           = 0.002,
        soc_init: float      = 0.3,
    ):
        self.capacitance = capacitance_f
        self.v_max       = v_max
        self.v_min       = v_min
        self.esr         = esr

        # Derive initial voltage from SoC
        self.v_cap       = v_min + soc_init * (v_max - v_min)
        self.current     = 0.0
        self.power       = 0.0
        self.energy_in   = 0.0
        self.energy_out  = 0.0

    @property
    def soc(self) -> float:
        """Energy-based SoC: 0 at V_min, 1 at V_max."""
        v_range = self.v_max - self.v_min
        if v_range == 0:
            return 0.0
        return np.clip((self.v_cap - self.v_min) / v_range, 0.0, 1.0)

    def step(self, power_w: float, dt: float) -> dict:
        """
        Advance the supercapacitor model by one time step.

        Parameters
        ----------
        power_w : Power demand (W). Positive = charging, Negative = discharging.
        dt      : Time step (s)
        """
        # SoC hard limits
        if self.v_cap >= self.v_max and power_w > 0:
            power_w = 0.0
        if self.v_cap <= self.v_min and power_w < 0:
            power_w = 0.0

        # Solve for current: P = I * (V_cap - I * ESR)
        # Quadratic: ESR * I^2 - V_cap * I + P = 0
        a = self.esr
        b = -self.v_cap
        c_coef = power_w
        discriminant = b**2 - 4 * a * c_coef

        if discriminant < 0:
            discriminant = 0.0

        i1 = (-b + np.sqrt(discriminant)) / (2 * a)
        i2 = (-b - np.sqrt(discriminant)) / (2 * a)
        current = i1 if abs(i1) < abs(i2) else i2

        # Update capacitor voltage: dV/dt = I / C
        dv = (current / self.capacitance) * dt
        self.v_cap = np.clip(self.v_cap + dv, self.v_min, self.v_max)

        # Terminal voltage
        v_terminal = self.v_cap - current * self.esr

        # Energy accounting
        actual_power = current * v_terminal
        if actual_power > 0:
            self.energy_in  += actual_power * dt
        else:
            self.energy_out += abs(actual_power) * dt

        self.current = current
        self.power   = actual_power

        return self._state()

    def _state(self) -> dict:
        return {
            "soc":       self.soc,
            "v_cap":     self.v_cap,
            "current":   self.current,
            "power":     self.power,
            "energy_in": self.energy_in,
            "energy_out":self.energy_out,
        }

    def reset(self, soc: float = 0.5):
        v_init = self.v_min + soc * (self.v_max - self.v_min)
        self.v_cap     = v_init
        self.current   = 0.0
        self.power     = 0.0
        self.energy_in = self.energy_out = 0.0


# ─────────────────────────────────────────────
#  HYBRID ENERGY STORAGE SYSTEM
# ─────────────────────────────────────────────
class HybridStorageSystem:
    """
    Manages both storage elements together.
    Applies a power split ratio to distribute braking power.

    split_ratio : float in [0, 1]
        0.0 → all power to supercapacitor
        1.0 → all power to battery
        0.5 → equal split
    """

    def __init__(self, battery: BatteryModel, supercap: SupercapacitorModel):
        self.battery  = battery
        self.supercap = supercap

    def step(self, braking_power_w: float, split_ratio: float, dt: float) -> dict:
        """
        Distribute braking power between battery and supercapacitor.
        Includes overflow redirect: if one element is full, excess goes to the other.
        """
        split_ratio = np.clip(split_ratio, 0.0, 1.0)

        # Override split if one element is at its limit
        sc_full  = self.supercap.soc >= 0.95
        bat_full = self.battery.soc  >= self.battery.soc_max
        sc_empty = self.supercap.soc <= 0.05

        if sc_full and not bat_full:
            split_ratio = 1.0   # Supercap full → all to battery
        elif bat_full and not sc_full:
            split_ratio = 0.0   # Battery full → all to supercap
        elif sc_full and bat_full:
            split_ratio = 0.5   # Both full → split evenly (will be clamped internally)

        p_battery  = braking_power_w * split_ratio
        p_supercap = braking_power_w * (1.0 - split_ratio)

        bat_state = self.battery.step(p_battery, dt)
        sc_state  = self.supercap.step(p_supercap, dt)

        total_energy_recovered = (
            self.battery.energy_in + self.supercap.energy_in
        )

        return {
            "battery":               bat_state,
            "supercap":              sc_state,
            "split_ratio":           split_ratio,
            "p_battery":             p_battery,
            "p_supercap":            p_supercap,
            "total_energy_recovered":total_energy_recovered,
        }
