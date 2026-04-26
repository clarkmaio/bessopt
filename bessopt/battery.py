

from dataclasses import dataclass


@dataclass
class BatteryConstraints:
    max_daily_cycles: float = 1
    min_soc: float = 0.0
    max_soc: float = 1.0
    soc_end: float = None


class Battery:

    def __init__(
        self,
        capacity: float,
        max_charge_power: float,
        max_discharge_power: float,
        charge_efficiency: float = 1.0,
        discharge_efficiency: float = 1.0,
        soc: float = 0.0,
    ):
        self._capacity            = capacity
        self._max_charge_power    = max_charge_power
        self._max_discharge_power = max_discharge_power
        self._charge_efficiency   = charge_efficiency
        self._discharge_efficiency = discharge_efficiency
        self._soc                 = soc

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def capacity(self) -> float:
        return self._capacity

    @property
    def max_charge_power(self) -> float:
        return self._max_charge_power

    @property
    def max_discharge_power(self) -> float:
        return self._max_discharge_power

    @property
    def charge_efficiency(self) -> float:
        return self._charge_efficiency

    @property
    def discharge_efficiency(self) -> float:
        return self._discharge_efficiency

    @property
    def soc(self) -> float:
        return self._soc

    def update_soc(self, value: float) -> None:
        self._soc = value

    def __repr__(self) -> str:
        rows = [
            ("Capacity",            f"{self.capacity} MWh"),
            ("Max charge power",    f"{self.max_charge_power} MW"),
            ("Max discharge power", f"{self.max_discharge_power} MW"),
            ("Charge efficiency",   f"{self.charge_efficiency}"),
            ("Discharge efficiency",f"{self.discharge_efficiency}"),
            ("Initial SoC",         f"{self.soc} MWh"),
        ]

        col_w = max(len(label) for label, _ in rows)
        val_w = max(len(value) for _, value in rows)
        sep   = f"+{'-' * (col_w + 2)}+{'-' * (val_w + 2)}+"
        fmt   = f"| {{:<{col_w}}} | {{:<{val_w}}} |"

        lines = [
            sep,
            fmt.format("Battery", ""),
            sep,
            *(fmt.format(label, value) for label, value in rows),
            sep,
        ]
        return "\n".join(lines)