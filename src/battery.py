


from dataclasses import dataclass


@dataclass
class Battery:
    capacity: float
    max_charge_power: float
    max_discharge_power: float
    charge_efficiency: float = 1
    discharge_efficiency: float = 1
    soc: float = 0
    min_soc: float = 0.0
    soc_end: float = None       # if None, no terminal SoC constraint is applied
    max_daily_cycles: float = 1
    

    def __repr__(self) -> str:
        rows = [
            ("Capacity",            f"{self.capacity} MWh"),
            ("Max charge power",    f"{self.max_charge_power} MW"),
            ("Max discharge power", f"{self.max_discharge_power} MW"),
            ("Charge efficiency",   f"{self.charge_efficiency}"),
            ("Discharge efficiency",f"{self.discharge_efficiency}"),
            ("Initial SoC",         f"{self.soc} MWh"),
            ("Min SoC",             f"{self.min_soc * 100:.0f}%"),
            ("Terminal SoC",        f"{self.soc_end} MWh" if self.soc_end is not None else "unconstrained"),
            ("Max daily cycles",    f"{self.max_daily_cycles}"),
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