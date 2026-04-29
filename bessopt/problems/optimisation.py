from abc import ABC, abstractmethod

import cvxpy
import numpy as np
import polars as pl

from bessopt.battery import Battery, BatteryConstraints


class BESSOptimisation(ABC):
    """Abstract base class for BESS mixed-integer optimisation problems.

    Subclasses share battery variable construction, physical constraints,
    and the solve() workflow. Each subclass implements its own power-balance
    and objective logic.
    """

    def __init__(
        self,
        battery: Battery,
        n_steps: int,
        degradation_cost: float = 0.0,
        battery_constraints: BatteryConstraints = None,
        product: str = '1h',
    ):
        if product not in ('1h', '15m'):
            raise ValueError(f"product must be '1h' or '15m', got '{product}'")

        self.battery = battery
        self.n_steps = n_steps
        self.degradation_cost = degradation_cost
        self.battery_constraints = battery_constraints or BatteryConstraints()
        self.product = product
        self.dt = 1.0 if product == '1h' else 0.25
        self.constraints: list = []
        self.problem: cvxpy.Problem | None = None

    def _build_variables(self) -> None:
        """Declare the five core CVXPY decision variables and the SoC expression."""
        self.battery_charge = cvxpy.Variable(self.n_steps, nonneg=True, name="battery_charge")
        self.battery_discharge = cvxpy.Variable(self.n_steps, nonneg=True, name="battery_discharge")
        self.battery_switch = cvxpy.Variable(self.n_steps, boolean=True, name="battery_switch")
        self.grid_in = cvxpy.Variable(self.n_steps, nonneg=True, name="grid_in")
        self.grid_out = cvxpy.Variable(self.n_steps, nonneg=True, name="grid_out")
        self.soc = (
            self.battery.soc
            + cvxpy.cumsum(
                self.battery_charge * self.battery.charge_efficiency
                - self.battery_discharge / self.battery.discharge_efficiency
            )
        )

    def _build_battery_constraints(self) -> list:
        """Return physical battery constraints shared by all subclasses."""
        bc, bat = self.battery_constraints, self.battery
        cons = [
            self.battery_charge <= self.battery_switch * bat.max_charge_power * self.dt,
            self.battery_discharge <= (1 - self.battery_switch) * bat.max_discharge_power * self.dt,
            self.soc >= bc.min_soc * bat.capacity,
            self.soc <= bc.max_soc * bat.capacity,
            cvxpy.sum(self.battery_discharge) / bat.capacity <= bc.max_daily_cycles,
        ]
        if bc.soc_end is not None:
            cons.append(self.soc[-1] >= bc.soc_end)
        return cons

    @abstractmethod
    def _build_constraints(self) -> None:
        """Populate self.constraints (must call _build_battery_constraints() inside)."""

    @abstractmethod
    def _build_objective(self) -> cvxpy.Maximize:
        """Return the CVXPY Maximize expression."""

    def solve(self) -> None:
        """Build and solve the MILP."""
        self._build_variables()
        self._build_constraints()
        objective = self._build_objective()
        self.problem = cvxpy.Problem(objective, self.constraints)
        self.problem.solve()

    @abstractmethod
    def get_results(self) -> pl.DataFrame: ...

    @property
    @abstractmethod
    def pnl(self) -> np.ndarray: ...

    def _repr_table(self, class_name: str, rows: list[tuple[str, str]]) -> str:
        col_w = max(len(label) for label, _ in rows)
        val_w = max(len(value) for _, value in rows)
        sep = f"+{'-' * (col_w + 2)}+{'-' * (val_w + 2)}+"
        fmt = f"| {{:<{col_w}}} | {{:<{val_w}}} |"
        lines = [
            sep,
            fmt.format(class_name, ""),
            sep,
            *(fmt.format(label, value) for label, value in rows),
            sep,
        ]
        return "\n".join(lines)
