import cvxpy
import numpy as np
import polars as pl

from bessopt.battery import Battery, BatteryConstraints
from bessopt.problems.optimisation import BESSOptimisation
from bessopt.utility import Utility
from bessopt.viz import plot_da_schedule


class AuctionOptimisation(BESSOptimisation):
    """Day-Ahead market battery optimisation using mixed-integer linear programming.

    Maximises revenue from buying (grid_out) and selling (grid_in) energy on the
    day-ahead market, subject to battery physical constraints (capacity, charge/
    discharge power limits, and round-trip efficiency).

    The charging and discharging directions are enforced via binary switch variables
    so that the battery cannot charge and discharge simultaneously, and the grid
    cannot import and export simultaneously.

    When ``battery_charge_schedule`` and ``battery_discharge_schedule`` are both
    provided, the optimiser is run in *residual* mode (rolling-intrinsic, mirroring
    :class:`IntradayOptimisation`): the existing battery charge/discharge dispatch
    is treated as already monetised, and revenue is computed only on the
    incremental market trade between the new schedule and the existing one.

    Attributes:
        battery: Physical parameters of the battery asset.
        price: Day-ahead price series (one value per optimisation step), in €/MWh.
        pv: Optional PV generation profile [MW].
        c_bar: Optional existing charge schedule [MW] (the "bar" quantity from
            Oeltz & Pfingsten 2025). Same length as ``price``.
        d_bar: Optional existing discharge schedule [MW]. Same length as ``price``.
        n_steps: Number of time steps derived from the price series length.
    """

    def _input_validation(self, price, pv, battery_charge_schedule, battery_discharge_schedule):
        n_steps = len(price)
        if pv is not None and len(pv) != n_steps:
            raise ValueError(
                f"pv length ({len(pv)}) must match price length ({n_steps})"
            )

        c_given = battery_charge_schedule is not None
        d_given = battery_discharge_schedule is not None
        if c_given ^ d_given:
            raise ValueError(
                "battery_charge_schedule and battery_discharge_schedule must be "
                "provided together (both arrays or both None)."
            )
        if c_given and len(battery_charge_schedule) != n_steps:
            raise ValueError(
                f"battery_charge_schedule length ({len(battery_charge_schedule)}) "
                f"must match price length ({n_steps})"
            )
        if d_given and len(battery_discharge_schedule) != n_steps:
            raise ValueError(
                f"battery_discharge_schedule length ({len(battery_discharge_schedule)}) "
                f"must match price length ({n_steps})"
            )

    def __init__(self,
                 battery: Battery,
                 price: np.ndarray,
                 pv: np.ndarray = None,
                 battery_charge_schedule: np.ndarray = None,
                 battery_discharge_schedule: np.ndarray = None,
                 degradation_cost: float = 0.0,
                 utility: Utility = None,
                 battery_constraints: BatteryConstraints = None,
                 product: str = '1h',
                 ):
        """Initialise the optimisation with a battery and a day-ahead price series.

        Args:
            battery: Battery dataclass containing capacity, power limits and
                efficiency parameters.
            price: Array of day-ahead prices, one entry per time step [€/MWh].
            pv: Optional PV generation profile [MW], one value per time step.
            battery_charge_schedule: Optional existing battery charge dispatch
                [MW]. Must be provided together with ``battery_discharge_schedule``
                and have the same length as ``price``. Triggers residual-mode
                revenue computation, in which only incremental trades w.r.t. the
                existing schedule are monetised.
            battery_discharge_schedule: Optional existing battery discharge
                dispatch [MW]. See ``battery_charge_schedule``.
            degradation_cost: Cost per MWh of energy discharged [€/MWh], representing
                battery wear. Subtracted from the revenue objective to discourage
                unnecessary cycling. Defaults to 0 (no degradation penalty).
            utility: Optional piecewise-linear utility function applied to revenue.
            battery_constraints: Operational constraints (min/max SoC, terminal SoC,
                max daily cycles). Defaults to BatteryConstraints() if not provided.
            product: Time resolution of the traded product. '1h' for hourly
                (dt=1.0) or '15m' for quarter-hourly (dt=0.25). Defaults to '1h'.
        """
        self._input_validation(
            price=price,
            pv=pv,
            battery_charge_schedule=battery_charge_schedule,
            battery_discharge_schedule=battery_discharge_schedule,
        )
        n_steps = len(price)

        super().__init__(
            battery=battery,
            n_steps=n_steps,
            degradation_cost=degradation_cost,
            battery_constraints=battery_constraints,
            product=product,
        )
        self.price = np.asarray(price, dtype=float)
        self.pv = pv
        self.c_bar = (
            np.asarray(battery_charge_schedule, dtype=float)
            if battery_charge_schedule is not None else None
        )
        self.d_bar = (
            np.asarray(battery_discharge_schedule, dtype=float)
            if battery_discharge_schedule is not None else None
        )
        self.utility = utility

    @property
    def _residual_mode(self) -> bool:
        return self.c_bar is not None

    def update_status(self, soc=None, pv=None, price=None):
        """Update either battery soc or forecasts."""
        if soc is not None:
            self.battery.update_soc(value=soc)

        if price is not None:
            self.price = np.asarray(price, dtype=float)

        if self.pv is not None and pv is not None:
            self.pv = pv

    def _build_variables(self) -> None:
        """Declare core variables plus, in residual mode, the split variables.

        Residual split variables (linearisation of max())
        -------------------------------------------------
        _dc_pos / _dc_neg  : positive/negative part of (battery_charge − c_bar)
        _dd_pos / _dd_neg  : positive/negative part of (battery_discharge − d_bar)
        """
        super()._build_variables()
        if self._residual_mode:
            self._dc_pos = cvxpy.Variable(self.n_steps, nonneg=True, name="dc_pos")
            self._dc_neg = cvxpy.Variable(self.n_steps, nonneg=True, name="dc_neg")
            self._dd_pos = cvxpy.Variable(self.n_steps, nonneg=True, name="dd_pos")
            self._dd_neg = cvxpy.Variable(self.n_steps, nonneg=True, name="dd_neg")

    def _build_power_balance_constraint(self):
        """Build the nodal power balance constraint.

        Total generation (grid imports + battery discharge + optional PV) must
        equal total consumption (grid exports + battery charge) at every step.
        """
        power_generation = self.grid_out + self.battery_discharge
        power_consumption = self.grid_in + self.battery_charge
        if self.pv is not None:
            power_generation += self.pv
        return power_generation == power_consumption

    def _build_constraints(self) -> None:
        """Build the list of physical and operational constraints.

        Constraints
        -----------
        - battery_switch enforces mutual exclusion: charge only when switch=1,
          discharge only when switch=0.
        - Power balance: nodal balance across grid, battery, and PV
          (see :meth:`_build_power_balance_constraint`).
        - State of charge stays within [battery.min_soc * capacity, battery.capacity].
        - Terminal SoC: if battery.soc_end is set, the SoC at the last step must
          be at least that value.
        - Daily cycle constraint limits total throughput to max_daily_cycles.
        - Residual decomposition (when existing schedules are provided):
              battery_charge    − c_bar == _dc_pos − _dc_neg
              battery_discharge − d_bar == _dd_pos − _dd_neg
        """
        self.constraints = self._build_battery_constraints()
        self.constraints.append(self._build_power_balance_constraint())
        if self._residual_mode:
            self.constraints += [
                self.battery_charge - self.c_bar == self._dc_pos - self._dc_neg,
                self.battery_discharge - self.d_bar == self._dd_pos - self._dd_neg,
            ]

    def _build_objective(self) -> cvxpy.Maximize:
        """Construct the maximisation objective.

        Without existing schedules:

            max  Σ (grid_in[t] − grid_out[t]) · price[t]
                 − degradation_cost · Σ battery_discharge[t]

        With existing schedules (residual mode): revenue is earned only on the
        incremental market trade vs. the existing schedule, decomposed into new
        buy / new sell volumes following Oeltz & Pfingsten 2025 (eqs. 7-8):

            c_r = _dc_pos + _dd_neg     (new buy volume)
            d_r = _dd_pos + _dc_neg     (new sell volume)

            max  Σ price[t] · (d_r[t] − c_r[t])
                 − degradation_cost · Σ battery_discharge[t]
        """
        if not self._residual_mode:
            revenue = cvxpy.sum(cvxpy.multiply(self.grid_in - self.grid_out, self.price))
        else:
            c_r = self._dc_pos + self._dd_neg
            d_r = self._dd_pos + self._dc_neg
            revenue = cvxpy.sum(cvxpy.multiply(self.price, d_r - c_r))
        degradation = self.degradation_cost * cvxpy.sum(self.battery_discharge)
        return cvxpy.Maximize(revenue - degradation)

    @property
    def residual_charge(self) -> np.ndarray:
        """New buy volume per step [MW]: c_r = max(Δc, 0) + max(−Δd, 0)."""
        if not self._residual_mode:
            raise RuntimeError("residual_charge is only defined in residual mode.")
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")
        return self._dc_pos.value + self._dd_neg.value

    @property
    def residual_discharge(self) -> np.ndarray:
        """New sell volume per step [MW]: d_r = max(Δd, 0) + max(−Δc, 0)."""
        if not self._residual_mode:
            raise RuntimeError("residual_discharge is only defined in residual mode.")
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")
        return self._dd_pos.value + self._dc_neg.value

    @property
    def pnl(self) -> np.ndarray:
        """Per-step PnL [€].

        In normal mode: price · (grid_in − grid_out).
        In residual mode: price · (residual_discharge − residual_charge), i.e.
        PnL on the incremental trade only.

        Raises:
            RuntimeError: If called before the problem has been solved.
        """
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")
        if not self._residual_mode:
            return self.price * (self.grid_in.value - self.grid_out.value)
        return self.price * (self.residual_discharge - self.residual_charge)

    def get_results(self) -> pl.DataFrame:
        """Extract the optimised schedule as a Polars DataFrame.

        Must be called after :meth:`solve`. Returns one row per time step with
        all decision variable values and derived quantities. When residual mode
        is active, additional ``da_charge``, ``da_discharge``, ``residual_charge``
        and ``residual_discharge`` columns are included.

        Raises:
            RuntimeError: If called before the problem has been solved.
        """
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")

        charge = self.battery_charge.value
        discharge = self.battery_discharge.value
        grid_in = self.grid_in.value
        grid_out = self.grid_out.value
        revenue = self.pnl
        deg_cost = self.degradation_cost * discharge

        data = {
            "step":              np.arange(self.n_steps),
            "price":             self.price,
            "battery_charge":    charge,
            "battery_discharge": discharge,
            "soc":               self.soc.value,
            "grid_in":           grid_in,
            "grid_out":          grid_out,
            "revenue":           revenue,
            "degradation_cost":  deg_cost,
            "net_revenue":       revenue - deg_cost,
        }
        if self._residual_mode:
            data["da_charge"] = self.c_bar
            data["da_discharge"] = self.d_bar
            data["residual_charge"] = self.residual_charge
            data["residual_discharge"] = self.residual_discharge

        return pl.DataFrame(data)

    def __repr__(self) -> str:
        rows = [
            ("Horizon",          f"{self.n_steps} steps"),
            ("Price range",      f"{self.price.min():.2f} – {self.price.max():.2f} €/MWh"),
            ("Degradation cost", f"{self.degradation_cost} €/MWh"),
        ]
        if self._residual_mode:
            rows.append(("Mode", "residual (existing schedule provided)"))

        if self.problem is not None:
            rows.append(("Status", self.problem.status))
            if self.problem.status in ("optimal", "optimal_inaccurate"):
                results = self.get_results()
                rows += [
                    ("Gross revenue",    f"{results['revenue'].sum():.2f} €"),
                    ("Degradation cost", f"{results['degradation_cost'].sum():.2f} €"),
                    ("Net revenue",      f"{results['net_revenue'].sum():.2f} €"),
                    ("Total discharged", f"{results['battery_discharge'].sum():.2f} MWh"),
                ]
                if self._residual_mode:
                    rows += [
                        ("Total res. charge",    f"{results['residual_charge'].sum():.2f} MWh"),
                        ("Total res. discharge", f"{results['residual_discharge'].sum():.2f} MWh"),
                    ]
        else:
            rows.append(("Status", "not solved"))

        return self._repr_table("AuctionOptimisation", rows)

    def plot(self,
             figsize=None,
             return_fig: bool = False,
             soc: bool = True,
             price: bool = True,
             pnl: bool = True,
             pv: bool = True):
        return plot_da_schedule(
            battery_charge=self.battery_charge.value,
            battery_discharge=self.battery_discharge.value,
            soc_values=self.soc.value,
            daprice=self.price,
            pnl_values=self.pnl,
            pv=self.pv,
            figsize=figsize,
            return_fig=return_fig,
            show_soc=soc,
            show_price=price,
            show_pnl=pnl,
            show_pv=pv,
        )
