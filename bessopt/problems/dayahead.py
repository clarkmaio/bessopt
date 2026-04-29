import cvxpy
import numpy as np
import polars as pl

from bessopt.battery import Battery, BatteryConstraints
from bessopt.problems.optimisation import BESSOptimisation
from bessopt.utility import Utility
from bessopt.viz import plot_da_schedule


class DAOptimisation(BESSOptimisation):
    """Day-Ahead market battery optimisation using mixed-integer linear programming.

    Maximises revenue from buying (grid_in) and selling (grid_out) energy on the
    day-ahead market, subject to battery physical constraints (capacity, charge/
    discharge power limits, and round-trip efficiency).

    The charging and discharging directions are enforced via binary switch variables
    so that the battery cannot charge and discharge simultaneously, and the grid
    cannot import and export simultaneously.

    Attributes:
        battery: Physical parameters of the battery asset.
        daprice: Day-ahead price series (one value per optimisation step), in €/MWh.
        n_steps: Number of time steps derived from the price series length.
    """

    def __init__(self,
                 battery: Battery,
                 daprice: np.ndarray,
                 pv: np.ndarray = None,
                 demand: np.ndarray = None,
                 degradation_cost: float = 0.0,
                 utility: Utility = None,
                 battery_constraints: BatteryConstraints = None,
                 product: str = '1h',
                 ):
        """Initialise the optimisation with a battery and a day-ahead price series.

        Args:
            battery: Battery dataclass containing capacity, power limits and
                efficiency parameters.
            daprice: Polars Series of day-ahead prices, one entry per time step.
            pv: Optional PV generation profile [MW], one value per time step.
            demand: Optional demand profile [MW], one value per time step.
            degradation_cost: Cost per MWh of energy discharged [€/MWh], representing
                battery wear. Subtracted from the revenue objective to discourage
                unnecessary cycling. Defaults to 0 (no degradation penalty).
            utility: Optional piecewise-linear utility function applied to revenue.
            battery_constraints: Operational constraints (min/max SoC, terminal SoC,
                max daily cycles). Defaults to BatteryConstraints() if not provided.
            product: Time resolution of the traded product. '1h' for hourly
                (dt=1.0) or '15m' for quarter-hourly (dt=0.25). Defaults to '1h'.
        """
        super().__init__(
            battery=battery,
            n_steps=len(daprice),
            degradation_cost=degradation_cost,
            battery_constraints=battery_constraints,
            product=product,
        )
        self.daprice = daprice
        self.pv = pv
        self.demand = demand
        self.utility = utility

    def update_status(self, soc=None, pv=None, demand=None, daprice=None):
        """Update either battery soc or forecasts."""
        if soc is not None:
            self.battery.update_soc(value=soc)

        if daprice:
            self.daprice = daprice

        if self.pv is not None and pv is not None:
            self.pv = pv

        if self.demand is not None and demand is not None:
            self.demand = demand

    def _build_power_balance_constraint(self):
        """Build the nodal power balance constraint.

        Total generation (grid exports + battery discharge + optional PV) must
        equal total consumption (grid imports + battery charge + optional demand)
        at every time step.

        Returns:
            CVXPY constraint enforcing power balance across all steps.
        """
        power_generation = self.grid_out + self.battery_discharge
        power_consumption = self.grid_in + self.battery_charge
        if self.pv is not None:
            power_generation += self.pv
        if self.demand is not None:
            power_consumption += self.demand
        return power_generation == power_consumption

    def _build_constraints(self) -> None:
        """Build the list of physical and operational constraints.

        Constraints
        -----------
        - battery_switch enforces mutual exclusion: charge only when switch=1,
          discharge only when switch=0.
        - Power balance: nodal balance across grid, battery, PV, and demand
          (see :meth:`_build_power_balance_constraint`).
        - State of charge stays within [battery.min_soc * capacity, battery.capacity].
        - Terminal SoC: if battery.soc_end is set, the SoC at the last step must
          be at least that value, preventing the solver from draining the battery
          for free at the end of the horizon.
        - Daily cycle constraint limits total throughput to max_daily_cycles.
        """
        self.constraints = self._build_battery_constraints()
        self.constraints.append(self._build_power_balance_constraint())

    def _build_objective(self) -> cvxpy.Maximize:
        """Construct the maximisation objective.

        The objective is total DA revenue minus total degradation cost:

            max  Σ (grid_in[t] - grid_out[t]) * price[t]
                 - degradation_cost * Σ battery_discharge[t]

        The degradation term penalises throughput, discouraging the solver from
        cycling the battery more than the price spread justifies.

        Returns:
            CVXPY Maximize expression.
        """
        revenue = cvxpy.sum(cvxpy.multiply(self.grid_in - self.grid_out, self.daprice))
        degradation = self.degradation_cost * cvxpy.sum(self.battery_discharge)
        return cvxpy.Maximize(revenue - degradation)

    @property
    def pnl(self) -> np.ndarray:
        """Per-step PnL [€]: price * (grid_in - grid_out).

        Returns:
            numpy array of length n_steps with the net revenue at each step.

        Raises:
            RuntimeError: If called before the problem has been solved.
        """
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")
        return self.daprice * (self.grid_in.value - self.grid_out.value)

    def get_results(self) -> pl.DataFrame:
        """Extract the optimised schedule as a Polars DataFrame.

        Must be called after :meth:`solve`. Returns one row per time step with
        all decision variable values and derived quantities needed for analysis
        or reporting.

        Returns:
            DataFrame with columns:
                - ``step``               : integer time-step index.
                - ``price``              : day-ahead price [€/MWh].
                - ``battery_charge``     : power charged into the battery [MW].
                - ``battery_discharge``  : power discharged from the battery [MW].
                - ``soc``                : state of charge at end of step [MWh].
                - ``grid_in``            : power bought from the grid [MW].
                - ``grid_out``           : power sold to the grid [MW].
                - ``revenue``            : gross revenue per step [€].
                - ``degradation_cost``   : degradation cost per step [€].
                - ``net_revenue``        : net revenue per step [€].

        Raises:
            RuntimeError: If called before the problem has been solved.
        """
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")

        charge = self.battery_charge.value
        discharge = self.battery_discharge.value
        grid_in = self.grid_in.value
        grid_out = self.grid_out.value
        revenue = (grid_in - grid_out) * self.daprice
        deg_cost = self.degradation_cost * discharge

        return pl.DataFrame({
            "step":             np.arange(self.n_steps),
            "price":            self.daprice,
            "battery_charge":   charge,
            "battery_discharge": discharge,
            "soc":              self.soc.value,
            "grid_in":          grid_in,
            "grid_out":         grid_out,
            "revenue":          revenue,
            "degradation_cost": deg_cost,
            "net_revenue":      revenue - deg_cost,
        })

    def __repr__(self) -> str:
        rows = [
            ("Horizon",          f"{self.n_steps} steps"),
            ("Price range",      f"{self.daprice.min():.2f} – {self.daprice.max():.2f} €/MWh"),
            ("Degradation cost", f"{self.degradation_cost} €/MWh"),
        ]

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
        else:
            rows.append(("Status", "not solved"))

        return self._repr_table("DAOptimisation", rows)

    def plot(self,
             figsize=None,
             return_fig: bool = False,
             soc: bool = True,
             price: bool = True,
             pnl: bool = True,
             pv: bool = True,
             demand: bool = True):
        return plot_da_schedule(
            battery_charge=self.battery_charge.value,
            battery_discharge=self.battery_discharge.value,
            soc_values=self.soc.value,
            daprice=self.daprice,
            pnl_values=self.pnl,
            pv=self.pv,
            demand=self.demand,
            figsize=figsize,
            return_fig=return_fig,
            show_soc=soc,
            show_price=price,
            show_pnl=pnl,
            show_pv=pv,
            show_demand=demand,
        )
