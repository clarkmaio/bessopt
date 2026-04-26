import polars as pl
import numpy as np
import cvxpy
import matplotlib.pyplot as plt

from src.battery import Battery, BatteryConstraints


M = 1e8


class IntradayOptimisation:
    """Intraday redispatch of a battery position using imbalance bid/ask prices.

    Implements the rolling-intrinsic residual-trade methodology (Oeltz & Pfingsten
    2025, eqs. 7-10).  Given an existing schedule from the DA market, the class
    re-optimises the full battery dispatch over the same horizon and computes
    revenue only on the *residual* trades – i.e. the incremental volume needed to
    move from the current (DA) position to the new (intraday) position:

        c_r_i = max(c_i - c̄_i, 0) + max(d̄_i - d_i, 0)   [new buy volume]  (7)
        d_r_i = max(d_i - d̄_i, 0) + max(c̄_i - c_i, 0)   [new sell volume] (8)

    where c̄_i, d̄_i are the existing (DA) charge/discharge positions.

    Objective (eq. 10):
        max  Δt · Σ_i  p_long_i · d_r_i  −  p_short_i · c_r_i
             −  degradation_cost · Σ_i d_i

    The max() terms are linearised via auxiliary split variables so the problem
    remains a MILP (binary switch for mutual exclusion) solvable by CVXPY.

    Attributes:
        battery:         Physical parameters of the battery.
        price_long:      Imbalance long price series [€/MWh] – revenue received
                         when selling (discharging) residuals.
        price_short:     Imbalance short price series [€/MWh] – cost paid when
                         buying (charging) residuals.
        da_schedule:     Full DA dispatch DataFrame from DAOptimisation.get_results().
        degradation_cost: Cost per MWh discharged [€/MWh]. Defaults to 0.
        n_steps:         Number of time steps (derived from price series length).
    """

    def __init__(
        self,
        battery: Battery,
        price_long: np.ndarray,
        price_short: np.ndarray,
        da_schedule: pl.DataFrame,
        degradation_cost: float = 0.0,
        battery_constraints: BatteryConstraints = None,
        product: str = '1h',
    ):
        """
        Args:
            battery:             Battery dataclass.
            price_long:          Imbalance long (bid) prices, one per time step [€/MWh].
            price_short:         Imbalance short (ask) prices, one per time step [€/MWh].
            da_schedule:         Output of DAOptimisation.get_results(); must contain
                                 columns ``battery_charge``, ``battery_discharge``,
                                 ``grid_in``, ``grid_out``.
            degradation_cost:    Throughput penalty [€/MWh discharged]. Defaults to 0.
            battery_constraints: Operational constraints (min/max SoC, terminal SoC,
                                 max daily cycles). Defaults to BatteryConstraints() if not provided.
            product:             Time resolution of the traded product. '1h' for hourly
                                 (dt=1.0) or '15m' for quarter-hourly (dt=0.25). Defaults to '1h'.
        """
        if product not in ('1h', '15m'):
            raise ValueError(f"product must be '1h' or '15m', got '{product}'")

        self.battery = battery
        self.price_long = np.asarray(price_long, dtype=float)
        self.price_short = np.asarray(price_short, dtype=float)
        self.degradation_cost = degradation_cost
        self.battery_constraints = battery_constraints or BatteryConstraints()
        self.product = product
        self.dt = 1.0 if product == '1h' else 0.25

        # Current (DA) position – the "bar" quantities in the paper
        self.c_bar = da_schedule["battery_charge"].to_numpy()
        self.d_bar = da_schedule["battery_discharge"].to_numpy()
        self.grid_in_bar = da_schedule["grid_in"].to_numpy()
        self.grid_out_bar = da_schedule["grid_out"].to_numpy()

        self.n_steps = len(self.price_long)
        self.constraints = []
        self.problem = None

    # ------------------------------------------------------------------
    # Build helpers
    # ------------------------------------------------------------------

    def _build_variables(self) -> None:
        """Declare all CVXPY decision variables.

        Full new dispatch
        -----------------
        battery_charge     : non-negative, shape (n_steps,)  [MW]
        battery_discharge  : non-negative, shape (n_steps,)  [MW]
        battery_switch     : binary,       shape (n_steps,)
        grid_in            : non-negative, shape (n_steps,)  [MW]
        grid_out           : non-negative, shape (n_steps,)  [MW]
        soc                : expression,   shape (n_steps,)  [MWh]

        Residual split variables (linearisation of max())
        -------------------------------------------------
        _dc_pos / _dc_neg  : positive/negative part of (c_i − c̄_i)
        _dd_pos / _dd_neg  : positive/negative part of (d_i − d̄_i)
        """
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

        # Auxiliary split variables for the residual linearisation
        self._dc_pos = cvxpy.Variable(self.n_steps, nonneg=True, name="dc_pos")
        self._dc_neg = cvxpy.Variable(self.n_steps, nonneg=True, name="dc_neg")
        self._dd_pos = cvxpy.Variable(self.n_steps, nonneg=True, name="dd_pos")
        self._dd_neg = cvxpy.Variable(self.n_steps, nonneg=True, name="dd_neg")

    def _build_constraints(self) -> None:
        """Build all CVXPY constraints.

        Physical battery constraints (identical to DAOptimisation)
        -----------------------------------------------------------
        - Power limits with mutual-exclusion switch.
        - Power balance: grid_in/out equal charge/discharge.
        - SoC within [min_soc * capacity, capacity].
        - Optional terminal SoC.
        - Daily cycle throughput limit.

        Residual decomposition constraints
        -----------------------------------
        Decompose delta_c = c - c̄ and delta_d = d - d̄ into their positive
        and negative parts so that the max() in eqs. (7)-(8) can be expressed
        linearly:

            delta_c  =  dc_pos - dc_neg,   dc_pos, dc_neg ≥ 0
            delta_d  =  dd_pos - dd_neg,   dd_pos, dd_neg ≥ 0
        """
        self.constraints = [
            # ── Battery power limits & mutual exclusion ──────────────────────
            self.battery_charge <= self.battery_switch * self.battery.max_charge_power * self.dt,
            self.battery_discharge <= (1 - self.battery_switch) * self.battery.max_discharge_power * self.dt,

            # ── Power balance (no PV / demand in intraday context) ────────────
            self.grid_in == self.battery_charge,
            self.grid_out == self.battery_discharge,

            # ── State-of-charge bounds ────────────────────────────────────────
            self.soc >= self.battery_constraints.min_soc * self.battery.capacity,
            self.soc <= self.battery_constraints.max_soc * self.battery.capacity,

            # ── Daily cycle constraint ────────────────────────────────────────
            cvxpy.sum(self.battery_discharge) / self.battery.capacity <= self.battery_constraints.max_daily_cycles,

            # ── Residual split constraints (eqs. 7-8 linearisation) ──────────
            self.battery_charge - self.c_bar == self._dc_pos - self._dc_neg,
            self.battery_discharge - self.d_bar == self._dd_pos - self._dd_neg,
        ]

        if self.battery_constraints.soc_end is not None:
            self.constraints.append(self.soc[-1] >= self.battery_constraints.soc_end)

    def _build_objective(self) -> cvxpy.Maximize:
        """Construct the maximisation objective (eq. 10).

        Revenue is earned/paid only on residual trades:
            max  Σ_i  price_long_i · d_r_i  −  price_short_i · c_r_i
                 −  degradation_cost · Σ_i d_i

        where:
            c_r_i  =  dc_pos_i + dd_neg_i   (new buy trades, eq. 7)
            d_r_i  =  dd_pos_i + dc_neg_i   (new sell trades, eq. 8)
        """
        c_r = self._dc_pos + self._dd_neg   # eq. (7)
        d_r = self._dd_pos + self._dc_neg   # eq. (8)

        revenue = cvxpy.sum(
            cvxpy.multiply(self.price_long, d_r)
            - cvxpy.multiply(self.price_short, c_r)
        )
        degradation = self.degradation_cost * cvxpy.sum(self.battery_discharge)
        return cvxpy.Maximize(revenue - degradation)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def solve(self) -> None:
        """Build and solve the intraday redispatch MILP."""
        self._build_variables()
        self._build_constraints()
        objective = self._build_objective()
        self.problem = cvxpy.Problem(objective, self.constraints)
        self.problem.solve()

    @property
    def residual_charge(self) -> np.ndarray:
        """New buy volume per step [MW]: c_r_i = max(Δc, 0) + max(-Δd, 0)."""
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")
        return self._dc_pos.value + self._dd_neg.value

    @property
    def residual_discharge(self) -> np.ndarray:
        """New sell volume per step [MW]: d_r_i = max(Δd, 0) + max(-Δc, 0)."""
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")
        return self._dd_pos.value + self._dc_neg.value

    @property
    def pnl(self) -> np.ndarray:
        """Per-step PnL on residual trades [€]."""
        return self.price_long * self.residual_discharge - self.price_short * self.residual_charge

    def get_results(self) -> pl.DataFrame:
        """Extract the optimised intraday schedule as a Polars DataFrame.

        Returns one row per time step with the full new dispatch, the DA
        position, the residuals actually traded in intraday, and per-step PnL.

        Returns:
            DataFrame with columns:
                - ``step``                : integer time-step index.
                - ``price_long``          : imbalance long (bid) price [€/MWh].
                - ``price_short``         : imbalance short (ask) price [€/MWh].
                - ``battery_charge``      : new total charge [MW].
                - ``battery_discharge``   : new total discharge [MW].
                - ``soc``                 : state of charge at end of step [MWh].
                - ``grid_in``             : new grid import [MW].
                - ``grid_out``            : new grid export [MW].
                - ``da_charge``           : original DA charge position [MW].
                - ``da_discharge``        : original DA discharge position [MW].
                - ``residual_charge``     : new buy volume traded intraday [MW].
                - ``residual_discharge``  : new sell volume traded intraday [MW].
                - ``revenue``             : gross revenue on residuals [€].
                - ``degradation_cost``    : degradation cost per step [€].
                - ``net_revenue``         : net revenue per step [€].

        Raises:
            RuntimeError: If called before solve().
        """
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")

        rev = self.pnl
        deg = self.degradation_cost * self.battery_discharge.value

        return pl.DataFrame({
            "step":               np.arange(self.n_steps),
            "price_long":         self.price_long,
            "price_short":        self.price_short,
            "battery_charge":     self.battery_charge.value,
            "battery_discharge":  self.battery_discharge.value,
            "soc":                self.soc.value,
            "grid_in":            self.grid_in.value,
            "grid_out":           self.grid_out.value,
            "da_charge":          self.c_bar,
            "da_discharge":       self.d_bar,
            "residual_charge":    self.residual_charge,
            "residual_discharge": self.residual_discharge,
            "revenue":            rev,
            "degradation_cost":   deg,
            "net_revenue":        rev - deg,
        })

    def __repr__(self) -> str:
        rows = [
            ("Horizon",          f"{self.n_steps} steps"),
            ("Long price range",  f"{self.price_long.min():.2f} – {self.price_long.max():.2f} €/MWh"),
            ("Short price range", f"{self.price_short.min():.2f} – {self.price_short.max():.2f} €/MWh"),
            ("Degradation cost",  f"{self.degradation_cost} €/MWh"),
        ]

        if self.problem is not None:
            rows.append(("Status", self.problem.status))
            if self.problem.status in ("optimal", "optimal_inaccurate"):
                results = self.get_results()
                rows += [
                    ("Gross revenue",       f"{results['revenue'].sum():.2f} €"),
                    ("Degradation cost",    f"{results['degradation_cost'].sum():.2f} €"),
                    ("Net revenue",         f"{results['net_revenue'].sum():.2f} €"),
                    ("Total res. charge",   f"{results['residual_charge'].sum():.2f} MWh"),
                    ("Total res. discharge",f"{results['residual_discharge'].sum():.2f} MWh"),
                ]
        else:
            rows.append(("Status", "not solved"))

        col_w = max(len(label) for label, _ in rows)
        val_w = max(len(value) for _, value in rows)
        sep = f"+{'-' * (col_w + 2)}+{'-' * (val_w + 2)}+"
        fmt = f"| {{:<{col_w}}} | {{:<{val_w}}} |"

        lines = [
            sep,
            fmt.format("IntradayOptimisation", ""),
            sep,
            *(fmt.format(label, value) for label, value in rows),
            sep,
        ]
        return "\n".join(lines)

    def plot(self, figsize=None, return_fig: bool = False):
        """Visualise the intraday redispatch schedule.

        Shows four panels:
          1. DA vs new charge/discharge bars (green=charge, red=discharge).
          2. State of charge (new vs DA).
          3. Imbalance long/short prices.
          4. Per-step PnL on residual trades.

        Args:
            figsize:    Optional (width, height) tuple.
            return_fig: If True, return the Figure instead of calling plt.show().
        """
        if self.problem is None or self.problem.status not in ("optimal", "optimal_inaccurate"):
            raise RuntimeError("No optimal solution available. Call solve() first.")

        results = self.get_results()
        x = np.arange(self.n_steps)

        fig, axes = plt.subplots(4, 1, figsize=figsize)

        # Panel 1: charge/discharge comparison
        width = 0.4
        axes[0].bar(x - width / 2, results["battery_charge"].to_numpy(),
                    width=width, color="#2ecc71", label="charge (new)")
        axes[0].bar(x - width / 2, -results["battery_discharge"].to_numpy(),
                    width=width, color="#e74c3c", label="discharge (new)")
        axes[0].bar(x + width / 2, self.c_bar, width=width,
                    color="#2ecc71", alpha=0.3, label="charge (DA)")
        axes[0].bar(x + width / 2, -self.d_bar, width=width,
                    color="#e74c3c", alpha=0.3, label="discharge (DA)")
        axes[0].axhline(y=0, color="black", linewidth=0.8)
        axes[0].set_ylabel("MW")
        axes[0].legend(fontsize=7, ncol=2)

        # Panel 2: SoC
        axes[1].fill_between(x, results["soc"].to_numpy(), alpha=0.5,
                             color="#3498db", step="post", label="SoC (new)")
        axes[1].step(x, results["soc"].to_numpy(), color="#3498db",
                     linewidth=0.8, where="post")
        axes[1].set_ylabel("MWh")

        # Panel 3: imbalance prices
        axes[2].step(x, self.price_long, color="#27ae60", where="post", label="long")
        axes[2].step(x, self.price_short, color="#e74c3c", where="post",
                     linestyle="--", label="short")
        axes[2].set_ylabel("€/MWh")
        axes[2].legend(fontsize=7)

        # Panel 4: PnL on residuals
        pnl = self.pnl
        axes[3].bar(x, pnl, color=np.where(pnl >= 0, "#2ecc71", "#e74c3c"))
        axes[3].axhline(y=0, color="black", linewidth=0.8)
        axes[3].set_ylabel("€")
        ax2 = axes[3].twinx()
        ax2.plot(x, np.cumsum(pnl), color='#2980b9', linewidth=1.5)
        ax2.set_ylabel('Cumulative €', color='#2980b9')
        ax2.tick_params(axis='y', labelcolor='#2980b9')

        for ax in axes[:-1]:
            ax.tick_params(labelbottom=False)

        fig.suptitle("Intraday redispatch schedule", fontweight="bold")
        plt.tight_layout()

        if return_fig:
            return fig
        plt.show()
