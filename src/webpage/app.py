import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib
matplotlib.use('Agg')  # non-interactive backend — must be before pyplot is imported

from datetime import date, timedelta
from urllib.parse import urlencode
import base64, io
import matplotlib.pyplot as plt

from fasthtml.common import *

from src.battery import Battery
from src.da_opt import DAOptimisation
from src.data import load_entsoe_dayahead_prices

# --------------------------------------------------------------------------- #
# Date range helpers
# --------------------------------------------------------------------------- #
TODAY    = date.today()
DATE_MIN = TODAY - timedelta(days=90)
DATE_MAX = TODAY - timedelta(days=1)
N_DAYS   = 90


COUNTRY_CODE = 'DK_1'


def idx_to_date(idx: int) -> date:
    return DATE_MIN + timedelta(days=int(idx))


def fig_to_img(fig) -> Img:
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=120)
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode()
    plt.close(fig)
    return Img(src=f"data:image/png;base64,{b64}", style="width:100%;")


# --------------------------------------------------------------------------- #
# App
# --------------------------------------------------------------------------- #
app, rt = fast_app(
    hdrs=[
        Style("""
            *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

            body {
                font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
                background: #fff;
                color: #000;
                font-size: 16px;
                line-height: 1.6;
            }

            /* ---- centered column (used inside nav/slider only) ---- */
            .container {
                max-width: 960px;
                margin: 0 auto;
                padding: 0 32px;
            }

            /* ---- navbar ---- */
            nav {
                border-bottom: 1px solid #e8e8e8;
                position: sticky; top: 0;
                background: #fff;
                z-index: 100;
            }
            nav .container {
                display: flex;
                align-items: center;
                justify-content: space-between;
                height: 54px;
            }
            .brand {
                font-size: 1rem;
                font-weight: 700;
                letter-spacing: 0.5px;
                color: #000;
                text-decoration: none;
            }
            .nav-links { display: flex; gap: 0; }
            .nav-tab {
                font-size: 0.875rem;
                color: #666;
                text-decoration: none;
                padding: 6px 14px;
                border-radius: 4px;
                transition: color 0.15s, background 0.15s;
            }
            .nav-tab:hover { color: #000; background: #f5f5f5; }
            .nav-tab.active { color: #000; font-weight: 600; background: #f0f0f0; }

            /* ---- slider bar ---- */
            .slider-bar {
                border-bottom: 1px solid #e8e8e8;
            }
            .slider-bar .container {
                padding-top: 14px;
                padding-bottom: 14px;
                display: flex;
                align-items: center;
                gap: 16px;
            }
            .slider-bar label {
                font-size: 0.8rem;
                font-weight: 600;
                color: #888;
                text-transform: uppercase;
                letter-spacing: 0.5px;
                white-space: nowrap;
            }
            .slider-bar input[type=range] {
                flex: 1;
                height: 2px;
                accent-color: #000;
                cursor: pointer;
            }
            #date-display {
                font-size: 0.875rem;
                font-weight: 600;
                color: #000;
                font-variant-numeric: tabular-nums;
                min-width: 94px;
                text-align: right;
            }

            /* ---- page layout ---- */
            .layout {
                display: flex;
                align-items: flex-start;
            }

            /* ---- sidebar toggle button ---- */
            .sidebar-toggle {
                position: fixed;
                top: 50%;
                transform: translateY(-50%);
                left: 280px;
                z-index: 50;
                width: 18px;
                height: 36px;
                background: #fff;
                border: 1px solid #e8e8e8;
                border-left: none;
                border-radius: 0 4px 4px 0;
                cursor: pointer;
                display: flex;
                align-items: center;
                justify-content: center;
                font-size: 0.6rem;
                color: #aaa;
                transition: left 0.25s, color 0.15s;
            }
            .sidebar-toggle:hover { color: #000; }
            .sidebar-hidden .sidebar-toggle { left: 0; }

            /* ---- sidebar ---- */
            .sidebar {
                width: 280px;
                min-width: 280px;
                border-right: 1px solid #e8e8e8;
                padding: 32px 20px;
                position: sticky;
                top: 107px;
                height: calc(100vh - 107px);
                overflow-y: auto;
                transition: width 0.25s, min-width 0.25s, padding 0.25s, opacity 0.2s;
            }
            .sidebar-hidden .sidebar {
                width: 0;
                min-width: 0;
                padding: 0;
                opacity: 0;
                overflow: hidden;
            }
            .sidebar-title {
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #999;
                margin-bottom: 24px;
            }
            .sidebar-field {
                margin-bottom: 20px;
            }
            .sidebar-field label {
                display: block;
                font-size: 0.75rem;
                font-weight: 600;
                color: #555;
                margin-bottom: 5px;
                text-transform: uppercase;
                letter-spacing: 0.4px;
            }
            .sidebar-field input[type=number] {
                width: 100%;
                padding: 6px 9px;
                border: 1px solid #e0e0e0;
                border-radius: 3px;
                font-size: 0.875rem;
                color: #000;
                outline: none;
                -moz-appearance: textfield;
            }
            .sidebar-field input[type=number]::-webkit-inner-spin-button,
            .sidebar-field input[type=number]::-webkit-outer-spin-button { opacity: 1; }
            .sidebar-field input[type=number]:focus { border-color: #000; }
            .sidebar-field select {
                width: 100%;
                padding: 6px 9px;
                border: 1px solid #e0e0e0;
                border-radius: 3px;
                font-size: 0.875rem;
                color: #000;
                background: #fff;
                outline: none;
                cursor: pointer;
            }
            .sidebar-field select:focus { border-color: #000; }
            .sidebar-field input[type=range] {
                width: 100%;
                height: 2px;
                accent-color: #000;
                cursor: pointer;
                margin-top: 6px;
            }

            /* ---- main content ---- */
            .main-content {
                flex: 1;
                padding: 48px 40px;
                min-width: 0;
                display: flex;
                flex-direction: column;
                align-items: center;
            }
            .main-content > * {
                width: 100%;
                max-width: 720px;
            }

            /* ---- spinner ---- */
            .spinner-wrap {
                display: flex;
                flex-direction: column;
                align-items: center;
                padding: 80px 0;
                gap: 16px;
            }
            .spinner {
                width: 36px;
                height: 36px;
                border: 3px solid #e8e8e8;
                border-top-color: #000;
                border-radius: 50%;
                animation: spin 0.7s linear infinite;
            }
            @keyframes spin { to { transform: rotate(360deg); } }
            .spinner-label {
                font-size: 0.78rem;
                color: #999;
                text-transform: uppercase;
                letter-spacing: 0.6px;
            }

            /* ---- section heading ---- */
            .section-label {
                font-size: 0.75rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 1px;
                color: #999;
                margin-bottom: 20px;
            }

            /* ---- summary stats ---- */
            .stats-grid {
                display: grid;
                grid-template-columns: repeat(3, 1fr);
                gap: 1px;
                background: #e8e8e8;
                border: 1px solid #e8e8e8;
                margin-bottom: 48px;
            }
            .stat-cell {
                background: #fff;
                padding: 28px 24px;
            }
            .stat-cell .value {
                font-size: 1.8rem;
                font-weight: 700;
                letter-spacing: -0.5px;
                line-height: 1;
            }
            .stat-cell .label {
                font-size: 0.78rem;
                color: #888;
                margin-top: 6px;
                text-transform: uppercase;
                letter-spacing: 0.5px;
            }

            /* ---- table ---- */
            .table-wrap { margin-bottom: 48px; }
            table { width: 100%; border-collapse: collapse; font-size: 0.875rem; }
            thead th {
                text-align: left;
                font-size: 0.72rem;
                font-weight: 600;
                text-transform: uppercase;
                letter-spacing: 0.6px;
                color: #999;
                padding: 10px 14px;
                border-bottom: 1px solid #e8e8e8;
            }
            tbody td { padding: 10px 14px; border-bottom: 1px solid #f0f0f0; color: #111; }
            tbody tr:last-child td { border-bottom: none; }
            tbody tr:hover td { background: #fafafa; }

            /* ---- badges ---- */
            .badge {
                display: inline-block;
                padding: 2px 8px;
                border-radius: 3px;
                font-size: 0.72rem;
                font-weight: 600;
                letter-spacing: 0.3px;
            }
            .badge-charge    { background: #f0f0f0; color: #333; }
            .badge-discharge { background: #000; color: #fff; }
            .badge-idle      { color: #aaa; }

            /* ---- revenue colors ---- */
            .pos { color: #000; }
            .neg { color: #999; }
        """),
        Script("""
            function updateDateLabel(input) {
                const base = new Date('""" + DATE_MIN.isoformat() + """T00:00:00');
                base.setDate(base.getDate() + parseInt(input.value));
                document.getElementById('date-display').textContent = base.toISOString().slice(0, 10);
            }
            function updateParam(name, value) {
                const url = new URL(window.location.href);
                url.searchParams.set(name, value);
                window.location.href = url.toString();
            }
            function toggleSidebar() {
                const layout = document.getElementById('layout');
                const btn    = document.getElementById('sidebar-toggle');
                const hidden = layout.classList.toggle('sidebar-hidden');
                btn.textContent = hidden ? '›' : '‹';
            }
        """),
    ]
)

# --------------------------------------------------------------------------- #
# Sample data (replace with real DAOptimisation.get_results())
# --------------------------------------------------------------------------- #
import random, math

def sample_schedule(d: date, market: str):
    rng  = random.Random(d.toordinal() + hash(market))
    base = 48 + rng.uniform(-12, 18)
    rows = []
    for h in range(24):
        price = base + 22 * math.sin(math.pi * h / 12) + rng.uniform(-4, 4)
        if 2 <= h <= 4:
            action, power = "Charge", rng.uniform(3, 5)
            revenue = -price * power
        elif 8 <= h <= 10:
            action, power = "Discharge", rng.uniform(3, 5)
            revenue = price * power
        else:
            action, power, revenue = "Idle", 0.0, 0.0
        rows.append({
            "hour": f"{h:02d}:00",
            "action": action,
            "power_mw": round(power, 2),
            "price": round(price, 1),
            "revenue": round(revenue, 1),
        })
    return rows


# --------------------------------------------------------------------------- #
# Components
# --------------------------------------------------------------------------- #
def navbar(active: str, params: dict):
    qs = urlencode(params)
    return Nav(
        Div(
            A("BESSopt", href="/", cls="brand"),
            Div(
                A("Dayahead", href=f"/dayahead?{qs}",
                  cls="nav-tab" + (" active" if active == "dayahead" else "")),
                A("Intraday", href=f"/intraday?{qs}",
                  cls="nav-tab" + (" active" if active == "intraday" else "")),
                cls="nav-links",
            ),
            cls="container",
        )
    )


def date_slider(active: str, params: dict):
    idx      = params["idx"]
    selected = idx_to_date(idx)
    return Div(
        Div(
            Label("Date", fr="date-slider"),
            Input(
                type="range",
                id="date-slider",
                min="0", max=str(N_DAYS - 1),
                value=str(idx),
                oninput="updateDateLabel(this); updateParam('idx', this.value)",
            ),
            Span(selected.isoformat(), id="date-display"),
            cls="container",
        ),
        cls="slider-bar",
    )


COUNTRY_CODES = ["DE_LU", "DK_1", "FR"]


def sidebar(params: dict):
    def field(label_text, name, value, step):
        return Div(
            Label(label_text),
            Input(
                type="number", value=str(value), min="0", step=str(step),
                onchange=f"updateParam('{name}', this.value)",
            ),
            cls="sidebar-field",
        )

    def pct_slider(label_text, name, value, label_id):
        return Div(
            Label(f"{label_text} — {value}%", id=label_id),
            Input(
                type="range", min="0", max="100", step="10",
                value=str(value),
                oninput=f"document.getElementById('{label_id}').textContent = '{label_text} — ' + this.value + '%';",
                onchange=f"updateParam('{name}', this.value)",
            ),
            cls="sidebar-field",
        )

    country_options = [
        Option(c, value=c, selected=(c == params.get("country", COUNTRY_CODE)))
        for c in COUNTRY_CODES
    ]

    return Aside(
        P("Market", cls="sidebar-title"),
        Div(
            Label("Country"),
            Select(
                *country_options,
                onchange="updateParam('country', this.value)",
            ),
            cls="sidebar-field",
        ),
        P("Battery", cls="sidebar-title", style="margin-top:24px;"),
        field("Capacity (MWh)",     "capacity",  params["capacity"],  0.1),
        Div(
            Label(f'Starting SoC (MWh) — {params["soc"]:.2f}', id="soc-label"),
            Input(
                type="range",
                min="0", max=str(params["capacity"]), step="0.05",
                value=str(params["soc"]),
                oninput="document.getElementById('soc-label').textContent = 'Starting SoC (MWh) — ' + parseFloat(this.value).toFixed(2);",
                onchange="updateParam('soc', this.value)",
            ),
            cls="sidebar-field",
        ),
        Div(
            Label(f'Max Power (MW) — {params["max_power"]:.2f}', id="max-power-label"),
            Input(
                type="range",
                min="0.1", max=str(params["capacity"]), step="0.05",
                value=str(params["max_power"]),
                oninput="document.getElementById('max-power-label').textContent = 'Max Power (MW) — ' + parseFloat(this.value).toFixed(2);",
                onchange="updateParam('max_power', this.value)",
            ),
            cls="sidebar-field",
        ),
        pct_slider("Charge Efficiency",    "charge_eff",    params["charge_eff"],    "charge-eff-label"),
        pct_slider("Discharge Efficiency", "discharge_eff", params["discharge_eff"], "discharge-eff-label"),
        cls="sidebar",
    )


def action_badge(action: str):
    cls = {"Charge": "badge badge-charge", "Discharge": "badge badge-discharge"}.get(action, "badge badge-idle")
    return Span(action, cls=cls)


def page_layout(params: dict, *content):
    """Sidebar + toggle + main content wrapper shared by all pages."""
    return Div(
        sidebar(params),
        Button("‹", id="sidebar-toggle", cls="sidebar-toggle", onclick="toggleSidebar()"),
        Main(*content, cls="main-content"),
        id="layout",
        cls="layout",
    )


def intraday_content(rows):
    gross = sum(r["revenue"] for r in rows if r["revenue"] > 0)
    cost  = abs(sum(r["revenue"] for r in rows if r["revenue"] < 0))
    net   = gross - cost

    stats = Div(
        Div(Div(f"€{gross:,.0f}", cls="value"), Div("Gross revenue", cls="label"), cls="stat-cell"),
        Div(Div(f"€{cost:,.0f}",  cls="value"), Div("Charge cost",   cls="label"), cls="stat-cell"),
        Div(Div(f"€{net:,.0f}",   cls="value"), Div("Net revenue",   cls="label"), cls="stat-cell"),
        cls="stats-grid",
    )

    trows = [
        Tr(
            Td(r["hour"]),
            Td(action_badge(r["action"])),
            Td(f'{r["power_mw"]:.2f}'),
            Td(f'{r["price"]:.1f}'),
            Td(f'€ {r["revenue"]:+.0f}', cls="pos" if r["revenue"] >= 0 else "neg"),
        )
        for r in rows
    ]

    table = Div(
        Table(
            Thead(Tr(Th("Hour"), Th("Action"), Th("Power (MW)"), Th("Price (€/MWh)"), Th("Revenue (€)"))),
            Tbody(*trows),
        ),
        cls="table-wrap",
    )

    return P("Summary", cls="section-label"), stats, P("Schedule", cls="section-label"), table


# --------------------------------------------------------------------------- #
# Routes
# --------------------------------------------------------------------------- #
@rt("/")
def index(idx: int = N_DAYS - 1):
    return RedirectResponse(f"/dayahead?idx={idx}")


@rt("/dayahead")
def dayahead(idx: int = N_DAYS - 1, capacity: float = 1.0, soc: float = 1.0, max_power: float = 0.5, country: str = COUNTRY_CODE, charge_eff: int = 90, discharge_eff: int = 90):
    idx    = max(0, min(idx, N_DAYS - 1))
    params = dict(idx=idx, capacity=capacity, soc=soc, max_power=max_power, country=country, charge_eff=charge_eff, discharge_eff=discharge_eff)
    qs     = urlencode(params)

    spinner = Div(
        Div(cls="spinner"),
        P("Optimising…", cls="spinner-label"),
        hx_get=f"/dayahead/plot?{qs}",
        hx_trigger="load",
        hx_swap="outerHTML",
        cls="spinner-wrap",
    )

    return Title("BESSopt"), navbar("dayahead", params), date_slider("dayahead", params), page_layout(params, spinner)


@rt("/dayahead/plot")
def dayahead_plot(idx: int = N_DAYS - 1, capacity: float = 1.0, soc: float = 1.0, max_power: float = 0.5, country: str = COUNTRY_CODE, charge_eff: int = 90, discharge_eff: int = 90):
    try:
        df     = load_entsoe_dayahead_prices(country, idx_to_date(idx))
        prices = df['daprice'].to_numpy()

        battery = Battery(
            capacity=capacity,
            max_charge_power=max_power,
            max_discharge_power=max_power,
            soc=soc,
            max_daily_cycles=2,
            charge_efficiency=charge_eff / 100,
            discharge_efficiency=discharge_eff / 100,
        )

        opt = DAOptimisation(battery=battery, daprice=prices)
        opt.solve()

        status = opt.problem.status
        if status not in ("optimal", "optimal_inaccurate"):
            return P(f"Solver status: {status}", style="color:#c0392b; padding:16px; border:1px solid #f5c6cb; border-radius:4px;")

        return fig_to_img(opt.plot(return_fig=True, figsize=(10, 8)))
    except Exception as e:
        return P(f"Error: {e}", style="color:#c0392b; padding:16px; border:1px solid #f5c6cb; border-radius:4px;")


@rt("/intraday")
def intraday(idx: int = N_DAYS - 1, capacity: float = 1.0, soc: float = 1.0, max_power: float = 0.5, country: str = COUNTRY_CODE, charge_eff: int = 90, discharge_eff: int = 90):
    idx    = max(0, min(idx, N_DAYS - 1))
    params = dict(idx=idx, capacity=capacity, soc=soc, max_power=max_power, country=country, charge_eff=charge_eff, discharge_eff=discharge_eff)
    return Title("BESSopt"), navbar("intraday", params), date_slider("intraday", params), page_layout(params, P("Work in progress", style="color:#999; font-style:italic;"))


if __name__ == "__main__":
    serve()
