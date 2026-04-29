import marimo

__generated_with = "0.20.4"
app = marimo.App(width="columns")


@app.cell(column=0)
def _():
    import os
    import sys

    sys.path.append('/home/clarkmaio/workspace/bessopt/')
    return


@app.cell
def _():
    import marimo as mo
    from bessopt.battery import Battery
    from bessopt.problems.dayahead import DAOptimisation
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    from datetime import datetime

    from bessopt.data import load_entsoe_dayahead_prices
    from bessopt.data import load_entsoe_solar_forecast

    vdate = datetime(2026, 4, 12)
    country_code = 'FR'
    return (
        Battery,
        DAOptimisation,
        load_entsoe_dayahead_prices,
        load_entsoe_solar_forecast,
        mo,
        vdate,
    )


@app.cell
def _(load_entsoe_dayahead_prices, vdate):
    pricedf = load_entsoe_dayahead_prices(valuedate=vdate, country_code='DE_LU')
    return (pricedf,)


@app.cell
def _(load_entsoe_solar_forecast, vdate):
    pv = load_entsoe_solar_forecast(valuedate=vdate, country_code='FR')
    pv = None
    return (pv,)


@app.cell
def _():
    demand = None
    return (demand,)


@app.cell
def _(mo):
    capacity = mo.ui.number(start=1, stop=100, value=1, label='Capacity [MW]', step=1)
    return (capacity,)


@app.cell(hide_code=True)
def _(capacity, mo):
    max_charge_power = mo.ui.number(start=0, stop=capacity.value, value=0.1*capacity.value, label='Max charge [Mwh]', step=0.1)
    max_discharge_power = mo.ui.number(start=0, stop=capacity.value, value=0.1*capacity.value, label='Max discharge [MWh]', step=0.1)
    charge_efficiency = mo.ui.slider(start=0.1, stop=1, value=0.9, label='Charge efficiency [%]', step=0.1)
    discharge_efficiency = mo.ui.slider(start=0.1, stop=1, value=0.9, label='Discharge efficiency [%]', step=0.1)
    starting_soc = mo.ui.slider(start=0, stop=capacity.value, value=capacity.value, label='Starting SOC', step=0.1)


    mo.vstack([
        mo.md('## Battery'),
        capacity,
        starting_soc,
        max_charge_power,
        max_discharge_power,
        charge_efficiency,
        discharge_efficiency
    ])
    return (
        charge_efficiency,
        discharge_efficiency,
        max_charge_power,
        max_discharge_power,
        starting_soc,
    )


@app.cell
def _(
    Battery,
    capacity,
    charge_efficiency,
    discharge_efficiency,
    max_charge_power,
    max_discharge_power,
    starting_soc,
):
    battery = Battery(
        capacity=capacity.value, 
        max_charge_power=max_charge_power.value, 
        max_discharge_power=max_discharge_power.value,
        charge_efficiency=charge_efficiency.value,
        discharge_efficiency=discharge_efficiency.value,
        soc=starting_soc.value,
        soc_end=starting_soc.value,
        max_daily_cycles=2
    )
    battery
    return (battery,)


@app.cell
def _(DAOptimisation, battery, demand, pricedf, pv):
    daproblem = DAOptimisation(battery=battery, 
                               daprice=pricedf['daprice'].to_numpy(),
                               pv=pv,
                               demand=demand,
                               degradation_cost=0.,
                              product='15m')
    daproblem.solve()
    return (daproblem,)


@app.cell(column=1)
def _(daproblem):
    daproblem.plot(figsize=(6,5), soc=True, pv=True)
    return


@app.cell
def _(daproblem):
    daproblem
    return


if __name__ == "__main__":
    app.run()
