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
    from bessopt.battery import Battery, BatteryConstraints
    from bessopt.problems.auction import AuctionOptimisation
    import numpy as np
    import matplotlib.pyplot as plt
    import polars as pl
    from datetime import datetime

    from bessopt.data import load_entsoe_dayahead_price
    from bessopt.data import load_entsoe_solar_forecast

    vdate = datetime(2026, 4, 12)
    country_code = 'DE_LU'
    apikey_val = '95ac0990-852d-4f7d-bb76-58b1d3bcdbe2'
    return (
        AuctionOptimisation,
        Battery,
        BatteryConstraints,
        apikey_val,
        country_code,
        load_entsoe_dayahead_price,
        mo,
        vdate,
    )


@app.cell
def _(apikey_val, mo):
    apikey = mo.ui.text(label='entsoe apikey', value=apikey_val)
    apikey
    return (apikey,)


@app.cell
def _(apikey, country_code, load_entsoe_dayahead_price, vdate):
    pricedf = load_entsoe_dayahead_price(valuedate=vdate, country_code=country_code, api_key=apikey.value)
    return (pricedf,)


@app.cell
def _(mo):
    capacity = mo.ui.number(start=1, stop=100, value=1, label='Capacity [MW]', step=1)
    return


@app.cell
def _(Battery, BatteryConstraints):
    battery = Battery(
        capacity=1,
        max_charge_power = 0.1,
        max_discharge_power = 0.1,
        soc=0.5
    )

    battery_constraints = BatteryConstraints(

    )
    return battery, battery_constraints


@app.cell
def _(mo):
    mo.md("""
    ## Example 1 — Unconstrained auction (no pre-existing schedule)
    """)
    return


@app.cell
def _(AuctionOptimisation, battery, battery_constraints, pricedf):
    auction = AuctionOptimisation(
        battery=battery,
        battery_constraints = battery_constraints,
        price=pricedf['price'].to_numpy(),
        degradation_cost=0.,
        product='15m',
    )
    auction.solve()
    return (auction,)


@app.cell(column=1)
def _(auction):
    auction.plot(figsize=(6, 5), soc=True, pv=True)
    return


@app.cell
def _(auction):
    auction
    return


@app.cell(column=2)
def _(mo):
    mo.md("""
    ## Example 2 — Residual auction (with pre-existing schedule)

    We feed the charge and discharge schedules from Example 1 back into a fresh
    `AuctionOptimisation` as `battery_charge_schedule` and
    `battery_discharge_schedule`. In residual mode the existing dispatch is
    treated as already monetised; the solver only earns/pays on the
    *incremental* trade between the new schedule and the existing one. With
    unchanged prices the optimum is to leave the schedule untouched
    (residual revenue ≈ 0).
    """)
    return


@app.cell
def _(auction):
    existing_charge = auction.battery_charge.value
    existing_discharge = auction.battery_discharge.value
    return existing_charge, existing_discharge


@app.cell
def _(AuctionOptimisation, battery, existing_charge, existing_discharge, pricedf):
    auction_residual = AuctionOptimisation(
        battery=battery,
        price=pricedf['price'].to_numpy(),
        battery_charge_schedule=existing_charge,
        battery_discharge_schedule=existing_discharge,
        degradation_cost=0.,
        product='15m',
    )
    auction_residual.solve()
    return (auction_residual,)


@app.cell(column=3)
def _(auction_residual):
    auction_residual.plot(figsize=(6, 5), soc=True, pv=True)
    return


@app.cell
def _(auction_residual):
    auction_residual
    return


@app.cell
def _(auction_residual):
    auction_residual.get_results().select(
        ['step', 'price', 'da_charge', 'da_discharge',
         'battery_charge', 'battery_discharge',
         'residual_charge', 'residual_discharge', 'revenue']
    )
    return


if __name__ == "__main__":
    app.run()
