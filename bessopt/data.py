from entsoe import EntsoePandasClient
import pandas as pd
from datetime import date, datetime, time
import os
import polars as pl

_ENTSOE_TZ = 'Europe/Brussels'


def _get_client(api_key: str = None) -> EntsoePandasClient:
    return EntsoePandasClient(api_key=os.environ.get('ENTSOE_API_KEY', api_key))


def _day_bounds(valuedate: date) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return tz-aware start (00:00) and exclusive end (00:00 next day) for a date."""
    start = pd.Timestamp(datetime.combine(valuedate, time.min)).tz_localize(_ENTSOE_TZ)
    end   = pd.Timestamp(datetime.combine(valuedate, time.min)).tz_localize(_ENTSOE_TZ) + pd.DateOffset(days=1)
    return start, end


def _index_to_valuedate(df: pd.DataFrame) -> pl.DataFrame:
    """Reset tz-aware DatetimeIndex to a naive UTC 'valuedate' column and convert to polars."""
    df = df.copy()
    df.index = df.index.tz_convert(_ENTSOE_TZ).tz_localize(None)
    df.index.name = 'valuedate'
    return pl.from_pandas(df.reset_index())


def load_entsoe_solar_forecast(country_code: str, valuedate: date, api_key: str = None) -> pl.DataFrame:
    """
    Returns a polars DataFrame with columns:
        valuedate : datetime  – naive UTC timestamp
        solar     : float     – solar capacity factor (generation forecast / installed capacity, 0-1)
    """
    client = _get_client(api_key=api_key)
    start_ts, end_ts = _day_bounds(valuedate)

    forecast = client.query_wind_and_solar_forecast(country_code, start=start_ts, end=end_ts, psr_type='B16')
    forecast = forecast[['Solar']]

    capacity = client.query_installed_generation_capacity(country_code, start=start_ts, end=end_ts, psr_type='B16')
    # capacity is annual; forward-fill to align with hourly forecast timestamps
    cap_aligned = (
        capacity['Solar']
        .reindex(forecast.index.union(capacity.index))
        .sort_index()
        .ffill()
        .reindex(forecast.index)
    )

    df = (forecast['Solar'] / cap_aligned).to_frame('solar')
    return _index_to_valuedate(df)


def load_entsoe_dayahead_price(country_code: str, valuedate: date, api_key: str = None) -> pl.DataFrame:
    """
    Returns a polars DataFrame with columns:
        valuedate : datetime  – naive UTC timestamp
        daprice   : float     – day-ahead price (EUR/MWh)
    """
    client = _get_client(api_key=api_key)
    start_ts, end_ts = _day_bounds(valuedate)
    series = client.query_day_ahead_prices(country_code, start=start_ts, end=end_ts)
    df = series.to_frame('price')
    return _index_to_valuedate(df)


def load_entsoe_imbalance_price(country_code: str, valuedate: date, api_key: str = None) -> pl.DataFrame:
    """
    Returns a polars DataFrame with columns:
        valuedate : datetime  – naive UTC timestamp
        long      : float     – imbalance price for long position (EUR/MWh)
        short     : float     – imbalance price for short position (EUR/MWh)
    """
    client = _get_client(api_key=api_key)
    start_ts, end_ts = _day_bounds(valuedate)
    df = client.query_imbalance_prices(country_code, start=start_ts, end=end_ts, psr_type=None)
    df = df[['Long', 'Short']].rename(columns={'Long': 'long', 'Short': 'short'})
    return _index_to_valuedate(df)


if __name__ == '__main__':
    valuedate    = date(2025, 4, 1)
    country_code = 'DK_1'

    solar     = load_entsoe_solar_forecast(country_code, valuedate)
    da_price  = load_entsoe_dayahead_price(country_code, valuedate)
    imb_price = load_entsoe_imbalance_price(country_code, valuedate)

    print(solar.head())
    print(da_price.head())
    print(imb_price.head())
