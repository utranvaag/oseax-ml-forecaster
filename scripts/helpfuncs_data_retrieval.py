"""Functions and methods for assisting in data retrieval"""

import pandas as pd
import numpy as np
from darts import TimeSeries
from darts.utils.timeseries_generation import (
    datetime_attribute_timeseries,
    holidays_timeseries,
)


def add_date_and_time_series(
        dataframe: pd.DataFrame,
):
    """
    Retreives date and time-related time series/features (retreived by using
    Darts TimeSeries object) for the DataFrame provided as argument, add these
    to this DataFrame and returns it.
    """

    series = TimeSeries.from_dataframe(dataframe.copy())

    # Adding multiple date and time series:
    # Year
    series = series.stack(
        datetime_attribute_timeseries(
            series,
            attribute="year",
            one_hot=False,
            dtype=np.float32,
            with_columns="Year (Year)",
        )
    )
    # Month of year
    series = series.stack(
        datetime_attribute_timeseries(
            series,
            attribute="month",
            one_hot=False,
            dtype=np.float32,
            with_columns="Month of year (Month)",
        )
    )
    # Day of month
    series = series.stack(
        datetime_attribute_timeseries(
            series,
            attribute="day",
            one_hot=False,
            dtype=np.float32,
            with_columns="Day of month (Day)",
        )
    )
    # Week of year
    series = series.stack(
        datetime_attribute_timeseries(
            series,
            attribute="weekofyear",
            one_hot=False,
            dtype=np.float32,
            with_columns="Week of year (Week)",
        )
    )
    # Day of week
    series = series.stack(
        datetime_attribute_timeseries(
            series,
            attribute="weekday",
            one_hot=False,
            dtype=np.float32,
            with_columns="Day of week (Day/week)",
        )
    )
    # Holiday binary indicator
    series = series.stack(
        holidays_timeseries(
            series.time_index,
            country_code="NOR",
            dtype=np.float32,
            column_name="Holiday binary indicator (Holidays)",
        )
    )
    # Datetime value
    series = series.stack(
        TimeSeries.from_times_and_values(
            times=series.time_index,
            values=np.arange(len(series)),
            columns=["Datetime value (Datetime)"]
        )
    )

    return series.pd_dataframe(copy=True)


def add_index_technicals_series(
        dataframe: pd.DataFrame,
        index_name: str = "Oslo Børs all-share index (OSEAX)",
):
    """
    Calculates Moving Averages (MA), Residual Strength Index (RSI), and Moving
    Average Convergence Divergence (MACD), and adds them to the dataframe
    given as input.
    """

    df_calc = pd.DataFrame()

    # Moving Averages (MA):

    df_calc['OSEAX 50d moving average (50d MA)']   = (
        dataframe[index_name].dropna()).rolling(window=50).mean()
    df_calc['OSEAX 200d moving average (200d MA)'] = (
        dataframe[index_name].dropna()).rolling(window=200).mean()

    # Residual Strength Index (RSI):

    period = 14

    # 1. calculate price changes:
    df_calc['Price_Change'] = dataframe[index_name].dropna().diff()

    # 2. calculate gains and losses:
    df_calc['Gain'] = np.where(
        df_calc['Price_Change'] >= 0, df_calc['Price_Change'],
        0
    )
    df_calc['Loss'] = np.where(
        df_calc['Price_Change'] < 0, abs(df_calc['Price_Change']),
        0
    )

    # 3. Calculate average gain and average loss:
    df_calc['Avg_Gain'] = df_calc['Gain'].rolling(window=period).mean()
    df_calc['Avg_Loss'] = df_calc['Loss'].rolling(window=period).mean()

    # 4. Calculate relative strength (RS) and relative strength index (RSI):
    df_calc['RS'] = df_calc['Avg_Gain'] / df_calc['Avg_Loss']
    df_calc['OSEAX relative strength idx (RSI)'] = 100 - (
        100 / (1 + df_calc['RS'])
    )

    # Moving Average Convergence Divergence (MACD):

    # 1. get 12-d and 26-d exponential moving averages (EMA) of close price:
    df_calc['EMA_12'] = (  # 12 period EMA
        dataframe['Oslo Børs all-share index (OSEAX)'].dropna()
        ).ewm(span=12, adjust=False).mean()
    df_calc['EMA_26'] = (  # 26 period EMA
        dataframe['Oslo Børs all-share index (OSEAX)'].dropna()
        ).ewm(span=26, adjust=False).mean()

    # 2. Get the MACD line by subtracting the 26-day EMA from the 12-day EMA:
    df_calc['MACD_Line'] = df_calc['EMA_12'] - df_calc['EMA_26']

    # 3. Calculate the signal line as a 9-day EMA of the MACD line:
    df_calc['Signal_Line'] = df_calc['MACD_Line'].ewm(
        span=9,
        adjust=False,
    ).mean()

    # 4. Get the MACD histogram (difference between the MACD and signal line):
    macd_col_name = 'OSEAX MACD histogram (MACD)'
    df_calc[macd_col_name] = df_calc['MACD_Line'] - df_calc['Signal_Line']

    # Merging new time series/features with main DataFrame:

    df_merged = dataframe.merge(
        df_calc[[
            'OSEAX 50d moving average (50d MA)',
            'OSEAX 200d moving average (200d MA)',
            'OSEAX relative strength idx (RSI)',
            'OSEAX MACD histogram (MACD)',
        ]],
        on='Date',
        how='left'
    )

    return df_merged
