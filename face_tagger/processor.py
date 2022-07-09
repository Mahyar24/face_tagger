import pathlib
import time
from typing import Union

import numpy as np
import pandas as pd


def pretty_time(sec: int) -> str:
    return time.strftime("%H:%M:%S", time.gmtime(sec))


def make_df(data: np.ndarray, cols: list[str]) -> pd.DataFrame:
    return pd.DataFrame(data, columns=cols)


def generate_time_intervals(
    series: pd.Series, rolling_window: int = 5, threshold: int = 2, pretty: bool = True
) -> list[Union[tuple[str, str], tuple[int, int]]]:
    assert rolling_window >= threshold, "rolling_window must be greater than threshold"

    return (
        series.rolling(rolling_window, center=True)
        .sum()[lambda x: x >= threshold]
        .index.to_series()
        .pipe(
            lambda ser_: ser_.groupby(ser_.diff().ne(1).cumsum()).agg(["first", "last"])
        )
        .loc[lambda df_: (df_["last"] - df_["first"]) >= rolling_window]
        .apply(
            {
                "first": lambda x: max(0, x - rolling_window + threshold),
                "last": lambda x: max(0, x - rolling_window),
            }
        )
        .applymap(pretty_time if pretty else lambda x: x)
        .apply(tuple, 1)
        .tolist()
    )


def generate_json(
    file: pathlib.Path,
    data: np.ndarray,
    cols: list[str],
    rolling_window: int = 5,
    threshold: int = 2,
    pretty: bool = True,
) -> None:
    """Gen"""
    (
        make_df(data, cols)
        .apply(
            lambda ser: generate_time_intervals(ser, rolling_window, threshold, pretty)
        )
        .to_json(file, indent=4),
    )
