"""Round-trip and corruption tests for the cache CSV reader.

custom_read_csv must read back exactly what fetch_data writes via df.to_csv
(MultiIndex (Type, Ticker) columns + a Date index). The production bug this
guards against was KeyError: 'Date' — the reader assuming a literal Date
column the writer never emits.
"""

import numpy as np
import pandas as pd

from data_handler import custom_read_csv


def _make_frame():
    """A frame shaped exactly like what fetch_data caches."""
    index = pd.to_datetime(["2008-01-02", "2008-01-03", "2008-01-04"])
    index.name = "Date"
    columns = pd.MultiIndex.from_product(
        [["Close", "High", "Low", "Open", "Volume"], ["TSLA"]],
        names=["Type", "Ticker"],
    )
    data = np.arange(15, dtype=float).reshape(3, 5)
    return pd.DataFrame(data, index=index, columns=columns)


def test_round_trips_multiindex_cache(tmp_path):
    path = tmp_path / "TSLA_data.csv"
    _make_frame().to_csv(path)

    df = custom_read_csv(path)

    assert isinstance(df.index, pd.DatetimeIndex)
    assert df.index.name == "Date"
    assert df.index[-1] == pd.Timestamp("2008-01-04")  # delta path relies on this
    assert isinstance(df.columns, pd.MultiIndex)
    assert df.columns.names == ["Type", "Ticker"]
    assert all(np.issubdtype(dt, np.number) for dt in df.dtypes)
    assert df.shape == (3, 5)


def test_empty_file_is_treated_as_corrupt(tmp_path):
    path = tmp_path / "EMPTY_data.csv"
    path.write_text("", encoding="utf-8")

    # Must not raise — returns empty so fetch_data removes and re-fetches.
    assert custom_read_csv(path).empty


def test_header_only_file_is_empty(tmp_path):
    path = tmp_path / "HDR_data.csv"
    # Two header rows + stray Date row, no data — a crash before any row landed.
    path.write_text("Price,Close\nTicker,TSLA\nDate,\n", encoding="utf-8")

    assert custom_read_csv(path).empty


def test_all_nan_value_section_is_treated_as_corrupt(tmp_path):
    path = tmp_path / "NAN_data.csv"
    path.write_text(
        "Price,Close\nTicker,TSLA\nDate,\n2008-01-02,garbage\n2008-01-03,more\n",
        encoding="utf-8",
    )

    # Every value coerces to NaN -> corrupt -> empty, so the file gets removed.
    assert custom_read_csv(path).empty
