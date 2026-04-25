"""As-of-date Tushare hygiene helpers for public examples.

The functions here focus on two common leakage traps:

1. ST status must be evaluated on the signal date, not from the current name.
2. BSE symbols need old/new code mapping and pre-listing exclusion.

These utilities are deliberately generic. They do not contain production
feature engineering, private universes, or proprietary alpha logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd


OPEN_END_DATE = "20991231"


def to_yyyymmdd(value: object) -> str:
    """Normalize common date formats to Tushare-style YYYYMMDD strings."""
    if pd.isna(value):
        raise ValueError("date value is missing")
    if isinstance(value, pd.Timestamp):
        return value.strftime("%Y%m%d")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "").replace("/", "")
    if len(text) == 8 and text.isdigit():
        return text
    return pd.to_datetime(value).strftime("%Y%m%d")


def normalize_ts_code(values: Iterable[object] | pd.Series) -> pd.Series:
    """Return uppercase Tushare codes without changing the index."""
    return pd.Series(values).astype("string").str.strip().str.upper()


def _first_existing(columns: Iterable[str], candidates: Iterable[str]) -> str | None:
    names = set(columns)
    for candidate in candidates:
        if candidate in names:
            return candidate
    return None


def build_st_intervals_from_namechange(namechange: pd.DataFrame) -> pd.DataFrame:
    """Build historical ST intervals from Tushare `namechange`.

    Expected columns include `ts_code`, `start_date`, and optionally
    `end_date`, `name`, `change_reason`, or `ann_date`.
    """
    required = {"ts_code", "start_date"}
    missing = required - set(namechange.columns)
    if missing:
        raise ValueError(f"namechange is missing columns: {sorted(missing)}")

    text_cols = [c for c in ["name", "change_reason"] if c in namechange.columns]
    if text_cols:
        text = namechange[text_cols].fillna("").astype(str).agg(" ".join, axis=1)
        st_mask = text.str.contains("ST", case=False, regex=False)
    else:
        st_mask = pd.Series(True, index=namechange.index)

    end_col = "end_date" if "end_date" in namechange.columns else None
    out = namechange.loc[st_mask, ["ts_code", "start_date"]].copy()
    out["end_date"] = (
        namechange.loc[st_mask, end_col].fillna(OPEN_END_DATE)
        if end_col
        else OPEN_END_DATE
    )
    out["ts_code"] = normalize_ts_code(out["ts_code"]).to_numpy()
    out["start_date"] = out["start_date"].map(to_yyyymmdd)
    out["end_date"] = out["end_date"].map(to_yyyymmdd)
    out["source"] = "namechange"
    return out.drop_duplicates().sort_values(["ts_code", "start_date"]).reset_index(drop=True)


def st_mask_asof(
    universe: pd.DataFrame,
    trade_date: object,
    *,
    code_col: str = "ts_code",
    stock_st_daily: pd.DataFrame | None = None,
    namechange_intervals: pd.DataFrame | None = None,
) -> pd.Series:
    """Return a boolean mask for stocks that are ST on `trade_date`.

    `stock_st_daily` is treated as the preferred source when available.
    `namechange_intervals` is a useful audit/fallback source.
    """
    if code_col not in universe.columns:
        raise ValueError(f"universe is missing code column: {code_col}")

    date = to_yyyymmdd(trade_date)
    codes = normalize_ts_code(universe[code_col])
    st_codes: set[str] = set()

    if stock_st_daily is not None and not stock_st_daily.empty:
        date_col = _first_existing(stock_st_daily.columns, ["trade_date", "date"])
        code_source_col = _first_existing(stock_st_daily.columns, ["ts_code", code_col])
        if date_col is None or code_source_col is None:
            raise ValueError("stock_st_daily needs a date column and a ts_code column")
        daily = stock_st_daily.copy()
        daily[date_col] = daily[date_col].map(to_yyyymmdd)
        st_codes.update(
            normalize_ts_code(daily.loc[daily[date_col] == date, code_source_col]).dropna().tolist()
        )

    if namechange_intervals is not None and not namechange_intervals.empty:
        required = {"ts_code", "start_date", "end_date"}
        missing = required - set(namechange_intervals.columns)
        if missing:
            raise ValueError(f"namechange_intervals is missing columns: {sorted(missing)}")
        intervals = namechange_intervals.copy()
        intervals["ts_code"] = normalize_ts_code(intervals["ts_code"]).to_numpy()
        intervals["start_date"] = intervals["start_date"].map(to_yyyymmdd)
        intervals["end_date"] = intervals["end_date"].fillna(OPEN_END_DATE).map(to_yyyymmdd)
        active = intervals[(intervals["start_date"] <= date) & (intervals["end_date"] >= date)]
        st_codes.update(active["ts_code"].dropna().tolist())

    return pd.Series(codes.isin(st_codes).to_numpy(), index=universe.index, name="is_st_asof")


def filter_st_asof(
    universe: pd.DataFrame,
    trade_date: object,
    *,
    code_col: str = "ts_code",
    stock_st_daily: pd.DataFrame | None = None,
    namechange_intervals: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Exclude ST stocks using only information available as of `trade_date`."""
    mask = st_mask_asof(
        universe,
        trade_date,
        code_col=code_col,
        stock_st_daily=stock_st_daily,
        namechange_intervals=namechange_intervals,
    )
    return universe.loc[~mask].copy()


@dataclass(frozen=True)
class BseMapping:
    """Normalized Tushare BSE old/new code mapping."""

    frame: pd.DataFrame

    @classmethod
    def from_tushare(cls, bse_mapping: pd.DataFrame) -> "BseMapping":
        required = {"o_code", "n_code", "list_date"}
        missing = required - set(bse_mapping.columns)
        if missing:
            raise ValueError(f"bse_mapping is missing columns: {sorted(missing)}")
        frame = bse_mapping[["o_code", "n_code", "list_date"]].copy()
        frame["o_code"] = normalize_ts_code(frame["o_code"]).to_numpy()
        frame["n_code"] = normalize_ts_code(frame["n_code"]).to_numpy()
        frame["list_date"] = frame["list_date"].map(to_yyyymmdd)
        return cls(frame.drop_duplicates().sort_values("list_date").reset_index(drop=True))


def apply_bse_mapping_asof(
    frame: pd.DataFrame,
    trade_date: object,
    mapping: BseMapping,
    *,
    code_col: str = "ts_code",
) -> pd.DataFrame:
    """Map old BSE/NEEQ-style codes to post-listing BSE codes as of a date."""
    if code_col not in frame.columns:
        raise ValueError(f"frame is missing code column: {code_col}")
    date = to_yyyymmdd(trade_date)
    eligible = mapping.frame[mapping.frame["list_date"] <= date]
    code_map = dict(zip(eligible["o_code"], eligible["n_code"]))

    out = frame.copy()
    codes = normalize_ts_code(out[code_col])
    out[code_col] = codes.map(lambda x: code_map.get(x, x)).to_numpy()
    return out


def bse_eligible_mask_asof(
    universe: pd.DataFrame,
    trade_date: object,
    *,
    code_col: str = "ts_code",
    stock_basic: pd.DataFrame | None = None,
    mapping: BseMapping | None = None,
) -> pd.Series:
    """Return True for rows that are valid for the BSE universe on `trade_date`.

    This prevents pre-listing NEEQ/selected-layer rows from leaking into a BSE
    backtest before the listing or mapping date.
    """
    if code_col not in universe.columns:
        raise ValueError(f"universe is missing code column: {code_col}")

    date = to_yyyymmdd(trade_date)
    codes = normalize_ts_code(universe[code_col])
    eligible = pd.Series(True, index=universe.index, name="bse_eligible_asof")

    if stock_basic is not None and not stock_basic.empty and "list_date" in stock_basic.columns:
        basic_code_col = _first_existing(stock_basic.columns, ["ts_code", code_col])
        if basic_code_col is None:
            raise ValueError("stock_basic needs a ts_code column")
        basic = stock_basic[[basic_code_col, "list_date"]].copy()
        basic[basic_code_col] = normalize_ts_code(basic[basic_code_col]).to_numpy()
        basic["list_date"] = basic["list_date"].map(to_yyyymmdd)
        list_dates = dict(zip(basic[basic_code_col], basic["list_date"]))
        bse_rows = codes.str.endswith(".BJ", na=False)
        too_early = codes.map(list_dates).fillna("00000000") > date
        eligible.loc[bse_rows & too_early] = False

    if mapping is not None and not mapping.frame.empty:
        old_to_date = dict(zip(mapping.frame["o_code"], mapping.frame["list_date"]))
        new_to_date = dict(zip(mapping.frame["n_code"], mapping.frame["list_date"]))
        mapped_dates = codes.map(lambda x: old_to_date.get(x, new_to_date.get(x)))
        pre_listing = mapped_dates.notna() & (mapped_dates > date)
        eligible.loc[pre_listing] = False

    return eligible


def filter_bse_prelisting_asof(
    universe: pd.DataFrame,
    trade_date: object,
    *,
    code_col: str = "ts_code",
    stock_basic: pd.DataFrame | None = None,
    mapping: BseMapping | None = None,
) -> pd.DataFrame:
    """Exclude BSE rows that are not yet valid as of `trade_date`."""
    mask = bse_eligible_mask_asof(
        universe,
        trade_date,
        code_col=code_col,
        stock_basic=stock_basic,
        mapping=mapping,
    )
    return universe.loc[mask].copy()
