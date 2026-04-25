"""Execution-aware Top-K sleeve simulator for public examples.

Protocol:
    signal date T -> buy at T+1 adjusted open -> target sell at T+3 adjusted close

The simulator includes blocked entry, delayed exit, equal-weight Top-K sleeves,
and an optional cost layer. It is intentionally compact and does not contain
private selector logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class BacktestConfig:
    top_k: int = 10
    holding_days: int = 3
    buy_cost_bps: float = 12.6
    sell_cost_bps: float = 17.6
    initial_nav: float = 1.0


def _date(value: object) -> str:
    if pd.isna(value):
        raise ValueError("date value is missing")
    text = str(value).strip()
    if text.endswith(".0"):
        text = text[:-2]
    text = text.replace("-", "").replace("/", "")
    if len(text) == 8 and text.isdigit():
        return text
    return pd.to_datetime(value).strftime("%Y%m%d")


def _prepare_prices(price_status: pd.DataFrame) -> dict[tuple[str, str], dict[str, Any]]:
    required = {"trade_date", "ts_code", "adj_open", "adj_close"}
    missing = required - set(price_status.columns)
    if missing:
        raise ValueError(f"price_status is missing columns: {sorted(missing)}")

    prices = price_status.copy()
    prices["trade_date"] = prices["trade_date"].map(_date)
    prices["ts_code"] = prices["ts_code"].astype(str).str.upper().str.strip()
    if "open_tradable" not in prices.columns:
        prices["open_tradable"] = prices["adj_open"].notna() & (prices["adj_open"] > 0)
    if "close_tradable" not in prices.columns:
        prices["close_tradable"] = prices["adj_close"].notna() & (prices["adj_close"] > 0)
    return {
        (row.trade_date, row.ts_code): row._asdict()
        for row in prices.itertuples(index=False)
    }


def add_execution_dates(
    selections: pd.DataFrame,
    calendar: list[str],
    *,
    signal_col: str = "trade_date",
    holding_days: int = 3,
) -> pd.DataFrame:
    """Attach entry and target-exit dates using a trading-day calendar."""
    if signal_col not in selections.columns or "ts_code" not in selections.columns:
        raise ValueError("selections needs signal date and ts_code columns")
    dates = [_date(d) for d in calendar]
    pos = {d: i for i, d in enumerate(dates)}

    out = selections.copy()
    out["signal_date"] = out[signal_col].map(_date)
    out["ts_code"] = out["ts_code"].astype(str).str.upper().str.strip()

    def entry_date(signal_date: str) -> str | None:
        i = pos.get(signal_date)
        return dates[i + 1] if i is not None and i + 1 < len(dates) else None

    def target_exit_date(signal_date: str) -> str | None:
        i = pos.get(signal_date)
        j = i + holding_days if i is not None else None
        return dates[j] if j is not None and j < len(dates) else None

    out["entry_date"] = out["signal_date"].map(entry_date)
    out["target_exit_date"] = out["signal_date"].map(target_exit_date)
    return out.dropna(subset=["entry_date", "target_exit_date"]).reset_index(drop=True)


def simulate_topk_sleeves(
    selections: pd.DataFrame,
    price_status: pd.DataFrame,
    calendar: list[str] | None = None,
    config: BacktestConfig = BacktestConfig(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run gross and net simulations and return `(curve, trades)`.

    `selections` should contain one row per selected stock with columns
    `trade_date`, `ts_code`, and optionally `rank`. `price_status` should
    contain adjusted open/close prices and tradability flags.
    """
    if calendar is None:
        calendar = sorted({_date(d) for d in price_status["trade_date"].unique()})
    prepared = add_execution_dates(selections, calendar, holding_days=config.holding_days)
    if "rank" in prepared.columns:
        prepared = prepared.sort_values(["signal_date", "rank"])
    prepared = prepared.groupby("signal_date", group_keys=False).head(config.top_k)

    gross_curve, gross_trades = _simulate(prepared, price_status, calendar, config, cost_side="gross")
    net_curve, net_trades = _simulate(prepared, price_status, calendar, config, cost_side="net")

    curve = gross_curve.merge(net_curve, on="trade_date", suffixes=("_gross", "_net"))
    curve["drawdown_gross"] = curve["nav_gross"] / curve["nav_gross"].cummax() - 1.0
    curve["drawdown_net"] = curve["nav_net"] / curve["nav_net"].cummax() - 1.0

    trades = pd.concat([gross_trades, net_trades], ignore_index=True)
    return curve, trades


def _simulate(
    selections: pd.DataFrame,
    price_status: pd.DataFrame,
    calendar: list[str],
    config: BacktestConfig,
    *,
    cost_side: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    prices = _prepare_prices(price_status)
    buy_cost = config.buy_cost_bps / 10_000.0 if cost_side == "net" else 0.0
    sell_cost = config.sell_cost_bps / 10_000.0 if cost_side == "net" else 0.0
    by_entry = {d: g.copy() for d, g in selections.groupby("entry_date")}

    cash = float(config.initial_nav)
    positions: list[dict[str, Any]] = []
    curve_rows: list[dict[str, Any]] = []
    trade_rows: list[dict[str, Any]] = []

    for trade_date in [_date(d) for d in calendar]:
        still_open: list[dict[str, Any]] = []
        for pos in positions:
            px = prices.get((trade_date, pos["ts_code"]))
            due = trade_date >= pos["target_exit_date"]
            can_exit = px is not None and bool(px["close_tradable"])
            if due and can_exit:
                proceeds = pos["shares"] * float(px["adj_close"]) * (1.0 - sell_cost)
                cash += proceeds
                trade_rows.append(
                    {
                        "side": cost_side,
                        "event": "sell",
                        "trade_date": trade_date,
                        "ts_code": pos["ts_code"],
                        "target_exit_date": pos["target_exit_date"],
                        "value": proceeds,
                    }
                )
            else:
                still_open.append(pos)
                if due and not can_exit:
                    trade_rows.append(
                        {
                            "side": cost_side,
                            "event": "delayed_exit",
                            "trade_date": trade_date,
                            "ts_code": pos["ts_code"],
                            "target_exit_date": pos["target_exit_date"],
                            "value": 0.0,
                        }
                    )
        positions = still_open

        day = by_entry.get(trade_date)
        if day is not None and not day.empty:
            nav_before_entry = cash + _mark_positions(positions, prices, trade_date)
            sleeve_budget = min(cash, nav_before_entry / float(config.holding_days))
            name_budget = sleeve_budget / float(max(len(day), 1))

            for row in day.itertuples(index=False):
                code = row.ts_code
                px = prices.get((trade_date, code))
                can_enter = px is not None and bool(px["open_tradable"])
                if can_enter and name_budget > 0:
                    execution_value = name_budget * (1.0 - buy_cost)
                    shares = execution_value / float(px["adj_open"])
                    cash -= name_budget
                    positions.append(
                        {
                            "ts_code": code,
                            "shares": shares,
                            "entry_date": trade_date,
                            "target_exit_date": row.target_exit_date,
                        }
                    )
                    trade_rows.append(
                        {
                            "side": cost_side,
                            "event": "buy",
                            "trade_date": trade_date,
                            "ts_code": code,
                            "target_exit_date": row.target_exit_date,
                            "value": name_budget,
                        }
                    )
                else:
                    trade_rows.append(
                        {
                            "side": cost_side,
                            "event": "blocked_entry",
                            "trade_date": trade_date,
                            "ts_code": code,
                            "target_exit_date": row.target_exit_date,
                            "value": 0.0,
                        }
                    )

        nav = cash + _mark_positions(positions, prices, trade_date)
        curve_rows.append({"trade_date": trade_date, "nav": nav, "cash": cash, "open_positions": len(positions)})

    curve = pd.DataFrame(curve_rows)
    trades = pd.DataFrame(trade_rows)
    return curve, trades


def _mark_positions(
    positions: list[dict[str, Any]],
    prices: dict[tuple[str, str], dict[str, Any]],
    trade_date: str,
) -> float:
    value = 0.0
    for pos in positions:
        px = prices.get((trade_date, pos["ts_code"]))
        if px is not None and pd.notna(px["adj_close"]):
            value += pos["shares"] * float(px["adj_close"])
    return value


def summarize_curve(curve: pd.DataFrame, *, nav_col: str = "nav_net") -> dict[str, float]:
    """Return a compact performance summary for a daily NAV curve."""
    if nav_col not in curve.columns:
        raise ValueError(f"curve is missing {nav_col}")
    nav = pd.to_numeric(curve[nav_col], errors="coerce").dropna()
    ret = nav.pct_change().dropna()
    if nav.empty:
        return {}
    ann_ret = (nav.iloc[-1] / nav.iloc[0]) ** (252.0 / max(len(nav) - 1, 1)) - 1.0
    ann_vol = ret.std(ddof=0) * (252.0**0.5) if not ret.empty else 0.0
    sharpe = ret.mean() * 252.0 / ann_vol if ann_vol > 0 else 0.0
    drawdown = nav / nav.cummax() - 1.0
    return {
        "final_nav": float(nav.iloc[-1]),
        "annualized_return": float(ann_ret),
        "annualized_vol": float(ann_vol),
        "sharpe": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
