from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


INITIAL_EQUITY = 100_000.0
MAX_SLOTS = 5
SLOT_ALLOC_PCT = 0.25
COMMISSION_EACH_SIDE = 0.002


@dataclass(frozen=True)
class TwoLaneParams:
    leader_min: float = 94.0
    standard_setup_min: float = 70.0
    standard_trigger_min: float = 80.0
    standard_cum_vol_ratio_min: float = 1.35
    tight_setup_min: float = 50.0
    tight_trigger_min: float = 65.0
    tight_cum_vol_ratio_min: float = 1.0
    tight_entry_dist_norm_max: float = 1.20
    tight_gap_norm_max: float = 1.10


def _clip01(series: pd.Series | np.ndarray) -> pd.Series:
    return pd.Series(np.clip(np.asarray(series, dtype=float), 0.0, 1.0))


def _num(row: pd.Series, *cols: str, default: float = np.nan) -> float:
    for col in cols:
        if col in row.index and pd.notna(row[col]):
            value = pd.to_numeric(row[col], errors="coerce")
            if pd.notna(value):
                return float(value)
    return default


def _equity_factor(entry_price: float, realized: float, remaining: float, mark_price: float) -> float:
    return 1.0 + realized + remaining * (mark_price / entry_price - 1.0)


def load_two_lane_events(standard_events: str | Path, tight_events: str | Path) -> pd.DataFrame:
    standard = pd.read_csv(standard_events, low_memory=False)
    tight = pd.read_csv(tight_events, low_memory=False)
    standard["entry_source"] = "standard_breakout"
    tight["entry_source"] = "tight_reversal"
    standard["entry_priority_bucket"] = 0
    tight["entry_priority_bucket"] = 1
    events = pd.concat([standard, tight], ignore_index=True, sort=False)
    events["date"] = pd.to_datetime(events["date"]).dt.normalize()
    events["symbol"] = events["symbol"].astype(str).str.upper()
    return events


def base_entry_mask(events: pd.DataFrame, params: TwoLaneParams = TwoLaneParams()) -> pd.Series:
    leader = pd.to_numeric(events["leader_score"], errors="coerce")
    setup = pd.to_numeric(events["setup_score_pre"], errors="coerce")
    trigger = pd.to_numeric(events["trigger_score"], errors="coerce")
    cumvol = pd.to_numeric(events["cum_vol_ratio_at_trigger"], errors="coerce").fillna(0.0)
    barvol = pd.to_numeric(events.get("bar_vol_ratio_at_trigger"), errors="coerce").fillna(0.0)
    dist_norm = pd.to_numeric(events.get("entry_dist_norm"), errors="coerce")
    gap_norm = pd.to_numeric(events.get("positive_gap_norm"), errors="coerce")

    standard = (
        events["entry_source"].eq("standard_breakout")
        & (leader >= params.leader_min)
        & (setup >= params.standard_setup_min)
        & (trigger >= params.standard_trigger_min)
        & (cumvol >= params.standard_cum_vol_ratio_min)
        & (barvol >= 0.0)
    )
    tight = (
        events["entry_source"].eq("tight_reversal")
        & (leader >= params.leader_min)
        & (setup >= params.tight_setup_min)
        & (trigger >= params.tight_trigger_min)
        & (cumvol >= params.tight_cum_vol_ratio_min)
        & (dist_norm <= params.tight_entry_dist_norm_max)
        & (gap_norm <= params.tight_gap_norm_max)
    )
    return standard | tight


def dedupe_events(events: pd.DataFrame) -> pd.DataFrame:
    work = events.copy()
    work["trigger_time"] = pd.to_datetime(work["trigger_time"], errors="coerce")
    return (
        work.sort_values(
            ["date", "symbol", "entry_priority_bucket", "trigger_time", "leader_score"],
            ascending=[True, True, True, True, False],
            kind="mergesort",
        )
        .drop_duplicates(["symbol", "date"], keep="first")
        .reset_index(drop=True)
    )


def prepare_daily_features(daily: pd.DataFrame) -> pd.DataFrame:
    required = {"symbol", "date", "open", "high", "low", "close", "volume"}
    missing = required - set(daily.columns)
    if missing:
        raise ValueError(f"daily missing columns: {sorted(missing)}")

    df = daily.copy()
    df["symbol"] = df["symbol"].astype(str).str.upper()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    df = df.sort_values(["symbol", "date"], kind="mergesort").reset_index(drop=True)
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce").astype(float)

    g = df.groupby("symbol", sort=False)
    df["prev_close"] = g["close"].shift(1)
    df["tr"] = np.maximum.reduce(
        [
            (df["high"] - df["low"]).to_numpy(),
            (df["high"] - df["prev_close"]).abs().fillna(0.0).to_numpy(),
            (df["low"] - df["prev_close"]).abs().fillna(0.0).to_numpy(),
        ]
    )
    df["atr20"] = g["tr"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    df["dma10"] = g["close"].rolling(10, min_periods=10).mean().reset_index(level=0, drop=True)
    df["dma21"] = g["close"].rolling(21, min_periods=21).mean().reset_index(level=0, drop=True)
    df["dma50"] = g["close"].rolling(50, min_periods=50).mean().reset_index(level=0, drop=True)

    daily_range = df["high"] - df["low"]
    df["adr20_pct"] = (
        (daily_range / df["close"].replace(0, np.nan))
        .groupby(df["symbol"], sort=False)
        .rolling(20, min_periods=20)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["avg_volume_20"] = g["volume"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    df["range_pct"] = (df["high"] - df["low"]) / df["close"].replace(0, np.nan)
    df["median_range_10"] = (
        df["range_pct"]
        .groupby(df["symbol"], sort=False)
        .rolling(10, min_periods=10)
        .median()
        .reset_index(level=0, drop=True)
    )

    df["dma10_lag3"] = g["dma10"].shift(3)
    df["dma21_lag5"] = g["dma21"].shift(5)
    ma_regime_exit = (
        (df["close"] >= df["dma10"])
        & (df["close"] >= df["dma21"])
        & (df["dma10"] > df["dma10_lag3"])
        & (df["dma21"] > df["dma21_lag5"])
    ).astype(float)

    above_dma10_5d = (
        (df["close"] >= df["dma10"]).astype(float)
        .groupby(df["symbol"], sort=False)
        .rolling(5, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    dist_dma10_atr5 = (
        ((df["close"] - df["dma10"]).abs() / df["atr20"].replace(0, np.nan))
        .groupby(df["symbol"], sort=False)
        .rolling(5, min_periods=5)
        .mean()
        .reset_index(level=0, drop=True)
    )
    deep_below_dma10_5d = (
        (df["close"] < df["dma10"] * 0.985).astype(float)
        .groupby(df["symbol"], sort=False)
        .rolling(5, min_periods=5)
        .sum()
        .reset_index(level=0, drop=True)
    )
    surf10_exit = (
        0.40 * _clip01(above_dma10_5d / 0.80)
        + 0.35 * _clip01(1.0 - (dist_dma10_atr5 / 1.00))
        + 0.25 * _clip01(1.0 - (deep_below_dma10_5d / 1.0))
    )

    above_dma21_8d = (
        (df["close"] >= df["dma21"]).astype(float)
        .groupby(df["symbol"], sort=False)
        .rolling(8, min_periods=8)
        .mean()
        .reset_index(level=0, drop=True)
    )
    dist_dma21_atr8 = (
        ((df["close"] - df["dma21"]).abs() / df["atr20"].replace(0, np.nan))
        .groupby(df["symbol"], sort=False)
        .rolling(8, min_periods=8)
        .mean()
        .reset_index(level=0, drop=True)
    )
    deep_below_dma21_8d = (
        (df["close"] < df["dma21"] * 0.98).astype(float)
        .groupby(df["symbol"], sort=False)
        .rolling(8, min_periods=8)
        .sum()
        .reset_index(level=0, drop=True)
    )
    surf21_exit = (
        0.40 * _clip01(above_dma21_8d / 0.75)
        + 0.35 * _clip01(1.0 - (dist_dma21_atr8 / 1.20))
        + 0.25 * _clip01(1.0 - (deep_below_dma21_8d / 1.5))
    )
    df["surf_quality_exit"] = np.fmax(surf10_exit, surf21_exit)

    rolling_high_10 = (
        df.groupby("symbol", sort=False)["high"]
        .rolling(10, min_periods=10)
        .max()
        .reset_index(level=0, drop=True)
    )
    drawdown_from_high10 = ((rolling_high_10 - df["close"]) / rolling_high_10.replace(0, np.nan)).clip(lower=0)
    shallow_pullback_exit = _clip01(1.0 - (drawdown_from_high10 / 0.10))
    slope10 = df["dma10"] / df["dma10_lag3"].replace(0, np.nan) - 1.0
    slope21 = df["dma21"] / df["dma21_lag5"].replace(0, np.nan) - 1.0
    slope_quality_exit = 0.60 * _clip01(slope10 / 0.03) + 0.40 * _clip01(slope21 / 0.05)
    tight_low_volume_day = (
        (df["range_pct"] < df["median_range_10"])
        & (df["volume"] < df["avg_volume_20"])
        & (df["close"] <= df["prev_close"])
    ).astype(float)
    close_pos_in_bar = (df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan)
    distribution_day = (
        (df["close"] < df["prev_close"])
        & (df["volume"] > 1.2 * df["avg_volume_20"])
        & (close_pos_in_bar < 0.35)
        & (df["range_pct"] > df["median_range_10"] * 1.05)
    ).astype(float)

    df["hold_score"] = (
        25.0 * ma_regime_exit
        + 25.0 * df["surf_quality_exit"].fillna(0.0)
        + 15.0 * shallow_pullback_exit.fillna(0.0)
        + 15.0 * slope_quality_exit.fillna(0.0)
        + 10.0 * tight_low_volume_day.fillna(0.0)
        + 10.0 * pd.Series(1.0 - distribution_day, index=df.index).fillna(0.0)
    )
    df["tight_low_volume_day"] = tight_low_volume_day.astype(bool)
    return df


def load_selected_inputs(
    standard_events: str | Path,
    tight_events: str | Path,
    daily_path: str | Path,
    params: TwoLaneParams = TwoLaneParams(),
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    events = load_two_lane_events(standard_events, tight_events)
    selected = dedupe_events(events.loc[base_entry_mask(events, params)].copy())
    min_date = selected["date"].min() - pd.Timedelta(days=260)
    max_date = selected["date"].max() + pd.Timedelta(days=180)
    symbols = set(selected["symbol"].astype(str).str.upper().unique())

    daily = pd.read_parquet(
        daily_path,
        columns=["symbol", "date", "open", "high", "low", "close", "volume"],
    )
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.loc[
        daily["symbol"].isin(symbols)
        & (daily["date"] >= min_date)
        & (daily["date"] <= max_date)
    ].copy()
    daily_prepared = prepare_daily_features(daily)
    daily_by_symbol = {
        symbol: sub.sort_values("date", kind="mergesort").reset_index(drop=True)
        for symbol, sub in daily_prepared.groupby("symbol", sort=False)
    }
    return selected, daily_by_symbol


def build_industry_rs(daily_path: str | Path, universe_path: str | Path) -> pd.DataFrame:
    universe = pd.read_parquet(universe_path, columns=["symbol", "industry"])
    universe["symbol"] = universe["symbol"].astype(str).str.upper()
    daily = pd.read_parquet(daily_path, columns=["symbol", "date", "close"])
    daily["symbol"] = daily["symbol"].astype(str).str.upper()
    daily["date"] = pd.to_datetime(daily["date"]).dt.normalize()
    daily = daily.merge(universe, on="symbol", how="left")
    daily = daily.dropna(subset=["industry"]).sort_values(["symbol", "date"], kind="mergesort")
    g = daily.groupby("symbol", sort=False)
    daily["roc_21"] = daily["close"] / g["close"].shift(21).replace(0, np.nan) - 1.0
    daily["roc_63"] = daily["close"] / g["close"].shift(63).replace(0, np.nan) - 1.0
    daily["roc_126"] = daily["close"] / g["close"].shift(126).replace(0, np.nan) - 1.0
    daily["industry_momentum"] = 0.25 * daily["roc_21"] + 0.35 * daily["roc_63"] + 0.40 * daily["roc_126"]
    industry_daily = (
        daily.groupby(["date", "industry"], as_index=False)["industry_momentum"]
        .median()
        .dropna(subset=["industry_momentum"])
    )
    industry_daily["industry_rs_pct"] = industry_daily.groupby("date")["industry_momentum"].rank(pct=True)
    return industry_daily[["date", "industry", "industry_momentum", "industry_rs_pct"]]


def add_priority_features(events: pd.DataFrame, daily_path: str | Path, universe_path: str | Path) -> pd.DataFrame:
    out = events.copy()
    universe = pd.read_parquet(universe_path)
    universe["symbol"] = universe["symbol"].astype(str).str.upper()
    keep_cols = ["symbol", "market_cap", "sector", "industry", "rank_market_cap"]
    out = out.merge(universe[keep_cols], on="symbol", how="left")
    out = out.merge(build_industry_rs(daily_path, universe_path), on=["date", "industry"], how="left")

    leader = pd.to_numeric(out["leader_score"], errors="coerce")
    setup = pd.to_numeric(out["setup_score_pre"], errors="coerce")
    trigger = pd.to_numeric(out["trigger_score"], errors="coerce")
    cumvol = pd.to_numeric(out["cum_vol_ratio_at_trigger"], errors="coerce")
    barvol = pd.to_numeric(out["bar_vol_ratio_at_trigger"], errors="coerce")
    move = pd.to_numeric(out["move_from_open_at_trigger"], errors="coerce")
    rs = pd.to_numeric(out["industry_rs_pct"], errors="coerce").fillna(0.5)
    market_cap = pd.to_numeric(out["market_cap"], errors="coerce")

    cap_rank = market_cap.rank(pct=True, ascending=True)
    small_cap_score = (1.0 - cap_rank).fillna(0.5)
    leader98 = (leader >= 98.0).astype(float)
    leader_score_norm = _clip01((leader - 94.0) / 6.0)
    volume_thrust = _clip01(np.log1p(cumvol.fillna(0.0)) / np.log1p(5.0))
    bar_volume_thrust = _clip01(np.log1p(barvol.fillna(0.0)) / np.log1p(5.0))
    move_thrust = _clip01((move.fillna(0.0) - 0.02) / 0.08)
    setup_sweet = _clip01(1.0 - (setup - 74.0).abs() / 8.0)
    standard_bonus = out["entry_source"].eq("standard_breakout").astype(float)

    out["priority_leader98"] = leader98
    out["priority_leader_score_norm"] = leader_score_norm
    out["priority_volume_thrust"] = volume_thrust
    out["priority_bar_volume_thrust"] = bar_volume_thrust
    out["priority_move_thrust"] = move_thrust
    out["priority_setup_sweet"] = setup_sweet
    out["priority_industry_rs"] = rs
    out["priority_small_cap"] = small_cap_score
    out["priority_standard_bonus"] = standard_bonus
    out["same_day_priority_score"] = (
        24.0 * leader98
        + 16.0 * leader_score_norm
        + 20.0 * volume_thrust
        + 10.0 * bar_volume_thrust
        + 18.0 * move_thrust
        + 14.0 * rs
        + 8.0 * setup_sweet
        + 5.0 * small_cap_score
        + 3.0 * standard_bonus
    )
    return out


def sort_by_industry_composite(events: pd.DataFrame) -> pd.DataFrame:
    return (
        events.sort_values(
            ["date", "same_day_priority_score", "trigger_time", "symbol"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .reset_index(drop=True)
    )


def is_loose_a_plus_candidate(event: pd.Series) -> bool:
    leader98 = float(event.get("priority_leader98", 0.0)) >= 1.0
    volume = float(event.get("priority_volume_thrust", 0.0))
    move = float(event.get("priority_move_thrust", 0.0))
    industry_rs = float(event.get("priority_industry_rs", 0.5))
    setup_sweet = float(event.get("priority_setup_sweet", 0.0))
    return leader98 and industry_rs >= 0.70 and setup_sweet >= 0.65 and (volume >= 0.75 or move >= 0.50)


def simulate_super_winner_trade(
    event: pd.Series,
    daily_by_symbol: dict[str, pd.DataFrame],
    *,
    use_atr_trail: bool = True,
) -> tuple[dict[str, Any], pd.DataFrame]:
    symbol = str(event["symbol"])
    entry_date = pd.Timestamp(event["date"]).normalize()
    daily = daily_by_symbol.get(symbol)
    if daily is None or daily.empty:
        return {"symbol": symbol, "entry_date": entry_date, "exit_reason": "missing_daily", "trade_return_pct": np.nan}, pd.DataFrame()

    rows = daily.loc[daily["date"] >= entry_date].reset_index(drop=True)
    if rows.empty:
        return {"symbol": symbol, "entry_date": entry_date, "exit_reason": "missing_entry", "trade_return_pct": np.nan}, pd.DataFrame()

    day0 = rows.iloc[0]
    entry_price = _num(event, "trigger_close", "close", default=float(day0["close"]))
    pivot = _num(event, "effective_pivot_level", "pivot_high", "zigzag_line_value", default=float(day0["low"]))
    initial_stop = min(float(day0["low"]), pivot) * 0.999
    risk_per_share = entry_price - initial_stop
    if not np.isfinite(entry_price) or entry_price <= 0 or not np.isfinite(risk_per_share) or risk_per_share <= 0:
        return {"symbol": symbol, "entry_date": entry_date, "exit_reason": "invalid_entry", "trade_return_pct": np.nan}, pd.DataFrame()

    leader_score = _num(event, "leader_score", default=np.nan)
    realized = 0.0
    remaining = 1.0
    partial_taken = False
    super_winner = False
    super_winner_date = pd.NaT
    reduced_21dma = False
    consecutive_21dma_breaks = 0
    highest_close = float(day0["close"])
    exit_reason = "open_end"
    exit_date = pd.NaT
    marks: list[dict[str, Any]] = []

    def add_mark(date: pd.Timestamp, factor: float, reason: str | None = None) -> None:
        marks.append(
            {
                "date": pd.Timestamp(date).normalize(),
                "equity_factor": factor,
                "reason": reason,
                "super_winner": super_winner,
                "remaining": remaining,
            }
        )

    if float(day0["close"]) < pivot:
        ret = float(day0["close"]) / entry_price - 1.0
        add_mark(entry_date, 1.0 + ret, "day0_pivot_fail")
        return (
            {
                "symbol": symbol,
                "entry_date": entry_date,
                "exit_date": entry_date,
                "entry_source": event.get("entry_source"),
                "exit_reason": "day0_pivot_fail",
                "trade_return_pct": ret,
                "partial_taken": False,
                "super_winner": False,
                "super_winner_date": pd.NaT,
            },
            pd.DataFrame(marks),
        )

    add_mark(entry_date, _equity_factor(entry_price, realized, remaining, float(day0["close"])))

    for session_idx in range(1, len(rows)):
        row = rows.iloc[session_idx]
        cur_date = pd.Timestamp(row["date"]).normalize()
        days_since_entry = int((cur_date - entry_date).days)
        low = float(row["low"])
        high = float(row["high"])
        close = float(row["close"])
        highest_close = max(highest_close, close)
        hold_score = float(row["hold_score"]) if pd.notna(row.get("hold_score", np.nan)) else np.nan
        dma10 = float(row["dma10"]) if pd.notna(row.get("dma10", np.nan)) else np.nan
        dma21 = float(row["dma21"]) if pd.notna(row.get("dma21", np.nan)) else np.nan
        dma50 = float(row["dma50"]) if pd.notna(row.get("dma50", np.nan)) else np.nan
        atr20 = float(row["atr20"]) if pd.notna(row.get("atr20", np.nan)) else np.nan

        if low <= initial_stop and remaining > 0:
            realized += remaining * (initial_stop / entry_price - 1.0)
            remaining = 0.0
            exit_reason = "hard_stop_lod"
            exit_date = cur_date
            add_mark(cur_date, 1.0 + realized, exit_reason)
            break

        if days_since_entry <= 1 and close < pivot and remaining > 0:
            realized += remaining * (close / entry_price - 1.0)
            remaining = 0.0
            exit_reason = "pivot_fail_exit_all"
            exit_date = cur_date
            add_mark(cur_date, 1.0 + realized, exit_reason)
            break

        if not partial_taken and remaining > 0:
            tp_price = min(entry_price * 1.10, entry_price + 1.75 * risk_per_share)
            if high >= tp_price:
                sell_frac = min(remaining, 0.33)
                realized += sell_frac * (tp_price / entry_price - 1.0)
                remaining -= sell_frac
                partial_taken = True

        close_gain = close / entry_price - 1.0
        peak_gain = highest_close / entry_price - 1.0
        leader_fast_track = pd.notna(leader_score) and leader_score >= 98.0 and close_gain >= 0.35
        if (
            not super_winner
            and remaining > 0
            and (
                peak_gain >= 1.0
                or (close_gain >= 0.50 and pd.notna(dma21) and close > dma21)
                or (leader_fast_track and pd.notna(dma21) and close > dma21)
            )
        ):
            super_winner = True
            super_winner_date = cur_date

        dma_allowed = session_idx >= 10
        if super_winner and dma_allowed and remaining > 0:
            if pd.notna(dma21) and close < dma21:
                consecutive_21dma_breaks += 1
            else:
                consecutive_21dma_breaks = 0

            if consecutive_21dma_breaks >= 2 and pd.notna(hold_score) and hold_score < 40.0 and not reduced_21dma:
                target = 0.50
                sell_frac = max(0.0, remaining - target)
                realized += sell_frac * (close / entry_price - 1.0)
                remaining -= sell_frac
                reduced_21dma = True

            active_atr_mult = 3.0
            if peak_gain >= 2.0:
                active_atr_mult = 2.0
            if peak_gain >= 3.0:
                active_atr_mult = 1.5

            hit_50dma = pd.notna(dma50) and close < dma50
            hit_atr = use_atr_trail and pd.notna(atr20) and close < highest_close - active_atr_mult * atr20
            if hit_50dma or hit_atr:
                realized += remaining * (close / entry_price - 1.0)
                remaining = 0.0
                exit_reason = "super_winner_50dma" if hit_50dma else f"super_winner_{active_atr_mult:g}atr"
                exit_date = cur_date
                add_mark(cur_date, 1.0 + realized, exit_reason)
                break

            add_mark(cur_date, _equity_factor(entry_price, realized, remaining, close))
            continue

        if dma_allowed and remaining > 0:
            if pd.notna(dma21) and close < dma21:
                if pd.notna(hold_score) and hold_score < 45.0:
                    target = 0.15
                    sell_frac = max(0.0, remaining - target)
                    realized += sell_frac * (close / entry_price - 1.0)
                    remaining -= sell_frac
                    reduced_21dma = True
                elif reduced_21dma and pd.notna(hold_score) and hold_score < 40.0:
                    realized += remaining * (close / entry_price - 1.0)
                    remaining = 0.0
                    exit_reason = "dma21_exit_after_reduce"
                    exit_date = cur_date
                    add_mark(cur_date, 1.0 + realized, exit_reason)
                    break

            if pd.notna(dma10) and close < dma10:
                if pd.notna(hold_score) and hold_score < 50.0:
                    realized += remaining * (close / entry_price - 1.0)
                    remaining = 0.0
                    exit_reason = "dma10_holdscore_exit_all"
                    exit_date = cur_date
                    add_mark(cur_date, 1.0 + realized, exit_reason)
                    break
                if pd.notna(hold_score) and hold_score < 55.0:
                    target = 0.25
                    sell_frac = max(0.0, remaining - target)
                    realized += sell_frac * (close / entry_price - 1.0)
                    remaining -= sell_frac

        add_mark(cur_date, _equity_factor(entry_price, realized, remaining, close))

    if remaining > 0:
        last = rows.iloc[-1]
        exit_date = pd.Timestamp(last["date"]).normalize()
        realized += remaining * (float(last["close"]) / entry_price - 1.0)
        add_mark(exit_date, 1.0 + realized, exit_reason)

    return (
        {
            "symbol": symbol,
            "entry_date": entry_date,
            "exit_date": exit_date,
            "entry_source": event.get("entry_source"),
            "exit_reason": exit_reason,
            "trade_return_pct": realized,
            "partial_taken": partial_taken,
            "super_winner": super_winner,
            "super_winner_date": super_winner_date,
        },
        pd.DataFrame(marks).drop_duplicates("date", keep="last"),
    )


def simulate_events(
    events: pd.DataFrame,
    daily_by_symbol: dict[str, pd.DataFrame],
) -> list[tuple[int, pd.Series, dict[str, Any], pd.DataFrame]]:
    simulations = []
    for event_id, event in events.reset_index(drop=True).iterrows():
        trade, path = simulate_super_winner_trade(event, daily_by_symbol)
        trade["event_id"] = event_id
        if not path.empty:
            path = path.copy()
            path["symbol"] = event["symbol"]
            path["event_id"] = event_id
        simulations.append((event_id, event, trade, path))
    return simulations


def run_portfolio_with_replacement(
    simulations: list[tuple[int, pd.Series, dict[str, Any], pd.DataFrame]],
    commission_each_side: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    signal_dates = {pd.Timestamp(event["date"]).normalize() for _, event, _, _ in simulations}
    path_dates: set[pd.Timestamp] = set()
    for _, _, _, path in simulations:
        if not path.empty:
            path_dates.update(pd.to_datetime(path["date"]).dt.normalize().tolist())
    all_dates = sorted(signal_dates | {pd.Timestamp(d).normalize() for d in path_dates})
    sims_by_date: dict[pd.Timestamp, list[tuple[int, pd.Series, dict[str, Any], pd.DataFrame]]] = {}
    for item in simulations:
        _, event, _, _ = item
        sims_by_date.setdefault(pd.Timestamp(event["date"]).normalize(), []).append(item)

    cash = INITIAL_EQUITY
    open_positions: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    trades: list[dict[str, Any]] = []
    total_commission = 0.0

    for cur_date in all_dates:
        survivors = []
        for pos in open_positions:
            factor = pos["factor_by_date"].get(cur_date)
            if factor is not None:
                pos["last_factor"] = factor
            if cur_date >= pos["exit_date"]:
                gross_exit_value = pos["allocation"] * pos["last_factor"]
                exit_commission = gross_exit_value * commission_each_side
                cash += gross_exit_value - exit_commission
                total_commission += exit_commission
                trade = dict(pos["trade"])
                trade["exit_reason"] = trade.get("exit_reason", "natural_exit")
                trade["entry_commission"] = pos["entry_commission"]
                trade["exit_commission"] = exit_commission
                trade["total_commission"] = pos["entry_commission"] + exit_commission
                trade["replaced"] = False
                trades.append(trade)
            else:
                survivors.append(pos)
        open_positions = survivors
        held_symbols = {pos["symbol"] for pos in open_positions}

        for _, event, trade, path in sims_by_date.get(cur_date, []):
            symbol = str(event["symbol"])
            if symbol in held_symbols or path.empty or pd.isna(trade.get("trade_return_pct")):
                continue

            current_position_value = sum(pos["allocation"] * pos["last_factor"] for pos in open_positions)
            current_equity = cash + current_position_value
            desired_allocation = current_equity * SLOT_ALLOC_PCT

            if (len(open_positions) >= MAX_SLOTS or cash < desired_allocation) and is_loose_a_plus_candidate(event):
                replaceable = []
                candidate_score = float(event.get("same_day_priority_score", 0.0))
                for idx, pos in enumerate(open_positions):
                    current_gain = pos["last_factor"] - 1.0
                    if pos.get("a_plus_candidate", False) and current_gain > 0.10:
                        continue
                    if current_gain >= 0.50:
                        continue
                    replacement_score = pos["priority_score"] + 40.0 * max(current_gain, 0.0) - 20.0 * max(-current_gain, 0.0)
                    replaceable.append((replacement_score, idx))
                if replaceable:
                    replacement_score, replace_idx = min(replaceable, key=lambda item: item[0])
                    if candidate_score >= replacement_score + 18.0:
                        pos = open_positions.pop(replace_idx)
                        gross_exit_value = pos["allocation"] * pos["last_factor"]
                        exit_commission = gross_exit_value * commission_each_side
                        cash += gross_exit_value - exit_commission
                        total_commission += exit_commission
                        replaced_trade = dict(pos["trade"])
                        replaced_trade["exit_date"] = cur_date
                        replaced_trade["exit_reason"] = "priority_replacement_exit"
                        replaced_trade["trade_return_pct"] = pos["last_factor"] - 1.0
                        replaced_trade["entry_commission"] = pos["entry_commission"]
                        replaced_trade["exit_commission"] = exit_commission
                        replaced_trade["total_commission"] = pos["entry_commission"] + exit_commission
                        replaced_trade["replaced"] = True
                        trades.append(replaced_trade)
                        held_symbols = {p["symbol"] for p in open_positions}

            if len(open_positions) >= MAX_SLOTS:
                continue

            current_position_value = sum(pos["allocation"] * pos["last_factor"] for pos in open_positions)
            current_equity = cash + current_position_value
            gross_allocation = min(cash, current_equity * SLOT_ALLOC_PCT)
            if gross_allocation <= 0:
                continue
            entry_commission = gross_allocation * commission_each_side
            allocation = gross_allocation - entry_commission
            if allocation <= 0:
                continue
            factor_by_date = {
                pd.Timestamp(row.date).normalize(): float(row.equity_factor)
                for row in path.itertuples(index=False)
            }
            entry_factor = factor_by_date.get(cur_date, 1.0)
            exit_date = pd.Timestamp(path["date"].max()).normalize()
            cash -= gross_allocation
            total_commission += entry_commission

            if exit_date <= cur_date:
                gross_exit_value = allocation * entry_factor
                exit_commission = gross_exit_value * commission_each_side
                cash += gross_exit_value - exit_commission
                total_commission += exit_commission
                closed_trade = dict(trade)
                closed_trade["entry_commission"] = entry_commission
                closed_trade["exit_commission"] = exit_commission
                closed_trade["total_commission"] = entry_commission + exit_commission
                closed_trade["replaced"] = False
                trades.append(closed_trade)
            else:
                open_positions.append(
                    {
                        "symbol": symbol,
                        "allocation": allocation,
                        "factor_by_date": factor_by_date,
                        "last_factor": entry_factor,
                        "exit_date": exit_date,
                        "trade": trade,
                        "entry_commission": entry_commission,
                        "priority_score": float(event.get("same_day_priority_score", 0.0)),
                        "a_plus_candidate": is_loose_a_plus_candidate(event),
                    }
                )
                held_symbols.add(symbol)

        equity = cash + sum(pos["allocation"] * pos["last_factor"] for pos in open_positions)
        rows.append(
            {
                "date": cur_date,
                "equity": equity,
                "cash": cash,
                "open_positions": len(open_positions),
                "total_commission": total_commission,
            }
        )

    curve = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)
    curve["date"] = pd.to_datetime(curve["date"])
    curve["return_pct"] = curve["equity"] / INITIAL_EQUITY - 1.0
    curve["peak"] = curve["equity"].cummax()
    curve["drawdown"] = curve["equity"] / curve["peak"] - 1.0
    return curve, pd.DataFrame(trades), total_commission


def summarize(curve: pd.DataFrame, trades: pd.DataFrame, commission_each_side: float, total_commission: float) -> dict[str, object]:
    eras_0107 = trades[
        trades["symbol"].astype(str).eq("ERAS")
        & pd.to_datetime(trades["entry_date"]).dt.normalize().eq(pd.Timestamp("2026-01-07"))
    ]
    return {
        "strategy": "industry_composite_replacement",
        "commission_each_side": commission_each_side,
        "end_equity": float(curve["equity"].iloc[-1]),
        "total_return": float(curve["equity"].iloc[-1] / INITIAL_EQUITY - 1.0),
        "max_drawdown": float(curve["drawdown"].min()),
        "max_drawdown_date": curve.loc[curve["drawdown"].idxmin(), "date"],
        "trades": int(len(trades)),
        "win_rate": float((trades["trade_return_pct"] > 0).mean()) if not trades.empty else np.nan,
        "replacements": int(trades.get("replaced", pd.Series(dtype=bool)).fillna(False).astype(bool).sum()) if not trades.empty else 0,
        "eras_20260107_entered": bool(not eras_0107.empty),
        "eras_20260107_return": float(eras_0107["trade_return_pct"].iloc[0]) if not eras_0107.empty else np.nan,
        "total_commission": float(total_commission),
    }
