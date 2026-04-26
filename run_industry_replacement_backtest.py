#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import FuncFormatter

from core.industry_replacement import (
    COMMISSION_EACH_SIDE,
    INITIAL_EQUITY,
    add_priority_features,
    load_selected_inputs,
    run_portfolio_with_replacement,
    simulate_events,
    sort_by_industry_composite,
    summarize,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Backtest the Qullamaggie-style industry composite + replacement portfolio layer."
    )
    parser.add_argument("--standard-events", required=True, help="CSV with standard_breakout events and 5m trigger fields")
    parser.add_argument("--tight-events", required=True, help="CSV with tight_reversal events and 5m trigger fields")
    parser.add_argument("--daily", required=True, help="Parquet daily OHLCV history, ideally 10y for industry RS")
    parser.add_argument("--universe", required=True, help="Parquet universe with market_cap/sector/industry")
    parser.add_argument("--outdir", default="reports/industry_replacement_backtest", help="Output directory")
    parser.add_argument(
        "--commission-each-side",
        type=float,
        nargs="*",
        default=[0.0, COMMISSION_EACH_SIDE],
        help="One-way commission rates to test. Default: 0.0 0.002",
    )
    return parser.parse_args()


def plot_curves(curves: dict[str, pd.DataFrame], summary: pd.DataFrame, out_path: Path) -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True, gridspec_kw={"height_ratios": [3, 1]})
    colors = {"no_fee": "#d7301f", "fee_0p2": "#0b4ea2"}
    for name, curve in curves.items():
        ax1.plot(curve["date"], curve["equity"], color=colors.get(name, "#333333"), linewidth=2.0, label=name)
        ax2.plot(curve["date"], curve["drawdown"], color=colors.get(name, "#333333"), linewidth=1.4, label=name)

    ax1.axhline(INITIAL_EQUITY, color="#666666", linestyle="--", linewidth=1)
    ax1.set_title("Industry Composite + Replacement Backtest", fontsize=14, weight="bold")
    ax1.set_ylabel("Equity (USD)")
    ax1.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"${x:,.0f}"))
    ax1.legend(loc="upper left", fontsize=9)

    lines = []
    for name in curves:
        row = summary.loc[summary["run_name"].eq(name)].iloc[0]
        eras = "ERAS in" if bool(row.eras_20260107_entered) else "ERAS out"
        lines.append(f"{name}: ${row.end_equity:,.0f} / {row.total_return:+.2%} / MaxDD {row.max_drawdown:.2%} / {eras}")
    ax1.text(
        0.01,
        0.78,
        "\n".join(lines),
        transform=ax1.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox=dict(facecolor="white", edgecolor="#dddddd", alpha=0.9),
    )
    ax2.axhline(0, color="#666666", linewidth=0.8)
    ax2.set_ylabel("Drawdown")
    ax2.set_xlabel("Date")
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f"{x:.0%}"))
    fig.tight_layout()
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def run() -> None:
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    selected, daily_by_symbol = load_selected_inputs(args.standard_events, args.tight_events, args.daily)
    selected = add_priority_features(selected, args.daily, args.universe)
    selected = sort_by_industry_composite(selected)
    selected.to_csv(outdir / "selected_signals_with_industry_priority.csv", index=False)
    selected.loc[selected["date"].eq(pd.Timestamp("2026-01-07"))].to_csv(outdir / "jan7_2026_candidates_ranked.csv", index=False)

    simulations = simulate_events(selected, daily_by_symbol)
    rows = []
    curves: dict[str, pd.DataFrame] = {}
    for commission in args.commission_each_side:
        curve, trades, total_commission = run_portfolio_with_replacement(simulations, commission)
        run_name = "no_fee" if commission == 0 else f"fee_{commission:.4f}".replace(".", "p")
        if abs(commission - COMMISSION_EACH_SIDE) < 1e-12:
            run_name = "fee_0p2"

        curve.to_csv(outdir / f"equity_{run_name}.csv", index=False)
        trades.to_csv(outdir / f"trades_{run_name}.csv", index=False)
        trades.sort_values("trade_return_pct", ascending=False).head(30).to_csv(outdir / f"top30_trades_{run_name}.csv", index=False)
        row = summarize(curve, trades, commission, total_commission)
        row["run_name"] = run_name
        row["selected_signal_count"] = int(len(selected))
        rows.append(row)
        curves[run_name] = curve

    summary = pd.DataFrame(rows)
    summary.to_csv(outdir / "industry_replacement_summary.csv", index=False)
    plot_curves(curves, summary, outdir / "industry_replacement_equity.png")
    print(summary.to_string(index=False))
    print(f"Wrote {outdir}")


if __name__ == "__main__":
    run()
