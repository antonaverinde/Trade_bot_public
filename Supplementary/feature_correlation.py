"""
Pipeline Feature Correlation Report
===================================
Finds highly correlated default ForexPipeline features for a configured pair,
year, timeframe, and normalization setup.

Default use case:
    EURUSD, 2023, H1, fracdiff d=0.3

Outputs are written to Supplementary/feature_correlation_outputs/:
    - high_corr_pairs.csv
    - corr_groups.csv
    - corr_matrix.csv
    - corr_heatmap.html
    - top_pairs_bar.html

Usage:
    source ~/Trade_bot/.venv/bin/activate
    python Supplementary/feature_correlation.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def _find_root() -> Path:
    """Find the Trade_bot project root by looking for histdata/ + Pipeline/."""
    candidates: list[Path] = []
    try:
        candidates.append(Path(__file__).resolve().parent.parent)
    except NameError:
        pass
    p = Path.cwd()
    for _ in range(6):
        candidates.append(p)
        p = p.parent
    for c in candidates:
        if (c / "histdata").is_dir() and (c / "Pipeline").is_dir():
            return c
    raise RuntimeError(
        "Cannot locate Trade_bot project root. "
        "Run from inside the Trade_bot directory or set ROOT manually."
    )


ROOT = _find_root()
sys.path.insert(0, str(ROOT))

from Pipeline.pipeline import ForexDataLoader, ForexPipeline  # noqa: E402

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------

PAIR = "EURUSD"
YEARS = [2023]
TIMEFRAME = "H1"
WEEKENDS = "filled"
NORM_METHOD = "fracdiff"
FRACDIFF_D = 0.3
THRESHOLD = 6e-4
SCALING = "none"
TARGET_TYPE = "lag"
TARGET_HORIZONS = [1, 5, 15]
LAGS = [1, 2, 5, 10]
GAP_BARS = 50

CORR_METHOD = "pearson"  # "pearson" | "spearman" | "kendall"
HIGH_CORR_THRESHOLD = 0.85
TOP_N_PAIRS = 30
NEAR_CONSTANT_NUNIQUE = 1

OUT_DIR = ROOT / "Supplementary" / "feature_correlation_outputs"


# -----------------------------------------------------------------------------
# Pipeline data
# -----------------------------------------------------------------------------

def load_pipeline_results(
    pair: str = PAIR,
    years: list[int] | None = YEARS,
    timeframe: str = TIMEFRAME,
    weekends: str = WEEKENDS,
    norm_method: str = NORM_METHOD,
    fracdiff_d: float = FRACDIFF_D,
    threshold: float = THRESHOLD,
    scaling: str = SCALING,
    target_type: str = TARGET_TYPE,
) -> dict:
    """Load M1 data and run ForexPipeline for the configured feature frame."""
    loader = ForexDataLoader()
    df_m1 = loader.load_and_merge(
        ROOT / "histdata", pair=pair, years=years, weekends=weekends
    )
    pipeline = ForexPipeline(
        lags=LAGS,
        target_horizons=TARGET_HORIZONS,
        gap_bars=GAP_BARS,
        scaling=scaling,
        norm_method=norm_method,
        fracdiff_d=fracdiff_d,
        target_type=target_type,
        threshold=threshold,
        weekends=weekends,
    )
    return pipeline.run(df_m1, timeframe=timeframe)


def build_feature_frame(results: dict) -> pd.DataFrame:
    """Return one unscaled model-eligible feature frame from train/val/test splits."""
    feature_cols = results["feature_cols"]
    splits = [results["train_raw"], results["val_raw"], results["test_raw"]]
    df = pd.concat([split[feature_cols] for split in splits], axis=0).sort_index()
    df = df.replace([np.inf, -np.inf], np.nan).dropna(axis=0, how="any")
    return df


# -----------------------------------------------------------------------------
# Correlation analysis
# -----------------------------------------------------------------------------

def find_near_constant_columns(
    df: pd.DataFrame, max_nunique: int = NEAR_CONSTANT_NUNIQUE
) -> list[str]:
    """Columns with too few distinct values to produce useful correlations."""
    return [col for col in df.columns if df[col].nunique(dropna=True) <= max_nunique]


def compute_correlation(
    df: pd.DataFrame,
    method: str = CORR_METHOD,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """
    Compute feature correlation after excluding constant columns.

    Returns
    -------
    corr : full correlation matrix
    df_corr : feature frame used for correlation
    excluded : near-constant columns excluded before correlation
    """
    excluded = find_near_constant_columns(df)
    df_corr = df.drop(columns=excluded) if excluded else df.copy()
    corr = df_corr.corr(method=method)
    return corr, df_corr, excluded


def infer_family(feature: str) -> str:
    """Small naming heuristic for grouping related pipeline features."""
    if feature.startswith("rsi_"):
        return "rsi"
    if feature in {"adx_14", "di_diff", "adx_delta"}:
        return "adx"
    if feature.startswith("dist_ema"):
        return "ma_distance"
    if feature.startswith("atr"):
        return "atr"
    if feature.startswith("bb_"):
        return "bollinger"
    if feature.startswith("hour_") or feature.startswith("dow_") or feature.startswith("is_"):
        return "time"
    if feature in {"body_ratio", "shadow_ratio", "body_gap"}:
        return "candle"
    if feature.startswith("ret_") or feature.startswith("vol_"):
        return "distribution"
    if feature.startswith("close_lag"):
        return "lags"
    if feature in {"open", "high", "low", "close", "volume"}:
        return "ohlcv"
    return "other"


def high_corr_pairs(
    corr: pd.DataFrame,
    threshold: float = HIGH_CORR_THRESHOLD,
) -> pd.DataFrame:
    """Return all unique feature pairs with abs(correlation) above threshold."""
    cols = list(corr.columns)
    rows = []
    for i, left in enumerate(cols):
        for right in cols[i + 1:]:
            value = corr.loc[left, right]
            if pd.isna(value):
                continue
            abs_value = abs(float(value))
            if abs_value >= threshold:
                rows.append(
                    {
                        "feature_1": left,
                        "feature_2": right,
                        "corr": float(value),
                        "abs_corr": abs_value,
                        "family_1": infer_family(left),
                        "family_2": infer_family(right),
                    }
                )
    return pd.DataFrame(rows).sort_values(
        "abs_corr", ascending=False, ignore_index=True
    ) if rows else pd.DataFrame(
        columns=["feature_1", "feature_2", "corr", "abs_corr", "family_1", "family_2"]
    )


def _connected_components(nodes: Iterable[str], edges: Iterable[tuple[str, str]]) -> list[list[str]]:
    adjacency = {node: set() for node in nodes}
    for left, right in edges:
        adjacency[left].add(right)
        adjacency[right].add(left)

    seen: set[str] = set()
    groups: list[list[str]] = []
    for node in adjacency:
        if node in seen or not adjacency[node]:
            continue
        stack = [node]
        group = []
        seen.add(node)
        while stack:
            cur = stack.pop()
            group.append(cur)
            for nxt in adjacency[cur]:
                if nxt not in seen:
                    seen.add(nxt)
                    stack.append(nxt)
        groups.append(sorted(group))
    return sorted(groups, key=lambda g: (-len(g), g[0]))


def corr_groups(pairs: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """Build connected groups from high-correlation feature-pair edges."""
    if pairs.empty:
        return pd.DataFrame(columns=["group_id", "n_features", "families", "features"])

    edges = zip(pairs["feature_1"], pairs["feature_2"])
    groups = _connected_components(features, edges)
    rows = []
    for idx, group in enumerate(groups, 1):
        families = sorted({infer_family(feature) for feature in group})
        rows.append(
            {
                "group_id": idx,
                "n_features": len(group),
                "families": ", ".join(families),
                "features": ", ".join(group),
            }
        )
    return pd.DataFrame(rows)


def order_features_by_groups(corr: pd.DataFrame, groups: pd.DataFrame) -> list[str]:
    """Place highly correlated groups first, then remaining features alphabetically."""
    ordered: list[str] = []
    if not groups.empty:
        for features in groups["features"]:
            ordered.extend([f.strip() for f in features.split(",") if f.strip()])
    remaining = [feature for feature in corr.columns if feature not in ordered]
    return ordered + sorted(remaining)


# -----------------------------------------------------------------------------
# Plots and output
# -----------------------------------------------------------------------------

def plot_corr_heatmap(
    corr: pd.DataFrame,
    ordered_features: list[str],
    title: str,
) -> go.Figure:
    ordered_corr = corr.loc[ordered_features, ordered_features]
    fig = px.imshow(
        ordered_corr,
        zmin=-1,
        zmax=1,
        color_continuous_scale="RdBu_r",
        aspect="auto",
        title=title,
    )
    fig.update_layout(
        template="plotly_white",
        height=max(700, 24 * len(ordered_features)),
        width=max(900, 24 * len(ordered_features)),
        margin=dict(l=180, r=40, t=80, b=160),
    )
    fig.update_xaxes(tickangle=45, tickfont=dict(size=10))
    fig.update_yaxes(tickfont=dict(size=10))
    return fig


def plot_top_pairs(pairs: pd.DataFrame, top_n: int = TOP_N_PAIRS) -> go.Figure:
    top = pairs.head(top_n).copy()
    if top.empty:
        fig = go.Figure()
        fig.update_layout(title="No feature pairs above threshold")
        return fig

    top["pair"] = top["feature_1"] + "  |  " + top["feature_2"]
    fig = px.bar(
        top.sort_values("abs_corr", ascending=True),
        x="abs_corr",
        y="pair",
        color="corr",
        color_continuous_scale="RdBu_r",
        range_color=[-1, 1],
        orientation="h",
        hover_data=["corr", "family_1", "family_2"],
        title=f"Top {min(top_n, len(top))} absolute feature correlations",
    )
    fig.update_layout(
        template="plotly_white",
        height=max(500, 24 * len(top)),
        margin=dict(l=230, r=40, t=80, b=50),
    )
    return fig


def save_outputs(
    corr: pd.DataFrame,
    pairs: pd.DataFrame,
    groups: pd.DataFrame,
    heatmap: go.Figure,
    top_pairs_plot: go.Figure,
    out_dir: Path = OUT_DIR,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    pairs.to_csv(out_dir / "high_corr_pairs.csv", index=False)
    groups.to_csv(out_dir / "corr_groups.csv", index=False)
    corr.to_csv(out_dir / "corr_matrix.csv")
    heatmap.write_html(out_dir / "corr_heatmap.html", include_plotlyjs="cdn")
    top_pairs_plot.write_html(out_dir / "top_pairs_bar.html", include_plotlyjs="cdn")


def run_report(
    pair: str = PAIR,
    years: list[int] | None = YEARS,
    timeframe: str = TIMEFRAME,
    weekends: str = WEEKENDS,
    norm_method: str = NORM_METHOD,
    fracdiff_d: float = FRACDIFF_D,
    threshold: float = THRESHOLD,
    corr_method: str = CORR_METHOD,
    high_corr_threshold: float = HIGH_CORR_THRESHOLD,
    out_dir: Path = OUT_DIR,
    save: bool = True,
) -> dict:
    """Run the complete correlation report and optionally persist outputs."""
    results = load_pipeline_results(
        pair=pair,
        years=years,
        timeframe=timeframe,
        weekends=weekends,
        norm_method=norm_method,
        fracdiff_d=fracdiff_d,
        threshold=threshold,
    )
    features = build_feature_frame(results)
    corr, features_used, excluded = compute_correlation(features, method=corr_method)
    pairs = high_corr_pairs(corr, threshold=high_corr_threshold)
    groups = corr_groups(pairs, list(corr.columns))
    ordered_features = order_features_by_groups(corr, groups)

    years_label = "all" if years is None else ",".join(map(str, years))
    title = (
        f"{pair} {timeframe} {years_label} feature correlation "
        f"({norm_method}, d={fracdiff_d}, {corr_method})"
    )
    heatmap = plot_corr_heatmap(corr, ordered_features, title)
    top_pairs_plot = plot_top_pairs(pairs)

    if save:
        save_outputs(corr, pairs, groups, heatmap, top_pairs_plot, out_dir=out_dir)

    return {
        "results": results,
        "features": features,
        "features_used": features_used,
        "excluded_columns": excluded,
        "corr": corr,
        "pairs": pairs,
        "groups": groups,
        "heatmap": heatmap,
        "top_pairs_plot": top_pairs_plot,
        "out_dir": out_dir,
    }


def print_summary(report: dict, top_n: int = TOP_N_PAIRS) -> None:
    features = report["features_used"]
    pairs = report["pairs"]
    groups = report["groups"]
    excluded = report["excluded_columns"]

    print("\n[Feature correlation report]")
    print(f"Rows analyzed     : {len(features):,}")
    print(f"Features analyzed : {features.shape[1]:,}")
    print(f"Excluded constants: {excluded if excluded else 'none'}")
    print(f"High-corr pairs   : {len(pairs):,}")
    print(f"High-corr groups  : {len(groups):,}")
    print(f"Output directory  : {report['out_dir']}")

    print(f"\nTop {min(top_n, len(pairs))} correlated pairs:")
    if pairs.empty:
        print("  none above threshold")
    else:
        print(pairs.head(top_n).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print("\nCorrelation groups:")
    if groups.empty:
        print("  none")
    else:
        print(groups.to_string(index=False))


if __name__ == "__main__":
    report = run_report()
    print_summary(report)
