"""Render factor and industry exposure diagnostics from rolling outputs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .errors import InputDataError
from .io import normalize_date


_REQUIRED_COLUMNS = {
    "date",
    "exposure_type",
    "name",
    "benchmark_exposure",
    "portfolio_exposure",
    "active_exposure",
}


@dataclass(frozen=True)
class ExposureReportConfig:
    exposure_file: Path
    output_dir: Path
    industry_frequency: str = "monthly"
    title_prefix: str = "Portfolio exposure"
    portfolio: str = "risk_optimized"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exposures(path: Path) -> pd.DataFrame:
    source = path.expanduser().resolve()
    if not source.is_file():
        raise InputDataError(f"exposure file does not exist: {source}")
    try:
        frame = pd.read_parquet(source)
    except Exception as exc:
        raise InputDataError(f"cannot read exposure file {source}: {exc}") from exc
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise InputDataError("exposure file missing column(s): " + ", ".join(missing))
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"].map(normalize_date), format="%Y%m%d")
    result["exposure_type"] = result["exposure_type"].astype(str).str.strip().str.lower()
    result["name"] = result["name"].astype(str).str.strip()
    if "portfolio" not in result.columns:
        result["portfolio"] = "risk_optimized"
    else:
        result["portfolio"] = result["portfolio"].astype(str).str.strip()
    if result["portfolio"].eq("").any():
        raise InputDataError("exposure file contains an empty portfolio name")
    if result["name"].eq("").any():
        raise InputDataError("exposure file contains an empty exposure name")
    if not result["exposure_type"].isin({"style", "industry"}).all():
        values = sorted(result.loc[~result["exposure_type"].isin({"style", "industry"}), "exposure_type"].unique())
        raise InputDataError("unsupported exposure type(s): " + ", ".join(values))
    for column in ("benchmark_exposure", "portfolio_exposure", "active_exposure", "lower_bound", "upper_bound", "lower_slack", "upper_slack"):
        if column in result.columns:
            result[column] = pd.to_numeric(result[column], errors="coerce")
    if not np.isfinite(result["active_exposure"].to_numpy(dtype=float)).all():
        raise InputDataError("active_exposure contains missing or non-finite values")
    keys = ["date", "portfolio", "exposure_type", "name"]
    if result.duplicated(keys).any():
        raise InputDataError("exposure file contains duplicate date/type/name rows")
    return result.sort_values(keys, kind="stable").reset_index(drop=True)


def _prepare_output(path: Path) -> Path:
    destination = path.expanduser().resolve()
    if destination.exists() and any(destination.iterdir()):
        raise InputDataError(f"output directory must be new or empty: {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _format_dates(axis: plt.Axes) -> None:
    locator = mdates.AutoDateLocator(minticks=4, maxticks=8)
    axis.xaxis.set_major_locator(locator)
    axis.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


def _plot_style(frame: pd.DataFrame, destination: Path, title_prefix: str) -> str | None:
    style = frame.loc[frame["exposure_type"] == "style"].copy()
    if style.empty:
        return None
    names = sorted(style["name"].unique())
    columns = 2
    rows = math.ceil(len(names) / columns)
    figure, axes = plt.subplots(rows, columns, figsize=(14, max(3.2 * rows, 4)), squeeze=False)
    for index, name in enumerate(names):
        axis = axes.flat[index]
        item = style.loc[style["name"] == name].sort_values("date")
        axis.plot(item["date"], item["active_exposure"], color="#1f77b4", linewidth=1.5, label="active")
        for column, label, color in (("lower_bound", "lower", "#a33"), ("upper_bound", "upper", "#a33")):
            if column in item.columns and item[column].notna().any():
                axis.plot(item["date"], item[column], color=color, linewidth=0.9, linestyle="--", label=label)
        axis.axhline(0.0, color="#555", linewidth=0.7)
        axis.set_title(name)
        axis.grid(True, alpha=0.25)
        _format_dates(axis)
        if index == 0:
            axis.legend(loc="best", fontsize=8)
    for axis in axes.flat[len(names):]:
        axis.set_visible(False)
    figure.suptitle(f"{title_prefix}: style active exposure", y=1.01)
    figure.tight_layout()
    output = destination / "style_active_exposure.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output.name


def _periodic_industry(frame: pd.DataFrame, frequency: str) -> pd.DataFrame:
    industry = frame.loc[frame["exposure_type"] == "industry"].copy()
    if industry.empty:
        return industry
    if frequency == "rebalance":
        return industry
    code = {"weekly": "W-FRI", "monthly": "M"}[frequency]
    industry["period"] = industry["date"].dt.to_period(code)
    return (
        industry.sort_values("date")
        .groupby(["period", "name"], as_index=False, sort=True)
        .tail(1)
        .drop(columns="period")
    )


def _plot_industry(frame: pd.DataFrame, destination: Path, frequency: str, title_prefix: str) -> str | None:
    industry = _periodic_industry(frame, frequency)
    if industry.empty:
        return None
    matrix = industry.pivot(index="name", columns="date", values="active_exposure").sort_index()
    values = matrix.to_numpy(dtype=float)
    limit = float(np.nanquantile(np.abs(values), 0.98)) if values.size else 0.0
    limit = max(limit, 1.0e-4)
    height = max(5.0, min(0.28 * len(matrix.index) + 2.0, 18.0))
    width = max(10.0, min(0.20 * len(matrix.columns) + 5.0, 22.0))
    figure, axis = plt.subplots(figsize=(width, height))
    image = axis.imshow(values, aspect="auto", cmap="coolwarm", vmin=-limit, vmax=limit, interpolation="nearest")
    axis.set_yticks(np.arange(len(matrix.index)))
    axis.set_yticklabels(matrix.index, fontsize=8)
    ticks = np.linspace(0, max(len(matrix.columns) - 1, 0), num=min(8, len(matrix.columns)), dtype=int)
    axis.set_xticks(ticks)
    axis.set_xticklabels([pd.Timestamp(matrix.columns[item]).strftime("%Y-%m-%d") for item in ticks], rotation=35, ha="right", fontsize=8)
    axis.set_title(f"{title_prefix}: industry active exposure ({frequency})")
    figure.colorbar(image, ax=axis, label="active exposure")
    figure.tight_layout()
    output = destination / "industry_active_exposure.png"
    figure.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(figure)
    return output.name


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (kind, name), item in frame.groupby(["exposure_type", "name"], sort=True):
        active = item["active_exposure"].astype(float)
        passed = item.get("passed", pd.Series(True, index=item.index)).fillna(True).astype(bool)
        binding = item.get("binding", pd.Series(False, index=item.index)).fillna(False).astype(bool)
        slack_columns = [column for column in ("lower_slack", "upper_slack") if column in item]
        slacks = pd.concat([item[column] for column in slack_columns], axis=0).dropna() if slack_columns else pd.Series(dtype=float)
        rows.append({
            "exposure_type": kind,
            "name": name,
            "observations": int(len(item)),
            "mean_active_exposure": float(active.mean()),
            "max_absolute_active_exposure": float(active.abs().max()),
            "p95_absolute_active_exposure": float(active.abs().quantile(0.95)),
            "violation_count": int((~passed).sum()),
            "binding_count": int(binding.sum()),
            "minimum_constraint_slack": None if slacks.empty else float(slacks.min()),
        })
    return pd.DataFrame(rows).sort_values(["exposure_type", "name"], kind="stable").reset_index(drop=True)


def _write_exposure_table(frame: pd.DataFrame, destination: Path) -> str:
    """Write the historical factor-backtest exposure-table shape.

    The index deliberately has no name so pandas writes the same first
    unnamed column shape used by the original charting workflow.
    """

    table = frame.loc[:, ["name", "portfolio_exposure", "benchmark_exposure", "date"]].copy()
    table = table.rename(
        columns={
            "portfolio_exposure": "port",
            "benchmark_exposure": "benchmark",
        }
    )
    table["date"] = pd.to_datetime(table["date"]).dt.strftime("%Y-%m-%d")
    table = table.sort_values(["date", "name"], kind="stable").set_index("name")
    table.index.name = None
    output = destination / "Explo.csv"
    table.to_csv(output)
    return output.name


def _plot_radar(
    frame: pd.DataFrame,
    destination: Path,
    *,
    exposure_type: str,
    filename: str,
    title: str,
    annotate: bool,
) -> str | None:
    """Render mean portfolio/benchmark exposures in the standard radar form."""

    item = frame.loc[frame["exposure_type"] == exposure_type].copy()
    if item.empty:
        return None
    average = (
        item.groupby("name", sort=True)[["portfolio_exposure", "benchmark_exposure"]]
        .mean()
        .sort_index()
    )
    categories = average.index.tolist()
    count = len(categories)
    angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False).tolist()
    portfolio = average["portfolio_exposure"].tolist()
    benchmark = average["benchmark_exposure"].tolist()
    angles += angles[:1]
    portfolio += portfolio[:1]
    benchmark += benchmark[:1]

    figure, axis = plt.subplots(figsize=(6, 6), subplot_kw={"polar": True})
    portfolio_line, = axis.plot(angles, portfolio, color="blue", linewidth=0.5)
    axis.fill(angles, portfolio, color="blue", alpha=0.25)
    benchmark_line, = axis.plot(angles, benchmark, color="red", linewidth=0.5)
    axis.fill(angles, benchmark, color="red", alpha=0.15)
    axis.set_yticklabels([])
    axis.set_xticks(angles[:-1])
    axis.set_xticklabels(categories, fontsize=8 if exposure_type == "industry" else 10)
    if annotate:
        for index, value in enumerate(portfolio[:-1]):
            axis.annotate(
                f"{value:.2f}",
                (angles[index], value),
                xytext=(0.05, 0.05),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
        for index, value in enumerate(benchmark[:-1]):
            axis.annotate(
                f"{value:.2f}",
                (angles[index], value),
                xytext=(0.05, 0.05),
                textcoords="offset points",
                ha="center",
                va="bottom",
            )
    axis.set_title(title)
    axis.legend(
        [portfolio_line, benchmark_line],
        ["portfolio", "benchmark"],
        loc="upper right",
        bbox_to_anchor=(1.1, 1.1),
    )
    output = destination / filename
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
    return output.name


def render_exposure_report(config: ExposureReportConfig) -> dict[str, object]:
    if config.industry_frequency not in {"rebalance", "weekly", "monthly"}:
        raise InputDataError("industry_frequency must be rebalance, weekly, or monthly")
    source = config.exposure_file.expanduser().resolve()
    frame = _load_exposures(source)
    frame = frame.loc[frame["portfolio"].eq(config.portfolio)].copy()
    if frame.empty:
        raise InputDataError(
            f"exposure file contains no rows for portfolio: {config.portfolio}"
        )
    destination = _prepare_output(config.output_dir)
    summary = _summary(frame)
    summary_path = destination / "exposure_summary.csv"
    summary.to_csv(summary_path, index=False)
    outputs = [summary_path.name, _write_exposure_table(frame, destination)]
    style = _plot_style(frame, destination, config.title_prefix)
    industry = _plot_industry(frame, destination, config.industry_frequency, config.title_prefix)
    style_radar = _plot_radar(
        frame,
        destination,
        exposure_type="style",
        filename="Style_radar.png",
        title="Strategy Style Explo",
        annotate=True,
    )
    industry_radar = _plot_radar(
        frame,
        destination,
        exposure_type="industry",
        filename="Industry_radar.png",
        title="Strategy Industry Explo",
        annotate=False,
    )
    outputs.extend(
        name
        for name in (style, industry, style_radar, industry_radar)
        if name is not None
    )
    manifest = {
        "status": "success",
        "input": str(source),
        "input_sha256": _sha256(source),
        "rows": int(len(frame)),
        "date_min": frame["date"].min().strftime("%Y%m%d"),
        "date_max": frame["date"].max().strftime("%Y%m%d"),
        "style_factor_count": int(
            frame.loc[frame["exposure_type"] == "style", "name"].nunique()
        ),
        "industry_factor_count": int(
            frame.loc[frame["exposure_type"] == "industry", "name"].nunique()
        ),
        "industry_frequency": config.industry_frequency,
        "portfolio": config.portfolio,
        "outputs": outputs,
        "definition": "portfolio exposure minus benchmark exposure; no optimizer or backtest calculation is rerun",
    }
    (destination / "report_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=True, indent=2) + chr(10),
        encoding="utf-8",
    )
    return manifest
