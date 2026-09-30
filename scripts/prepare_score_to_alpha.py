#!/usr/bin/env python3
"""Convert frozen cross-sectional scores into PIT expected returns."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--signal-file", required=True, type=Path)
    parser.add_argument("--candidate-file", required=True, type=Path)
    parser.add_argument("--trade-price-file", required=True, type=Path)
    parser.add_argument("--risk-resolution-manifest", required=True, type=Path)
    parser.add_argument("--calibration-start", required=True)
    parser.add_argument("--calibration-end", required=True)
    parser.add_argument("--application-start", required=True)
    parser.add_argument("--application-end", required=True)
    parser.add_argument("--entry-lag-days", type=int, default=1)
    parser.add_argument("--return-horizon-days", type=int, default=5)
    parser.add_argument("--minimum-cross-section", type=int, default=200)
    parser.add_argument("--minimum-ic-observations", type=int, default=60)
    parser.add_argument("--ic-shrinkage", type=float, default=0.5)
    parser.add_argument(
        "--negative-ic-policy", choices=("zero", "preserve"), default="zero"
    )
    parser.add_argument("--maximum-absolute-alpha", type=float, default=0.20)
    parser.add_argument("--output-file", required=True, type=Path)
    parser.add_argument("--ic-file", required=True, type=Path)
    parser.add_argument("--manifest-file", required=True, type=Path)
    return parser.parse_args()


def normalize_dates(values: pd.Series) -> pd.Series:
    return values.astype(str).str.replace("-", "", regex=False).str[:8]


def ticker_code(value: object) -> str:
    text = str(value).strip().split(".")[0]
    if text.endswith(".0"):
        text = text[:-2]
    return text.zfill(6)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_signal(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"date", "ticker", "prediction"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"signal missing columns: {sorted(missing)}")
    frame = frame[["date", "ticker", "prediction"]].copy()
    frame["date"] = normalize_dates(frame["date"])
    frame["ticker"] = frame["ticker"].astype(str)
    frame["code"] = frame["ticker"].map(ticker_code)
    frame["prediction"] = pd.to_numeric(frame["prediction"], errors="coerce")
    if frame.duplicated(["date", "ticker"]).any():
        raise ValueError("signal contains duplicate date/ticker rows")
    if not np.isfinite(frame["prediction"]).all():
        raise ValueError("signal contains non-finite predictions")
    return frame


def load_trade_prices(path: Path) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    frame.index = normalize_dates(pd.Series(frame.index)).to_numpy()
    frame.columns = [ticker_code(value) for value in frame.columns]
    if frame.index.duplicated().any() or pd.Index(frame.columns).duplicated().any():
        raise ValueError("trade-price matrix contains duplicate dates or tickers")
    return frame.sort_index()


def estimate_daily_ic(
    signal: pd.DataFrame,
    prices: pd.DataFrame,
    *,
    calibration_start: str,
    calibration_end: str,
    entry_lag_days: int,
    return_horizon_days: int,
    minimum_cross_section: int,
) -> pd.DataFrame:
    calendar = prices.index.astype(str).tolist()
    positions = {date: index for index, date in enumerate(calendar)}
    rows = []
    selected = signal.loc[
        signal["date"].between(calibration_start, calibration_end)
    ]
    for date, group in selected.groupby("date", sort=True):
        if date not in positions:
            continue
        entry_position = positions[date] + entry_lag_days
        exit_position = entry_position + return_horizon_days
        if exit_position >= len(calendar):
            continue
        entry_date = calendar[entry_position]
        exit_date = calendar[exit_position]
        if exit_date > calibration_end:
            continue
        codes = group["code"].tolist()
        entry = prices.loc[entry_date].reindex(codes).to_numpy(dtype=float)
        exit_price = prices.loc[exit_date].reindex(codes).to_numpy(dtype=float)
        prediction = group["prediction"].to_numpy(dtype=float)
        valid = (
            np.isfinite(entry)
            & np.isfinite(exit_price)
            & np.isfinite(prediction)
            & (entry > 0.0)
            & (exit_price > 0.0)
        )
        observations = int(valid.sum())
        if observations < minimum_cross_section:
            continue
        realized = pd.Series(exit_price[valid] / entry[valid] - 1.0)
        score = pd.Series(prediction[valid])
        ic = score.corr(realized, method="spearman")
        if pd.isna(ic):
            continue
        rows.append(
            {
                "signal_date": date,
                "entry_date": entry_date,
                "label_maturity_date": exit_date,
                "observations": observations,
                "rank_ic": float(ic),
            }
        )
    return pd.DataFrame(rows)


def standardized_rank_score(prediction: pd.Series) -> pd.Series:
    score = prediction.rank(method="average", pct=True)
    score = score - float(score.mean())
    scale = float(score.std(ddof=0))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("application signal has no cross-sectional variation")
    return score / scale


def load_specific_variance(path: Path) -> pd.Series:
    frame = pd.read_parquet(path)
    if isinstance(frame, pd.Series):
        result = frame
    elif "specific_var" in frame.columns:
        result = frame["specific_var"]
    elif frame.shape[1] == 1:
        result = frame.iloc[:, 0]
    else:
        raise ValueError(f"cannot identify specific variance column in {path}")
    result.index = result.index.astype(str)
    result = pd.to_numeric(result, errors="coerce")
    if result.index.duplicated().any():
        raise ValueError(f"specific variance contains duplicate tickers: {path}")
    if result.isna().any() or not np.isfinite(result).all() or result.le(0).any():
        raise ValueError(f"specific variance must be positive and finite: {path}")
    return result


def materialize_expected_returns(
    signal: pd.DataFrame,
    candidates: pd.DataFrame,
    risk_manifest: dict[str, object],
    *,
    application_start: str,
    application_end: str,
    effective_ic: float,
    maximum_absolute_alpha: float,
) -> tuple[pd.DataFrame, dict[str, object]]:
    resolutions = {
        str(row["date"]): Path(str(row["specific_variance_file"]))
        for row in risk_manifest["risk_resolutions"]
        if application_start <= str(row["date"]) <= application_end
    }
    signal = signal.loc[
        signal["date"].between(application_start, application_end)
    ]
    candidates = candidates.loc[
        candidates["date"].between(application_start, application_end)
    ]
    expected_dates = sorted(candidates["date"].unique().tolist())
    missing_resolutions = sorted(set(expected_dates) - set(resolutions))
    if missing_resolutions:
        raise ValueError(
            f"risk manifest lacks application dates: {missing_resolutions[:10]}"
        )
    cache: dict[Path, pd.Series] = {}
    frames = []
    clipped = 0
    candidate_missing = []
    specific_variance_imputed = []
    for date in expected_dates:
        path = resolutions[date]
        if path not in cache:
            cache[path] = load_specific_variance(path)
        specific = cache[path]
        group = signal.loc[signal["date"].eq(date)].set_index("ticker")
        candidate_assets = pd.Index(
            candidates.loc[candidates["date"].eq(date), "ticker"].astype(str)
        )
        model_assets = pd.Index(specific.index.astype(str))
        required_assets = model_assets.union(candidate_assets)
        group = group.reindex(required_assets)
        if group["prediction"].isna().any():
            missing_signal = group.index[group["prediction"].isna()].tolist()
            raise ValueError(f"signal missing required assets on {date}: {missing_signal[:10]}")
        score = standardized_rank_score(group["prediction"])
        aligned_specific = specific.reindex(required_assets)
        imputed = aligned_specific.isna()
        if imputed.any():
            aligned_specific = aligned_specific.fillna(float(specific.median()))
            specific_variance_imputed.extend(
                {"date": date, "ticker": str(ticker)}
                for ticker in aligned_specific.index[imputed]
            )
        alpha = effective_ic * np.sqrt(aligned_specific) * score
        unclipped = alpha.copy()
        alpha = alpha.clip(
            lower=-maximum_absolute_alpha,
            upper=maximum_absolute_alpha,
        )
        clipped += int((alpha != unclipped).sum())
        frames.append(pd.DataFrame({"date": date, "ticker": required_assets, "prediction": alpha.to_numpy(dtype=float)}))
    if candidate_missing:
        raise ValueError(
            "expected returns lack candidate coverage: "
            f"{candidate_missing[:10]}"
        )
    output = pd.concat(frames, ignore_index=True)
    if output.duplicated(["date", "ticker"]).any():
        raise ValueError("expected-return output contains duplicate keys")
    if not np.isfinite(output["prediction"]).all():
        raise ValueError("expected-return output contains non-finite values")
    diagnostics = {
        "application_dates": len(expected_dates),
        "application_date_start": expected_dates[0],
        "application_date_end": expected_dates[-1],
        "output_rows": len(output),
        "output_tickers": output["ticker"].nunique(),
        "risk_files": len(cache),
        "clipped_alpha_rows": clipped,
        "candidate_missing_rows": len(candidate_missing),
        "specific_variance_imputed_rows": len(specific_variance_imputed),
        "specific_variance_imputed_tickers": len({row["ticker"] for row in specific_variance_imputed}),
        "specific_variance_imputation_examples": specific_variance_imputed[:20],
    }
    return output, diagnostics


def main() -> int:
    args = parse_args()
    if args.entry_lag_days < 0 or args.return_horizon_days <= 0:
        raise ValueError("entry lag must be non-negative and horizon positive")
    if args.minimum_cross_section < 2 or args.minimum_ic_observations < 2:
        raise ValueError("minimum observation counts must be at least two")
    if not 0.0 <= args.ic_shrinkage <= 1.0:
        raise ValueError("ic shrinkage must be between zero and one")
    if args.maximum_absolute_alpha <= 0.0:
        raise ValueError("maximum absolute alpha must be positive")
    if str(args.calibration_end) >= str(args.application_start):
        raise ValueError("calibration must end before the application window")

    signal = load_signal(args.signal_file)
    candidates = pd.read_parquet(args.candidate_file, columns=["date", "ticker"])
    candidates["date"] = normalize_dates(candidates["date"])
    candidates["ticker"] = candidates["ticker"].astype(str)
    if candidates.duplicated(["date", "ticker"]).any():
        raise ValueError("candidate input contains duplicate keys")
    prices = load_trade_prices(args.trade_price_file)
    ic = estimate_daily_ic(
        signal,
        prices,
        calibration_start=str(args.calibration_start),
        calibration_end=str(args.calibration_end),
        entry_lag_days=args.entry_lag_days,
        return_horizon_days=args.return_horizon_days,
        minimum_cross_section=args.minimum_cross_section,
    )
    if len(ic) < args.minimum_ic_observations:
        raise ValueError(
            f"only {len(ic)} valid IC dates; need {args.minimum_ic_observations}"
        )
    raw_ic = float(ic["rank_ic"].mean())
    shrunk_ic = raw_ic * args.ic_shrinkage
    effective_ic = (
        max(shrunk_ic, 0.0)
        if args.negative_ic_policy == "zero"
        else shrunk_ic
    )
    risk_manifest = json.loads(
        args.risk_resolution_manifest.read_text(encoding="utf-8")
    )
    expected, application = materialize_expected_returns(
        signal,
        candidates,
        risk_manifest,
        application_start=str(args.application_start),
        application_end=str(args.application_end),
        effective_ic=effective_ic,
        maximum_absolute_alpha=args.maximum_absolute_alpha,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output_file.with_suffix(args.output_file.suffix + ".tmp")
    expected.to_parquet(temporary, index=False)
    os.replace(temporary, args.output_file)
    args.ic_file.parent.mkdir(parents=True, exist_ok=True)
    ic.to_parquet(args.ic_file, index=False)
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "method": "alpha = shrunk_mean_rank_ic * annualized_specific_volatility * standardized_uniform_rank_score",
        "timing": {
            "calibration_start": str(args.calibration_start),
            "calibration_end": str(args.calibration_end),
            "application_start": str(args.application_start),
            "application_end": str(args.application_end),
            "entry_lag_days": args.entry_lag_days,
            "return_horizon_days": args.return_horizon_days,
            "latest_label_maturity_date": str(ic["label_maturity_date"].max()),
        },
        "calibration": {
            "valid_ic_dates": len(ic),
            "minimum_ic_observations": args.minimum_ic_observations,
            "minimum_cross_section": args.minimum_cross_section,
            "raw_mean_rank_ic": raw_ic,
            "ic_shrinkage": args.ic_shrinkage,
            "negative_ic_policy": args.negative_ic_policy,
            "effective_ic": effective_ic,
            "rank_ic_std": float(ic["rank_ic"].std(ddof=1)),
            "rank_ic_positive_fraction": float(ic["rank_ic"].gt(0).mean()),
        },
        "application": application,
        "maximum_absolute_alpha": args.maximum_absolute_alpha,
        "inputs": {
            "signal": {"path": str(args.signal_file.resolve()), "sha256": sha256(args.signal_file)},
            "candidates": {"path": str(args.candidate_file.resolve()), "sha256": sha256(args.candidate_file)},
            "trade_price": {"path": str(args.trade_price_file.resolve()), "sha256": sha256(args.trade_price_file)},
            "risk_resolution_manifest": {"path": str(args.risk_resolution_manifest.resolve()), "sha256": sha256(args.risk_resolution_manifest)},
        },
        "outputs": {
            "expected_returns": {"path": str(args.output_file.resolve()), "sha256": sha256(args.output_file)},
            "daily_ic": {"path": str(args.ic_file.resolve()), "sha256": sha256(args.ic_file)},
        },
    }
    args.manifest_file.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_file.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
