from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, NonlinearConstraint, linprog, minimize

from .diagnostics import (
    constraint_report,
    resolve_candidate_weight_range,
    resolve_industry_ranges,
    resolve_style_ranges,
)
from .errors import OptimizationError
from .risk import PortfolioRisk, as_portfolio_risk


@dataclass(frozen=True)
class OptimizationResult:
    weights: pd.Series
    solver: dict[str, Any]
    constraints: dict[str, Any]


def build_equal_weight_baseline(signal_score: pd.Series, top_n: int) -> pd.Series:
    count = min(int(top_n), len(signal_score))
    selected = signal_score.sort_values(ascending=False, kind="mergesort").head(count).index
    weights = pd.Series(0.0, index=signal_score.index, name="target_weight")
    weights.loc[selected] = 1.0 / count
    return weights


def _build_linear_system(
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    tradable: pd.Series,
    config: dict[str, Any],
    *,
    exit_only_mask: pd.Series | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Bounds, bool]:
    n_assets = len(benchmark)
    with_turnover = config["max_turnover"] is not None
    dimension = n_assets * 2 if with_turnover else n_assets
    lower = np.zeros(dimension)
    upper = np.ones(dimension)

    lower[:n_assets] = np.maximum(
        0.0, benchmark.to_numpy(dtype=float) - float(config["max_active_weight"])
    )
    upper[:n_assets] = np.minimum(
        float(config["max_weight"]),
        benchmark.to_numpy(dtype=float) + float(config["max_active_weight"]),
    )
    frozen = ~tradable.to_numpy(dtype=bool)
    if frozen.any():
        if current is None:
            raise OptimizationError("current weights are required for non-tradable assets")
        frozen_values = current.to_numpy(dtype=float)[frozen]
        lower[:n_assets][frozen] = frozen_values
        upper[:n_assets][frozen] = frozen_values

    if exit_only_mask is not None:
        if current is None:
            raise OptimizationError("exit-only constraints require current weights")
        aligned_exit_only = exit_only_mask.reindex(benchmark.index)
        if aligned_exit_only.isna().any():
            raise OptimizationError("exit_only_mask is missing optimization assets")
        exit_only = aligned_exit_only.to_numpy(dtype=bool)
        if np.any(exit_only & frozen):
            raise OptimizationError("exit-only assets must be tradable")
        upper[:n_assets][exit_only] = np.minimum(
            upper[:n_assets][exit_only],
            current.to_numpy(dtype=float)[exit_only],
        )

    if np.any(lower[:n_assets] > upper[:n_assets] + config["constraint_tolerance"]):
        raise OptimizationError("position, active-weight, and frozen-position bounds conflict")
    if lower[:n_assets].sum() > 1.0 + config["constraint_tolerance"]:
        raise OptimizationError("lower position bounds sum to more than one")
    if upper[:n_assets].sum() < 1.0 - config["constraint_tolerance"]:
        raise OptimizationError("upper position bounds cannot support full investment")

    equality_rows: list[np.ndarray] = []
    equality_values: list[float] = []
    budget = np.zeros(dimension)
    budget[:n_assets] = 1.0
    equality_rows.append(budget)
    equality_values.append(1.0)

    inequality_rows: list[np.ndarray] = []
    inequality_values: list[float] = []

    def add_active_range(
        coefficients: np.ndarray, center: float, lower_active: float, upper_active: float
    ) -> None:
        positive = np.zeros(dimension)
        positive[:n_assets] = coefficients
        inequality_rows.append(positive)
        inequality_values.append(center + upper_active)
        inequality_rows.append(-positive)
        inequality_values.append(-(center + lower_active))

    if sectors is not None:
        names = sorted(sectors.unique().tolist())
        ranges = resolve_industry_ranges(config, names)
        for name, specification in ranges.items():
            coefficients = (sectors == name).to_numpy(dtype=float)
            center = float(coefficients @ benchmark.to_numpy(dtype=float))
            add_active_range(
                coefficients, center, specification["lower_active"], specification["upper_active"]
            )

    if exposures is not None:
        names = list(exposures.columns)
        ranges = resolve_style_ranges(config, names)
        for name, specification in ranges.items():
            coefficients = exposures[name].to_numpy(dtype=float)
            center = float(coefficients @ benchmark.to_numpy(dtype=float))
            add_active_range(
                coefficients, center, specification["lower_active"], specification["upper_active"]
            )

    candidate_range = config.get("candidate_weight_range")
    if candidate_range is not None:
        if candidate_mask is None:
            raise OptimizationError(
                "candidate_mask is required when candidate_weight_range is configured"
            )
        candidate_range = resolve_candidate_weight_range(
            candidate_range, candidate_mask, current, tradable
        )
        assert candidate_range is not None
        coefficients = candidate_mask.to_numpy(dtype=float)
        if not coefficients.any():
            raise OptimizationError("candidate_mask contains no candidate assets")
        positive = np.zeros(dimension)
        positive[:n_assets] = coefficients
        inequality_rows.append(positive)
        inequality_values.append(float(candidate_range["max_weight"]))
        inequality_rows.append(-positive)
        inequality_values.append(-float(candidate_range["min_weight"]))

    if with_turnover:
        if current is None:
            raise OptimizationError("current weights are required when max_turnover is configured")
        current_values = current.to_numpy(dtype=float)
        for index in range(n_assets):
            row = np.zeros(dimension)
            row[index] = 1.0
            row[n_assets + index] = -1.0
            inequality_rows.append(row)
            inequality_values.append(current_values[index])

            row = np.zeros(dimension)
            row[index] = -1.0
            row[n_assets + index] = -1.0
            inequality_rows.append(row)
            inequality_values.append(-current_values[index])

        turnover = np.zeros(dimension)
        turnover[n_assets:] = 1.0
        inequality_rows.append(turnover)
        turnover_margin = float(config["constraint_tolerance"])
        inequality_values.append(
            2.0 * max(float(config["max_turnover"]) - turnover_margin, 0.0)
        )

    a_eq = np.vstack(equality_rows)
    b_eq = np.asarray(equality_values, dtype=float)
    a_ub = np.vstack(inequality_rows) if inequality_rows else np.empty((0, dimension))
    b_ub = np.asarray(inequality_values, dtype=float)
    return a_eq, b_eq, a_ub, b_ub, Bounds(lower, upper), with_turnover


def _linear_feasible_start(
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: Bounds,
) -> np.ndarray:
    scipy_bounds = list(zip(bounds.lb.tolist(), bounds.ub.tolist(), strict=True))
    result = linprog(
        np.zeros(len(bounds.lb)),
        A_ub=a_ub if len(a_ub) else None,
        b_ub=b_ub if len(a_ub) else None,
        A_eq=a_eq,
        b_eq=b_eq,
        bounds=scipy_bounds,
        method="highs",
    )
    if not result.success:
        raise OptimizationError(f"linear constraints are infeasible: {result.message}")
    return np.asarray(result.x, dtype=float)


def _preferred_benchmark_start(
    benchmark: pd.Series,
    current: pd.Series | None,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: Bounds,
    *,
    with_turnover: bool,
    tolerance: float,
) -> np.ndarray | None:
    n_assets = len(benchmark)
    candidate = np.zeros(len(bounds.lb), dtype=float)
    candidate[:n_assets] = benchmark.to_numpy(dtype=float)
    if with_turnover:
        if current is None:
            return None
        candidate[n_assets:] = np.abs(
            benchmark.to_numpy(dtype=float) - current.to_numpy(dtype=float)
        )
    if np.any(candidate < bounds.lb - tolerance) or np.any(
        candidate > bounds.ub + tolerance
    ):
        return None
    if not np.allclose(a_eq @ candidate, b_eq, rtol=0.0, atol=tolerance):
        return None
    if len(a_ub) and np.any(a_ub @ candidate > b_ub + tolerance):
        return None
    return candidate


def _optimize_scipy(
    expected_return: pd.Series,
    objective_signal: pd.Series,
    covariance: pd.DataFrame | PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    exit_only_mask: pd.Series | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
    *,
    initial_weights: np.ndarray | None = None,
) -> OptimizationResult:
    n_assets = len(expected_return)
    sigma = as_portfolio_risk(covariance).dense().to_numpy(dtype=float)
    mu = expected_return.to_numpy(dtype=float)
    benchmark_values = benchmark.to_numpy(dtype=float)
    current_values = None if current is None else current.to_numpy(dtype=float)

    full_a_eq, full_b_eq, full_a_ub, full_b_ub, full_bounds, with_turnover = _build_linear_system(
        benchmark, current, sectors, exposures, candidate_mask, tradable, constraint_config,
        exit_only_mask=exit_only_mask,
    )
    initial = _preferred_benchmark_start(
        benchmark,
        current,
        full_a_eq,
        full_b_eq,
        full_a_ub,
        full_b_ub,
        full_bounds,
        with_turnover=with_turnover,
        tolerance=float(constraint_config["constraint_tolerance"]),
    )
    if initial is None:
        initial = _linear_feasible_start(
            full_a_eq, full_b_eq, full_a_ub, full_b_ub, full_bounds
        )

    if with_turnover:
        reduced_config = {**constraint_config, "max_turnover": None}
        a_eq, b_eq, a_ub, b_ub, bounds, _ = _build_linear_system(
            benchmark,
            current,
            sectors,
            exposures,
            candidate_mask,
            tradable,
            reduced_config,
            exit_only_mask=exit_only_mask,
        )
        initial = np.asarray(initial[:n_assets], dtype=float)
    else:
        a_eq, b_eq, a_ub, b_ub, bounds = (
            full_a_eq,
            full_b_eq,
            full_a_ub,
            full_b_ub,
            full_bounds,
        )
    if initial_weights is not None:
        candidate = np.asarray(initial_weights, dtype=float)
        if candidate.shape != (n_assets,) or not np.isfinite(candidate).all():
            raise OptimizationError("hybrid SLSQP initial weights are invalid")
        feasibility_tolerance = float(constraint_config["constraint_tolerance"])
        if np.any(candidate < bounds.lb - feasibility_tolerance) or np.any(
            candidate > bounds.ub + feasibility_tolerance
        ):
            raise OptimizationError("hybrid SLSQP initial weights violate bounds")
        if not np.allclose(
            a_eq @ candidate, b_eq, rtol=0.0, atol=feasibility_tolerance
        ):
            raise OptimizationError("hybrid SLSQP initial weights violate equality constraints")
        if len(a_ub) and np.any(a_ub @ candidate > b_ub + feasibility_tolerance):
            raise OptimizationError("hybrid SLSQP initial weights violate linear constraints")
        initial = candidate

    linear_constraints: list[Any] = [LinearConstraint(a_eq, b_eq, b_eq)]
    if len(a_ub):
        linear_constraints.append(LinearConstraint(a_ub, -np.inf, b_ub))

    if with_turnover:
        if current_values is None:
            raise OptimizationError("current weights are required for turnover control")
        turnover_limit = 2.0 * float(constraint_config["max_turnover"])
        tolerance = float(constraint_config["constraint_tolerance"])
        turnover_epsilon = min(
            float(optimizer_config["smoothing_epsilon"]),
            max(tolerance / (10.0 * n_assets), 1.0e-14),
        )

        def smoothed_turnover(decision: np.ndarray) -> float:
            difference = decision[:n_assets] - current_values
            return float(np.sqrt(difference**2 + turnover_epsilon**2).sum())

        def smoothed_turnover_jacobian(decision: np.ndarray) -> np.ndarray:
            difference = decision[:n_assets] - current_values
            return difference / np.sqrt(difference**2 + turnover_epsilon**2)

        linear_constraints.append(
            NonlinearConstraint(
                smoothed_turnover,
                -np.inf,
                turnover_limit,
                jac=smoothed_turnover_jacobian,
            )
        )

    max_tracking_error = constraint_config["max_tracking_error"]
    if max_tracking_error is not None:
        limit_squared = float(max_tracking_error) ** 2

        def tracking_variance(decision: np.ndarray) -> float:
            active = decision[:n_assets] - benchmark_values
            return float(active @ sigma @ active)

        def tracking_jacobian(decision: np.ndarray) -> np.ndarray:
            gradient = np.zeros_like(decision)
            gradient[:n_assets] = 2.0 * sigma @ (decision[:n_assets] - benchmark_values)
            return gradient

        risk_constraint = NonlinearConstraint(
            tracking_variance, -np.inf, limit_squared, jac=tracking_jacobian
        )
        linear_constraints.append(risk_constraint)

        if tracking_variance(initial) > limit_squared:
            feasibility = minimize(
                tracking_variance,
                initial,
                method="SLSQP",
                jac=tracking_jacobian,
                bounds=bounds,
                constraints=linear_constraints[:-1],
                options={
                    "maxiter": optimizer_config["max_iterations"],
                    "ftol": optimizer_config["ftol"],
                    "disp": False,
                },
            )
            if not feasibility.success or tracking_variance(feasibility.x) > (
                limit_squared + constraint_config["constraint_tolerance"]
            ):
                raise OptimizationError("tracking-error limit is infeasible with linear constraints")
            initial = np.asarray(feasibility.x, dtype=float)

    risk_aversion = float(optimizer_config["risk_aversion"])
    turnover_penalty = float(optimizer_config["turnover_penalty"])
    epsilon = float(optimizer_config["smoothing_epsilon"])

    def objective(decision: np.ndarray) -> float:
        weights = decision[:n_assets]
        active = weights - benchmark_values
        if optimizer_config["objective_mode"] == "mean_variance":
            value = -float(weights @ mu) + 0.5 * risk_aversion * float(
                active @ sigma @ active
            )
        else:
            value = -float(weights @ objective_signal.to_numpy(dtype=float))
        if current_values is not None and turnover_penalty > 0:
            difference = weights - current_values
            value += turnover_penalty * float(np.sqrt(difference**2 + epsilon**2).sum())
        return value

    def objective_jacobian(decision: np.ndarray) -> np.ndarray:
        weights = decision[:n_assets]
        gradient = np.zeros_like(decision)
        if optimizer_config["objective_mode"] == "mean_variance":
            gradient[:n_assets] = -mu + risk_aversion * sigma @ (
                weights - benchmark_values
            )
        else:
            gradient[:n_assets] = -objective_signal.to_numpy(dtype=float)
        if current_values is not None and turnover_penalty > 0:
            difference = weights - current_values
            gradient[:n_assets] += (
                turnover_penalty * difference / np.sqrt(difference**2 + epsilon**2)
            )
        return gradient

    result = minimize(
        objective,
        initial,
        method="SLSQP",
        jac=objective_jacobian,
        bounds=bounds,
        constraints=linear_constraints,
        options={
            "maxiter": optimizer_config["max_iterations"],
            "ftol": optimizer_config["ftol"],
            "disp": False,
        },
    )
    if not result.success:
        raise OptimizationError(
            f"optimizer failed with status {result.status}: {result.message}"
        )

    weights = pd.Series(
        np.asarray(result.x[:n_assets], dtype=float),
        index=expected_return.index,
        name="target_weight",
    )
    report = constraint_report(
        weights,
        benchmark,
        current,
        covariance,
        sectors,
        exposures,
        tradable,
        constraint_config,
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
    )
    if not report["passed"]:
        details = ", ".join(item["constraint"] for item in report["violations"])
        raise OptimizationError(f"solver result violates hard constraints: {details}")

    solver = {
        "backend": "scipy_slsqp",
        "objective_mode": optimizer_config["objective_mode"],
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "iterations": int(result.nit),
        "objective_value": float(result.fun),
        "used_turnover_auxiliary_variables": False,
        "used_turnover_auxiliary_feasibility_lp": with_turnover,
        "turnover_constraint_form": (
            "smooth_l1_with_analytic_jacobian" if with_turnover else None
        ),
    }
    return OptimizationResult(weights=weights, solver=solver, constraints=report)


def _optimize_score_highs(
    expected_return: pd.Series,
    objective_signal: pd.Series,
    covariance: pd.DataFrame | PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    exit_only_mask: pd.Series | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
) -> OptimizationResult:
    """Solve score/TE with certified outer LP and inner feasible bounds."""
    n_assets = len(expected_return)
    sigma = as_portfolio_risk(covariance).dense().to_numpy(dtype=float)
    benchmark_values = benchmark.to_numpy(dtype=float)
    a_eq, b_eq, base_a_ub, base_b_ub, bounds, with_turnover = _build_linear_system(
        benchmark, current, sectors, exposures, candidate_mask, tradable, constraint_config,
        exit_only_mask=exit_only_mask,
    )
    objective = np.zeros(len(bounds.lb), dtype=float)
    objective[:n_assets] = -objective_signal.to_numpy(dtype=float)
    if with_turnover:
        objective[n_assets:] = float(optimizer_config["turnover_penalty"])
    scipy_bounds = list(zip(bounds.lb.tolist(), bounds.ub.tolist(), strict=True))
    max_tracking_error = constraint_config["max_tracking_error"]
    limit = None if max_tracking_error is None else float(max_tracking_error)
    tolerance = float(constraint_config["constraint_tolerance"])
    iteration_budget = int(optimizer_config["max_iterations"])
    cutting_plane_budget = int(
        optimizer_config.get("max_cutting_planes", iteration_budget)
    )
    cutting_planes = 0
    total_highs_iterations = 0
    risk_anchor_iterations = 0

    def solve_lp(
        coefficients: np.ndarray, a_ub: np.ndarray, b_ub: np.ndarray
    ) -> Any:
        nonlocal total_highs_iterations
        solved = linprog(
            coefficients,
            A_ub=a_ub if len(a_ub) else None,
            b_ub=b_ub if len(a_ub) else None,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=scipy_bounds,
            method="highs",
        )
        total_highs_iterations += int(solved.nit or 0)
        if not solved.success:
            raise OptimizationError(
                f"HiGHS outer-inner optimizer failed: {solved.message}"
            )
        return solved

    def tighten_turnover(decision: np.ndarray) -> np.ndarray:
        tightened = np.asarray(decision, dtype=float).copy()
        if with_turnover and current is not None:
            tightened[n_assets:] = np.abs(
                tightened[:n_assets] - current.to_numpy(dtype=float)
            )
        return tightened

    def tracking_variance(decision: np.ndarray) -> float:
        active = decision[:n_assets] - benchmark_values
        return float(active @ sigma @ active)

    result = solve_lp(objective, base_a_ub, base_b_ub)
    final_decision = tighten_turnover(result.x)
    optimality_gap = 0.0
    if limit is not None and tracking_variance(final_decision) > (
        limit + 0.5 * tolerance
    ) ** 2:
        anchor = final_decision.copy()
        risk_target = max(limit - 0.25 * tolerance, 0.0) ** 2
        for risk_anchor_iterations in range(1, iteration_budget + 1):
            active = anchor[:n_assets] - benchmark_values
            variance = float(active @ sigma @ active)
            if variance <= risk_target:
                break
            gradient = np.zeros(len(bounds.lb), dtype=float)
            gradient[:n_assets] = 2.0 * sigma @ active
            oracle = tighten_turnover(
                solve_lp(gradient, base_a_ub, base_b_ub).x
            )
            direction = oracle - anchor
            weight_direction = direction[:n_assets]
            denominator = float(weight_direction @ sigma @ weight_direction)
            directional = float(active @ sigma @ weight_direction)
            if denominator <= 1.0e-20:
                raise OptimizationError(
                    "tracking-error constraint is infeasible with linear constraints"
                )
            step = float(np.clip(-directional / denominator, 0.0, 1.0))
            if step <= 1.0e-12:
                raise OptimizationError(
                    "tracking-error feasibility search made no progress"
                )
            anchor = tighten_turnover(anchor + step * direction)
        else:
            raise OptimizationError(
                "tracking-error feasibility search exceeded its iteration budget"
            )

        strict_anchor = anchor.copy()
        a_ub = base_a_ub.copy()
        b_ub = base_b_ub.copy()
        best_feasible = anchor.copy()
        best_objective = float(objective @ best_feasible)
        objective_tolerance = max(float(optimizer_config["ftol"]), 1.0e-5)
        for _ in range(cutting_plane_budget):
            outside = tighten_turnover(result.x)
            outside_active = outside[:n_assets] - benchmark_values
            outside_variance = float(outside_active @ sigma @ outside_active)
            if outside_variance <= (limit + 0.5 * tolerance) ** 2:
                candidate_objective = float(objective @ outside)
                if candidate_objective < best_objective:
                    best_feasible = outside
                    best_objective = candidate_objective
                optimality_gap = max(0.0, best_objective - float(result.fun))
                break

            if tracking_variance(anchor) > limit**2:
                anchor = strict_anchor.copy()
            anchor_active = anchor[:n_assets] - benchmark_values
            direction = outside - anchor
            weight_direction = direction[:n_assets]
            quadratic = float(weight_direction @ sigma @ weight_direction)
            linear = float(2.0 * anchor_active @ sigma @ weight_direction)
            constant = float(anchor_active @ sigma @ anchor_active - limit**2)
            endpoint = quadratic + linear + constant
            scalar_tolerance = max(1.0e-14, tolerance**2)
            if (
                quadratic <= 0
                or constant > scalar_tolerance
                or endpoint < -scalar_tolerance
            ):
                raise OptimizationError(
                    "cannot bracket tracking-error boundary between inner and outer points"
                )
            lower_step = 0.0
            upper_step = 1.0
            for _boundary_iteration in range(80):
                midpoint = 0.5 * (lower_step + upper_step)
                midpoint_value = (
                    quadratic * midpoint**2 + linear * midpoint + constant
                )
                if midpoint_value <= 0:
                    lower_step = midpoint
                else:
                    upper_step = midpoint
            step = lower_step
            feasible = tighten_turnover(anchor + step * direction)
            feasible_objective = float(objective @ feasible)
            if feasible_objective < best_objective:
                best_feasible = feasible
                best_objective = feasible_objective
            optimality_gap = max(0.0, best_objective - float(result.fun))
            if optimality_gap <= objective_tolerance:
                break

            sigma_active = sigma @ outside_active
            cut = np.zeros(len(bounds.lb), dtype=float)
            cut[:n_assets] = sigma_active
            cut_bound = float(
                sigma_active @ benchmark_values + limit * np.sqrt(outside_variance)
            )
            a_ub = np.vstack([a_ub, cut])
            b_ub = np.append(b_ub, cut_bound)
            cutting_planes += 1
            result = solve_lp(objective, a_ub, b_ub)
            anchor = tighten_turnover(0.05 * anchor + 0.95 * best_feasible)
        else:
            raise OptimizationError(
                "HiGHS outer-inner optimizer exceeded its cutting-plane budget; "
                f"certified objective gap is {optimality_gap:.12g} versus "
                f"tolerance {objective_tolerance:.12g}"
            )
        final_decision = best_feasible

    weights = pd.Series(
        np.asarray(final_decision[:n_assets], dtype=float),
        index=expected_return.index,
        name="target_weight",
    )
    report = constraint_report(
        weights,
        benchmark,
        current,
        covariance,
        sectors,
        exposures,
        tradable,
        constraint_config,
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
    )
    if not report["passed"]:
        details = ", ".join(item["constraint"] for item in report["violations"])
        raise OptimizationError(
            f"HiGHS outer-inner result violates hard constraints: {details}"
        )
    solver = {
        "backend": "scipy_highs_outer_inner",
        "solver": "highs",
        "objective_mode": optimizer_config["objective_mode"],
        "success": True,
        "status": int(result.status),
        "message": str(result.message),
        "iterations": cutting_planes + 1,
        "cutting_planes": cutting_planes,
        "cutting_plane_budget": cutting_plane_budget,
        "highs_iterations_total": total_highs_iterations,
        "objective_value": float(objective @ final_decision),
        "objective_lower_bound": float(result.fun),
        "certified_optimality_gap": optimality_gap,
        "objective_gap_tolerance": (
            max(float(optimizer_config["ftol"]), 1.0e-5) if limit is not None else 0.0
        ),
        "risk_anchor_iterations": risk_anchor_iterations,
        "used_turnover_auxiliary_variables": with_turnover,
        "turnover_constraint_form": "exact_l1_auxiliary" if with_turnover else None,
        "tracking_error_constraint_form": (
            "certified_outer_lp_and_inner_feasible_bound" if limit is not None else None
        ),
    }
    return OptimizationResult(weights=weights, solver=solver, constraints=report)


def _optimize_cvxpy(
    expected_return: pd.Series,
    objective_signal: pd.Series,
    covariance: pd.DataFrame | PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    exit_only_mask: pd.Series | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
    *,
    signal_score: pd.Series | None = None,
    signal_floor: float | None = None,
) -> OptimizationResult:
    from .conic_optimizer import solve_clarabel_socp

    n_assets = len(expected_return)
    benchmark_values = benchmark.to_numpy(dtype=float)
    a_eq, b_eq, a_ub, b_ub, bounds, with_turnover = _build_linear_system(
        benchmark,
        current,
        sectors,
        exposures,
        candidate_mask,
        tradable,
        constraint_config,
        exit_only_mask=exit_only_mask,
    )
    requested_backend = str(optimizer_config["solver_backend"])
    solved = solve_clarabel_socp(
        expected_return=expected_return.to_numpy(dtype=float),
        objective_signal=objective_signal.to_numpy(dtype=float),
        risk_input=covariance,
        benchmark=benchmark_values,
        current=None if current is None else current.to_numpy(dtype=float),
        a_eq=a_eq,
        b_eq=b_eq,
        a_ub=a_ub,
        b_ub=b_ub,
        lower_bounds=bounds.lb,
        upper_bounds=bounds.ub,
        with_turnover_auxiliary=with_turnover,
        optimizer_config=optimizer_config,
        signal_score=(
            None if signal_score is None else signal_score.to_numpy(dtype=float)
        ),
        signal_floor=signal_floor,
        max_tracking_error=constraint_config["max_tracking_error"],
        reported_backend=(
            "cvxpy"
            if requested_backend == "cvxpy"
            else "cvxpy_clarabel_socp"
        ),
    )

    weights = pd.Series(
        np.asarray(solved.decision[:n_assets], dtype=float),
        index=expected_return.index,
        name="target_weight",
    )
    report = constraint_report(
        weights,
        benchmark,
        current,
        covariance,
        sectors,
        exposures,
        tradable,
        constraint_config,
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
    )
    if not report["passed"]:
        details = ", ".join(item["constraint"] for item in report["violations"])
        raise OptimizationError(
            f"CVXPY result violates independently checked constraints: {details}"
        )
    return OptimizationResult(
        weights=weights,
        solver=solved.solver,
        constraints=report,
    )



def _execution_aware_anchor(
    anchor_weights: pd.Series,
    current: pd.Series | None,
    tradable: pd.Series,
    *,
    tolerance: float,
) -> pd.Series:
    index = anchor_weights.index
    anchor = anchor_weights.reindex(index)
    aligned_tradable = tradable.reindex(index)
    if anchor.isna().any() or not np.isfinite(anchor).all():
        raise OptimizationError("anchor_weights contains missing or non-finite values")
    if (anchor < -tolerance).any():
        raise OptimizationError("anchor_weights contains negative values")
    if aligned_tradable.isna().any():
        raise OptimizationError("tradable is missing anchor assets")

    if current is None:
        if (~aligned_tradable.astype(bool)).any():
            raise OptimizationError(
                "current weights are required to build an anchor with non-tradable assets"
            )
        frozen_weights = pd.Series(0.0, index=index)
    else:
        aligned_current = current.reindex(index)
        if aligned_current.isna().any() or not np.isfinite(aligned_current).all():
            raise OptimizationError("current contains missing or non-finite values")
        frozen_weights = aligned_current.where(~aligned_tradable.astype(bool), 0.0)

    frozen_total = float(frozen_weights.sum())
    if frozen_total > 1.0 + tolerance:
        raise OptimizationError("frozen anchor weight exceeds full investment")
    residual = max(1.0 - frozen_total, 0.0)
    eligible = aligned_tradable.astype(bool) & (anchor > 0.0)
    eligible_mass = float(anchor[eligible].sum())
    if residual > tolerance and eligible_mass <= tolerance:
        raise OptimizationError("anchor has no tradable positive-weight assets")

    result = frozen_weights.copy()
    if residual > tolerance:
        result.loc[eligible] = anchor.loc[eligible] * (residual / eligible_mass)
    result.name = "target_weight"
    return result

def _optimize_signal_preserving_minimum_variance(
    expected_return: pd.Series,
    covariance: pd.DataFrame | PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
    *,
    linear_cost_bps: float,
    signal_score: pd.Series | None,
    anchor_weights: pd.Series | None,
    candidate_mask: pd.Series | None,
    exit_only_mask: pd.Series | None,
) -> OptimizationResult:
    """Minimize active risk while preserving most of the signal utility."""
    if signal_score is None:
        raise OptimizationError(
            "signal_preserving_minimum_variance requires signal_score"
        )
    if anchor_weights is None:
        raise OptimizationError(
            "signal_preserving_minimum_variance requires anchor_weights"
        )
    if candidate_mask is None:
        raise OptimizationError(
            "signal_preserving_minimum_variance requires a candidate_mask"
        )

    from .lexicographic import (
        signal_capture_ratio,
        signal_utility,
        signal_utility_floor,
    )

    index = expected_return.index
    score = signal_score.reindex(index)
    if score.isna().any() or not np.isfinite(score).all():
        raise OptimizationError(
            "signal_score contains missing or non-finite values"
        )
    benchmark = benchmark.reindex(index)
    anchor = _execution_aware_anchor(
        anchor_weights.reindex(index),
        current,
        tradable,
        tolerance=float(constraint_config["constraint_tolerance"]),
    )
    anchor_report = constraint_report(
        anchor,
        benchmark,
        current,
        covariance,
        sectors,
        exposures,
        tradable,
        constraint_config,
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
    )
    if not anchor_report["passed"]:
        details = ", ".join(
            item["constraint"] for item in anchor_report["violations"]
        )
        raise OptimizationError(
            "executable signal anchor violates configured constraints: "
            + details
        )

    anchor_utility = signal_utility(anchor, benchmark, score)
    minimum_capture = float(optimizer_config["minimum_signal_capture"])
    utility_floor = signal_utility_floor(
        anchor_utility, minimum_capture
    )
    absolute_signal_floor = utility_floor + float(score @ benchmark)

    endpoint_optimizer = dict(optimizer_config)
    endpoint_optimizer["objective_mode"] = (
        "signal_preserving_minimum_variance"
    )
    endpoint = _optimize_cvxpy(
        pd.Series(0.0, index=index),
        score,
        covariance,
        benchmark,
        current,
        sectors,
        exposures,
        candidate_mask,
        exit_only_mask,
        tradable,
        endpoint_optimizer,
        constraint_config,
        signal_score=score,
        signal_floor=absolute_signal_floor,
    )
    weights = endpoint.weights
    final_utility = signal_utility(weights, benchmark, score)
    tolerance = max(
        float(constraint_config["constraint_tolerance"]), 1.0e-9
    )
    if final_utility < utility_floor - tolerance:
        raise OptimizationError(
            "signal-preserving result violates minimum signal capture"
        )

    report = constraint_report(
        weights,
        benchmark,
        current,
        covariance,
        sectors,
        exposures,
        tradable,
        constraint_config,
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
    )
    if not report["passed"]:
        details = ", ".join(
            item["constraint"] for item in report["violations"]
        )
        raise OptimizationError(
            "signal-preserving result violates hard constraints: " + details
        )

    risk = as_portfolio_risk(covariance)
    benchmark_values = benchmark.to_numpy(dtype=float)
    anchor_active = anchor.to_numpy(dtype=float) - benchmark_values
    optimized_active = weights.to_numpy(dtype=float) - benchmark_values
    anchor_volatility = float(
        np.sqrt(max(risk.variance(anchor_active), 0.0))
    )
    optimized_volatility = float(
        np.sqrt(max(risk.variance(optimized_active), 0.0))
    )
    turnover = (
        None
        if current is None
        else float(0.5 * (weights - current).abs().sum())
    )
    estimated_cost = (
        None
        if turnover is None
        else float(turnover * linear_cost_bps / 10000.0)
    )
    solver = dict(endpoint.solver)
    solver.update(
        {
            "objective_mode": "signal_preserving_minimum_variance",
            "inner_objective_mode": "active_minimum_variance",
            "minimum_signal_capture": minimum_capture,
            "primary_signal_utility": anchor_utility,
            "anchor_signal_utility": anchor_utility,
            "signal_utility_floor": utility_floor,
            "signal_utility_solver_floor": utility_floor,
            "signal_utility_tolerance": tolerance,
            "final_signal_utility": final_utility,
            "one_way_turnover": turnover,
            "estimated_transaction_cost": estimated_cost,
            "signal_capture_ratio": signal_capture_ratio(
                anchor_utility, final_utility
            ),
            "anchor_predicted_volatility": anchor_volatility,
            "optimized_predicted_volatility": optimized_volatility,
            "predicted_risk_reduction": (
                anchor_volatility - optimized_volatility
            ),
            "anchor_reallocation": float(
                0.5 * (weights - anchor).abs().sum()
            ),
            "maximum_anchor_weight_deviation": float(
                (weights - anchor).abs().max()
            ),
            "anchor_asset_count": int((anchor > tolerance).sum()),
            "objective_value": risk.variance(optimized_active),
        }
    )
    return OptimizationResult(
        weights=weights,
        solver=solver,
        constraints=report,
    )



def _optimize_blended_minimum_variance(
    expected_return: pd.Series,
    covariance: pd.DataFrame | PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
    *,
    anchor_weights: pd.Series | None,
    candidate_mask: pd.Series | None,
    exit_only_mask: pd.Series | None,
) -> OptimizationResult:
    if anchor_weights is None:
        raise OptimizationError(
            "blended_minimum_variance requires anchor_weights"
        )
    if candidate_mask is None:
        raise OptimizationError(
            "blended_minimum_variance requires a candidate_mask"
        )

    tolerance = float(constraint_config["constraint_tolerance"])
    anchor = _execution_aware_anchor(
        anchor_weights.reindex(expected_return.index),
        current,
        tradable,
        tolerance=tolerance,
    )
    anchor_report = constraint_report(
        anchor,
        benchmark,
        current,
        covariance,
        sectors,
        exposures,
        tradable,
        constraint_config,
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
    )
    if not anchor_report["passed"]:
        details = ", ".join(
            item["constraint"] for item in anchor_report["violations"]
        )
        raise OptimizationError(
            "executable signal anchor violates configured constraints: " + details
        )

    strength = float(optimizer_config["blend_strength"])
    # Keep the endpoint inside the same signal candidate universe as the anchor.
    # Using anchor>0 here would silently promote carried/frozen holdings to
    # candidates and can break the final candidate-weight-range check.
    endpoint_candidate_mask = candidate_mask.reindex(expected_return.index)
    if endpoint_candidate_mask.isna().any():
        raise OptimizationError("candidate_mask is missing optimization assets")
    endpoint_candidate_mask = endpoint_candidate_mask.astype(bool).rename("is_candidate")
    if strength == 0.0:
        endpoint_weights = anchor.copy()
        endpoint_solver: dict[str, Any] = {
            "backend": "anchor_only",
            "solver": None,
            "iterations": 0,
            "problem_cache_hit": False,
            "solver_wall_seconds": 0.0,
            "problem_compile_seconds": 0.0,
        }
    else:
        endpoint_optimizer = dict(optimizer_config)
        endpoint_optimizer.update(
            {
                "objective_mode": "minimum_variance",
                "risk_aversion": 1.0,
                "turnover_penalty": 0.0,
            }
        )
        zero_return = pd.Series(0.0, index=expected_return.index)
        endpoint = _optimize_cvxpy(
            zero_return,
            zero_return,
            covariance,
            benchmark,
            current,
            sectors,
            exposures,
            endpoint_candidate_mask,
            exit_only_mask,
            tradable,
            endpoint_optimizer,
            constraint_config,
        )
        endpoint_weights = endpoint.weights
        endpoint_solver = dict(endpoint.solver)

    weights = ((1.0 - strength) * anchor + strength * endpoint_weights).rename(
        "target_weight"
    )
    report = constraint_report(
        weights,
        benchmark,
        current,
        covariance,
        sectors,
        exposures,
        tradable,
        constraint_config,
        candidate_mask=candidate_mask,
        exit_only_mask=exit_only_mask,
    )
    if not report["passed"]:
        details = ", ".join(item["constraint"] for item in report["violations"])
        raise OptimizationError(
            "blended portfolio violates independently checked constraints: " + details
        )

    risk = as_portfolio_risk(covariance)
    anchor_volatility = float(
        np.sqrt(max(risk.variance(anchor.to_numpy(dtype=float)), 0.0))
    )
    endpoint_volatility = float(
        np.sqrt(max(risk.variance(endpoint_weights.to_numpy(dtype=float)), 0.0))
    )
    optimized_volatility = float(
        np.sqrt(max(risk.variance(weights.to_numpy(dtype=float)), 0.0))
    )
    risk_tolerance = max(100.0 * tolerance, 1.0e-7)
    if optimized_volatility > anchor_volatility + risk_tolerance:
        raise OptimizationError(
            "blended minimum-variance portfolio increases predicted total volatility"
        )

    solver = dict(endpoint_solver)
    solver.update(
        {
            "objective_mode": "blended_minimum_variance",
            "inner_objective_mode": "minimum_variance",
            "blend_strength": strength,
            "anchor_predicted_volatility": anchor_volatility,
            "minimum_variance_predicted_volatility": endpoint_volatility,
            "optimized_predicted_volatility": optimized_volatility,
            "predicted_risk_reduction": anchor_volatility - optimized_volatility,
            "anchor_reallocation": float(0.5 * (weights - anchor).abs().sum()),
            "maximum_anchor_weight_deviation": float((weights - anchor).abs().max()),
            "anchor_asset_count": int(endpoint_candidate_mask.sum()),
            "objective_value": risk.variance(weights.to_numpy(dtype=float)),
        }
    )
    return OptimizationResult(weights=weights, solver=solver, constraints=report)

def optimize_portfolio(
    expected_return: pd.Series,
    covariance: pd.DataFrame | PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
    signal_score: pd.Series | None = None,
    anchor_weights: pd.Series | None = None,
    candidate_mask: pd.Series | None = None,
    exit_only_mask: pd.Series | None = None,
    cost_model: dict[str, Any] | None = None,
) -> OptimizationResult:
    if candidate_mask is not None:
        candidate_mask = candidate_mask.reindex(expected_return.index)
        if candidate_mask.isna().any():
            raise OptimizationError("candidate_mask is missing optimization assets")
        candidate_mask = candidate_mask.astype(bool)
    elif constraint_config.get("candidate_weight_range") is not None:
        raise OptimizationError(
            "candidate_mask is required when candidate_weight_range is configured"
        )
    if exit_only_mask is not None:
        exit_only_mask = exit_only_mask.reindex(expected_return.index)
        if exit_only_mask.isna().any():
            raise OptimizationError("exit_only_mask is missing optimization assets")
        exit_only_mask = exit_only_mask.astype(bool)
        if exit_only_mask.any() and current is None:
            raise OptimizationError("exit-only constraints require current weights")
        if (exit_only_mask & ~tradable.reindex(expected_return.index).astype(bool)).any():
            raise OptimizationError("exit-only assets must be tradable")

    objective_mode = optimizer_config["objective_mode"]
    if objective_mode == "signal_preserving_minimum_variance":
        return _optimize_signal_preserving_minimum_variance(
            expected_return,
            covariance,
            benchmark,
            current,
            sectors,
            exposures,
            tradable,
            optimizer_config,
            constraint_config,
            linear_cost_bps=float(
                (cost_model or {"linear_cost_bps": 7.0})[
                    "linear_cost_bps"
                ]
            ),
            signal_score=signal_score,
            anchor_weights=anchor_weights,
            candidate_mask=candidate_mask,
            exit_only_mask=exit_only_mask,
        )
    if objective_mode == "blended_minimum_variance":
        return _optimize_blended_minimum_variance(
            expected_return,
            covariance,
            benchmark,
            current,
            sectors,
            exposures,
            tradable,
            optimizer_config,
            constraint_config,
            anchor_weights=anchor_weights,
            candidate_mask=candidate_mask,
            exit_only_mask=exit_only_mask,
        )
    if objective_mode in {"score_max_te", "lexicographic_signal_cost"}:
        if signal_score is None:
            raise OptimizationError(f"{objective_mode} requires a signal_score vector")
        objective_signal = signal_score.reindex(expected_return.index)
        if objective_signal.isna().any() or not np.isfinite(objective_signal).all():
            raise OptimizationError("signal_score contains missing or non-finite values")
    else:
        objective_signal = expected_return

    if objective_mode == "lexicographic_signal_cost":
        from .lexicographic import optimize_lexicographic_signal_cost

        return optimize_lexicographic_signal_cost(
            expected_return=expected_return,
            signal_score=objective_signal,
            risk_input=covariance,
            benchmark=benchmark,
            current=current,
            sectors=sectors,
            exposures=exposures,
            candidate_mask=candidate_mask,
            exit_only_mask=exit_only_mask,
            tradable=tradable,
            optimizer_config=optimizer_config,
            constraint_config=constraint_config,
            # Keep direct optimizer calls aligned with the ba875fc8 one-way
            # transaction-cost convention when no override is supplied.
            linear_cost_bps=float((cost_model or {"linear_cost_bps": 7.0})["linear_cost_bps"]),
        )

    backend = optimizer_config["solver_backend"]
    if backend in {"cvxpy", "clarabel_socp", "auto"}:
        try:
            return _optimize_cvxpy(
                expected_return,
                objective_signal,
                covariance,
                benchmark,
                current,
                sectors,
                exposures,
                candidate_mask,
                exit_only_mask,
                tradable,
                optimizer_config,
                constraint_config,
            )
        except OptimizationError as exc:
            if (
                backend in {"cvxpy", "clarabel_socp"}
                or optimizer_config.get("fallback_policy", "scipy_highs") == "error"
                or "backend is unavailable" not in str(exc)
            ):
                raise
    if objective_mode == "score_max_te":
        result = _optimize_score_highs(
            expected_return,
            objective_signal,
            covariance,
            benchmark,
            current,
            sectors,
            exposures,
            candidate_mask,
            exit_only_mask,
            tradable,
            optimizer_config,
            constraint_config,
        )
    else:
        result = _optimize_scipy(
            expected_return,
            objective_signal,
            covariance,
            benchmark,
            current,
            sectors,
            exposures,
            candidate_mask,
            exit_only_mask,
            tradable,
            optimizer_config,
            constraint_config,
        )
    result.solver["backend_fallback_used"] = bool(backend == "auto")
    return result
