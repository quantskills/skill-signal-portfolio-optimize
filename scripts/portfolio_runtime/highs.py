from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import Bounds, linprog

from .errors import OptimizationError
from .risk import PortfolioRisk


@dataclass(frozen=True)
class HighsLinearSolution:
    decision: np.ndarray
    objective_value: float
    lower_bound: float
    iterations: int
    cutting_planes: int
    highs_iterations: int
    risk_anchor_iterations: int
    status: int
    message: str
    risk_cut_matrix: np.ndarray
    risk_cut_bounds: np.ndarray


def solve_linear_over_tracking_error(
    *,
    objective: np.ndarray,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: Bounds,
    risk: PortfolioRisk,
    benchmark_values: np.ndarray,
    max_tracking_error: float | None,
    constraint_tolerance: float,
    objective_tolerance: float,
    iteration_budget: int,
    n_assets: int,
    current_values: np.ndarray | None,
    with_turnover: bool,
    feasible_anchor: np.ndarray | None = None,
    risk_tolerance_fraction: float = 0.9,
) -> HighsLinearSolution:
    """Minimize a linear objective over linear constraints and a TE ellipsoid."""

    scipy_bounds = list(zip(bounds.lb.tolist(), bounds.ub.tolist(), strict=True))
    limit = None if max_tracking_error is None else float(max_tracking_error)
    if not 0.0 <= risk_tolerance_fraction < 1.0:
        raise OptimizationError("risk tolerance fraction must be in [0, 1)")
    effective_limit = (
        None
        if limit is None
        else limit + risk_tolerance_fraction * constraint_tolerance
    )
    base_inequality_count = len(a_ub)
    local_a_ub = a_ub.copy()
    local_b_ub = b_ub.copy()
    cutting_planes = 0
    total_highs_iterations = 0
    risk_anchor_iterations = 0

    def solve_lp(
        coefficients: np.ndarray, local_a_ub: np.ndarray, local_b_ub: np.ndarray
    ) -> Any:
        nonlocal total_highs_iterations
        result = linprog(
            coefficients,
            A_ub=local_a_ub if len(local_a_ub) else None,
            b_ub=local_b_ub if len(local_a_ub) else None,
            A_eq=a_eq,
            b_eq=b_eq,
            bounds=scipy_bounds,
            method="highs",
            options={
                "primal_feasibility_tolerance": 1.0e-9,
                "dual_feasibility_tolerance": 1.0e-9,
                "ipm_optimality_tolerance": 1.0e-10,
            },
        )
        total_highs_iterations += int(result.nit or 0)
        if not result.success:
            raise OptimizationError(f"HiGHS linear oracle failed: {result.message}")
        return result

    def tighten_turnover(decision: np.ndarray) -> np.ndarray:
        tightened = np.asarray(decision, dtype=float).copy()
        if with_turnover:
            if current_values is None:
                raise OptimizationError("turnover auxiliary variables require current weights")
            tightened[n_assets:] = np.abs(tightened[:n_assets] - current_values)
        return tightened

    def active(decision: np.ndarray) -> np.ndarray:
        return decision[:n_assets] - benchmark_values

    def variance(decision: np.ndarray) -> float:
        return risk.variance(active(decision))

    result = solve_lp(objective, local_a_ub, local_b_ub)
    final = tighten_turnover(result.x)
    optimality_gap = 0.0
    if limit is not None and variance(final) > (
        effective_limit
    ) ** 2:
        anchor = (
            final.copy()
            if feasible_anchor is None
            else tighten_turnover(feasible_anchor)
        )
        if feasible_anchor is not None and variance(anchor) > (
            effective_limit
        ) ** 2:
            raise OptimizationError(
                "supplied tracking-error anchor is infeasible: "
                f"variance={variance(anchor):.12g}, limit_squared={effective_limit**2:.12g}"
            )
        risk_target = (
            max(limit - 0.25 * constraint_tolerance, 0.0) ** 2
            if feasible_anchor is None
            else effective_limit**2
        )
        for risk_anchor_iterations in range(1, iteration_budget + 1):
            anchor_active = active(anchor)
            if risk.variance(anchor_active) <= risk_target:
                break
            gradient = np.zeros(len(bounds.lb), dtype=float)
            gradient[:n_assets] = risk.gradient(anchor_active)
            oracle = tighten_turnover(solve_lp(gradient, a_ub, b_ub).x)
            direction = oracle - anchor
            weight_direction = direction[:n_assets]
            denominator = risk.variance(weight_direction)
            directional = 0.5 * float(
                risk.gradient(anchor_active) @ weight_direction
            )
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
        best_feasible = anchor.copy()
        best_objective = float(objective @ best_feasible)
        for _ in range(iteration_budget):
            outside = tighten_turnover(result.x)
            outside_active = active(outside)
            outside_variance = risk.variance(outside_active)
            if outside_variance <= (
                effective_limit
            ) ** 2:
                candidate_objective = float(objective @ outside)
                if candidate_objective < best_objective:
                    best_feasible = outside
                    best_objective = candidate_objective
                optimality_gap = max(0.0, best_objective - float(result.fun))
                break

            if variance(anchor) > effective_limit**2:
                anchor = strict_anchor.copy()
            anchor_active = active(anchor)
            direction = outside - anchor
            weight_direction = direction[:n_assets]
            quadratic = risk.variance(weight_direction)
            linear = float(risk.gradient(anchor_active) @ weight_direction)
            constant = float(risk.variance(anchor_active) - effective_limit**2)
            endpoint = quadratic + linear + constant
            scalar_tolerance = max(1.0e-14, constraint_tolerance**2)
            if (
                quadratic <= 0.0
                or constant > scalar_tolerance
                or endpoint < -scalar_tolerance
            ):
                raise OptimizationError(
                    "cannot bracket the tracking-error boundary between feasible and outer points"
                )
            lower_step = 0.0
            upper_step = 1.0
            for _boundary_iteration in range(80):
                midpoint = 0.5 * (lower_step + upper_step)
                midpoint_value = (
                    quadratic * midpoint**2 + linear * midpoint + constant
                )
                if midpoint_value <= 0.0:
                    lower_step = midpoint
                else:
                    upper_step = midpoint
            feasible = tighten_turnover(anchor + lower_step * direction)
            feasible_objective = float(objective @ feasible)
            if feasible_objective < best_objective:
                best_feasible = feasible
                best_objective = feasible_objective
            optimality_gap = max(0.0, best_objective - float(result.fun))
            if optimality_gap <= objective_tolerance:
                break

            sigma_active = 0.5 * risk.gradient(outside_active)
            cut = np.zeros(len(bounds.lb), dtype=float)
            cut[:n_assets] = sigma_active
            cut_bound = float(
                sigma_active @ benchmark_values
                + effective_limit * np.sqrt(outside_variance)
            )
            local_a_ub = np.vstack([local_a_ub, cut])
            local_b_ub = np.append(local_b_ub, cut_bound)
            cutting_planes += 1
            result = solve_lp(objective, local_a_ub, local_b_ub)
            anchor = tighten_turnover(0.05 * anchor + 0.95 * best_feasible)
        else:
            raise OptimizationError(
                "HiGHS outer-inner oracle exceeded its cutting-plane budget; "
                f"objective gap {optimality_gap:.12g} exceeds {objective_tolerance:.12g}"
            )
        final = best_feasible

    return HighsLinearSolution(
        decision=final,
        objective_value=float(objective @ final),
        lower_bound=float(result.fun),
        iterations=cutting_planes + risk_anchor_iterations + 1,
        cutting_planes=cutting_planes,
        highs_iterations=total_highs_iterations,
        risk_anchor_iterations=risk_anchor_iterations,
        status=int(result.status),
        message=str(result.message),
        risk_cut_matrix=local_a_ub[base_inequality_count:].copy(),
        risk_cut_bounds=local_b_ub[base_inequality_count:].copy(),
    )
