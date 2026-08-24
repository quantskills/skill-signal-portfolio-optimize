from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.optimize import LinearConstraint, NonlinearConstraint, minimize

from .cost import linear_transaction_cost, one_way_turnover
from .errors import OptimizationError
from .highs import solve_linear_over_tracking_error
from .risk import PortfolioRisk, as_portfolio_risk


@dataclass(frozen=True)
class StageSolution:
    decision: np.ndarray
    objective_value: float
    iterations: int
    status: int | str
    message: str
    duals: dict[str, float] | None = None
    optimality_gap: float | None = None
    risk_cut_matrix: np.ndarray | None = None
    risk_cut_bounds: np.ndarray | None = None


def signal_utility(
    weights: pd.Series | np.ndarray,
    benchmark: pd.Series | np.ndarray,
    signal_score: pd.Series | np.ndarray,
) -> float:
    return float(
        np.asarray(signal_score, dtype=float)
        @ (np.asarray(weights, dtype=float) - np.asarray(benchmark, dtype=float))
    )


def signal_utility_floor(primary_utility: float, minimum_capture: float) -> float:
    """Allow the configured fraction of absolute primary utility as maximum loss."""
    return float(primary_utility - (1.0 - minimum_capture) * abs(primary_utility))


def signal_capture_ratio(primary_utility: float, final_utility: float) -> float:
    scale = abs(primary_utility)
    if scale <= 1.0e-14:
        return 1.0 if final_utility >= primary_utility - 1.0e-12 else float("-inf")
    return float(1.0 - max(primary_utility - final_utility, 0.0) / scale)


def select_lexicographic_backend(requested: str) -> str:
    if requested not in {"auto", "cvxpy", "scipy_slsqp", "scipy_highs"}:
        raise OptimizationError(f"unsupported solver backend: {requested}")
    try:
        import cvxpy as cp

        has_clarabel = "CLARABEL" in set(cp.installed_solvers())
    except ImportError:
        has_clarabel = False
    if requested == "cvxpy":
        if not has_clarabel:
            raise OptimizationError(
                "CVXPY with CLARABEL is unavailable for lexicographic optimization"
            )
        return "cvxpy_clarabel"
    if requested == "auto" and has_clarabel:
        return "cvxpy_clarabel"
    return "scipy_highs_lexicographic"


def _risk_for_index(risk: PortfolioRisk, index: pd.Index) -> PortfolioRisk:
    if risk.tickers.equals(index):
        return risk
    if set(risk.tickers) != set(index):
        raise OptimizationError("risk universe does not match optimizer assets")
    if risk.form == "asset_covariance":
        assert risk.asset_covariance is not None
        return PortfolioRisk(
            form=risk.form,
            tickers=index,
            asset_covariance=risk.asset_covariance.reindex(index=index, columns=index),
            diagnostics=risk.diagnostics,
            input_hashes=risk.input_hashes,
        )
    assert risk.exposures is not None
    assert risk.factor_covariance is not None
    assert risk.specific_variance is not None
    return PortfolioRisk(
        form=risk.form,
        tickers=index,
        exposures=risk.exposures.reindex(index),
        factor_covariance=risk.factor_covariance,
        specific_variance=risk.specific_variance.reindex(index),
        diagnostics=risk.diagnostics,
        input_hashes=risk.input_hashes,
    )


def _linear_problem(
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    tradable: pd.Series,
    constraint_config: dict[str, Any],
    *,
    require_turnover_auxiliary: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Any, bool]:
    from .optimizer import _build_linear_system

    local_constraints = dict(constraint_config)
    if current is None:
        local_constraints["max_turnover"] = None
    elif require_turnover_auxiliary and local_constraints["max_turnover"] is None:
        local_constraints["max_turnover"] = 1.0
    return _build_linear_system(
        benchmark,
        current,
        sectors,
        exposures,
        candidate_mask,
        tradable,
        local_constraints,
    )


def _feasible_start(
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    bounds: Any,
) -> np.ndarray:
    from .optimizer import _linear_feasible_start

    return _linear_feasible_start(a_eq, b_eq, a_ub, b_ub, bounds)


def _constraints(
    *,
    risk: PortfolioRisk,
    benchmark_values: np.ndarray,
    max_tracking_error: float | None,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    utility_coefficients: np.ndarray | None = None,
    utility_floor: float | None = None,
) -> list[Any]:
    result: list[Any] = [LinearConstraint(a_eq, b_eq, b_eq)]
    if len(a_ub):
        result.append(LinearConstraint(a_ub, -np.inf, b_ub))
    if utility_coefficients is not None and utility_floor is not None:
        result.append(
            LinearConstraint(utility_coefficients, utility_floor, np.inf)
        )
    if max_tracking_error is not None:
        n_assets = len(benchmark_values)

        def variance(decision: np.ndarray) -> float:
            return risk.variance(decision[:n_assets] - benchmark_values)

        def gradient(decision: np.ndarray) -> np.ndarray:
            values = np.zeros_like(decision)
            values[:n_assets] = risk.gradient(
                decision[:n_assets] - benchmark_values
            )
            return values

        result.append(
            NonlinearConstraint(
                variance,
                -np.inf,
                float(max_tracking_error) ** 2,
                jac=gradient,
            )
        )
    return result


def _make_risk_feasible(
    initial: np.ndarray,
    constraints_without_risk: list[Any],
    bounds: Any,
    risk: PortfolioRisk,
    benchmark_values: np.ndarray,
    max_tracking_error: float | None,
    optimizer_config: dict[str, Any],
    tolerance: float,
) -> np.ndarray:
    if max_tracking_error is None:
        return initial
    n_assets = len(benchmark_values)

    def variance(decision: np.ndarray) -> float:
        return risk.variance(decision[:n_assets] - benchmark_values)

    def gradient(decision: np.ndarray) -> np.ndarray:
        values = np.zeros_like(decision)
        values[:n_assets] = risk.gradient(decision[:n_assets] - benchmark_values)
        return values

    limit_squared = float(max_tracking_error) ** 2
    if variance(initial) <= limit_squared + tolerance:
        return initial
    result = minimize(
        variance,
        initial,
        method="SLSQP",
        jac=gradient,
        bounds=bounds,
        constraints=constraints_without_risk,
        options={
            "maxiter": int(optimizer_config["max_iterations"]),
            "ftol": float(optimizer_config["ftol"]),
            "disp": False,
        },
    )
    if not result.success or variance(result.x) > limit_squared + tolerance:
        raise OptimizationError("tracking-error limit is infeasible with linear constraints")
    return np.asarray(result.x, dtype=float)


def solve_primary_signal_problem_scipy(
    *,
    signal_score: pd.Series,
    risk: PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
) -> StageSolution:
    n_assets = len(benchmark)
    a_eq, b_eq, a_ub, b_ub, bounds, with_turnover = _linear_problem(
        benchmark,
        current,
        sectors,
        exposures,
        candidate_mask,
        tradable,
        constraint_config,
        require_turnover_auxiliary=False,
    )
    coefficients = np.zeros(len(bounds.lb), dtype=float)
    coefficients[:n_assets] = -signal_score.to_numpy(dtype=float)
    solved = solve_linear_over_tracking_error(
        objective=coefficients,
        a_eq=a_eq,
        b_eq=b_eq,
        a_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        risk=risk,
        benchmark_values=benchmark.to_numpy(dtype=float),
        max_tracking_error=constraint_config["max_tracking_error"],
        constraint_tolerance=float(constraint_config["constraint_tolerance"]),
        objective_tolerance=max(float(optimizer_config["ftol"]), 1.0e-4),
        iteration_budget=int(optimizer_config["max_iterations"]),
        n_assets=n_assets,
        current_values=(
            None if current is None else current.to_numpy(dtype=float)
        ),
        with_turnover=with_turnover,
    )
    return StageSolution(
        decision=solved.decision,
        objective_value=solved.objective_value,
        iterations=solved.iterations + solved.highs_iterations,
        status=solved.status,
        message=solved.message,
        optimality_gap=max(0.0, solved.objective_value - solved.lower_bound),
        risk_cut_matrix=solved.risk_cut_matrix,
        risk_cut_bounds=solved.risk_cut_bounds,
    )


def solve_secondary_cost_problem_scipy(
    *,
    primary: StageSolution,
    primary_utility: float,
    utility_floor: float,
    signal_score: pd.Series,
    risk: PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
    linear_cost_bps: float,
) -> StageSolution:
    n_assets = len(benchmark)
    a_eq, b_eq, a_ub, b_ub, bounds, with_turnover = _linear_problem(
        benchmark,
        current,
        sectors,
        exposures,
        candidate_mask,
        tradable,
        constraint_config,
        require_turnover_auxiliary=True,
    )
    utility_coefficients = np.zeros(len(bounds.lb), dtype=float)
    utility_coefficients[:n_assets] = signal_score.to_numpy(dtype=float)
    numerical_utility_tolerance = min(
        float(constraint_config["constraint_tolerance"]),
        max(100.0 * float(optimizer_config["ftol"]), 1.0e-6),
    )
    numerical_utility_floor = utility_floor - numerical_utility_tolerance
    utility_absolute_floor = numerical_utility_floor + float(
        signal_score.to_numpy(dtype=float) @ benchmark.to_numpy(dtype=float)
    )
    utility_row = -utility_coefficients.reshape(1, -1)
    a_ub = utility_row if not len(a_ub) else np.vstack([a_ub, utility_row])
    b_ub = np.append(b_ub, -utility_absolute_floor)

    decision = np.zeros(len(bounds.lb), dtype=float)
    decision[:n_assets] = primary.decision[:n_assets]
    current_values = None if current is None else current.to_numpy(dtype=float)
    if with_turnover:
        assert current_values is not None
        decision[n_assets:] = np.abs(decision[:n_assets] - current_values)
    reference = (
        benchmark.to_numpy(dtype=float)
        if current_values is None
        else current_values
    )
    stability = float(optimizer_config["stability_penalty"])
    linear_objective = np.zeros(len(bounds.lb), dtype=float)
    if with_turnover:
        linear_objective[n_assets:] = (
            0.5 * float(linear_cost_bps) / 10000.0
        )
    tolerance = max(float(optimizer_config["ftol"]), 1.0e-7)
    gradient = linear_objective.copy()
    gradient[:n_assets] += 2.0 * stability * (
        decision[:n_assets] - reference
    )
    oracle = solve_linear_over_tracking_error(
        objective=gradient,
        a_eq=a_eq,
        b_eq=b_eq,
        a_ub=a_ub,
        b_ub=b_ub,
        bounds=bounds,
        risk=risk,
        benchmark_values=benchmark.to_numpy(dtype=float),
        max_tracking_error=None,
        constraint_tolerance=float(constraint_config["constraint_tolerance"]),
        objective_tolerance=tolerance,
        iteration_budget=int(optimizer_config["max_iterations"]),
        n_assets=n_assets,
        current_values=current_values,
        with_turnover=with_turnover,
    )
    total_iterations = oracle.iterations + oracle.highs_iterations
    direction = oracle.decision - decision
    oracle_gap = max(0.0, oracle.objective_value - oracle.lower_bound)
    start_objective = float(
        linear_objective @ decision
        + stability * np.square(decision[:n_assets] - reference).sum()
    )
    lower_bound = float(start_objective + gradient @ direction - oracle_gap)
    weight_direction = direction[:n_assets]
    feasible_step_cap = 1.0
    max_tracking_error = constraint_config["max_tracking_error"]
    if max_tracking_error is not None:
        active_start = decision[:n_assets] - benchmark.to_numpy(dtype=float)
        effective_limit = float(max_tracking_error) + 0.95 * float(
            constraint_config["constraint_tolerance"]
        )
        if risk.variance(active_start + weight_direction) > effective_limit**2:
            quadratic = risk.variance(weight_direction)
            linear = float(risk.gradient(active_start) @ weight_direction)
            constant = float(risk.variance(active_start) - effective_limit**2)
            if constant > max(1.0e-14, float(constraint_config["constraint_tolerance"]) ** 2):
                raise OptimizationError("secondary start is outside its TE numerical boundary")
            lower_step = 0.0
            upper_step = 1.0
            for _boundary_iteration in range(80):
                midpoint = 0.5 * (lower_step + upper_step)
                value = quadratic * midpoint**2 + linear * midpoint + constant
                if value <= 0.0:
                    lower_step = midpoint
                else:
                    upper_step = midpoint
            feasible_step_cap = lower_step
    max_turnover = constraint_config["max_turnover"]
    if current_values is not None and max_turnover is not None:
        turnover_limit = float(max_turnover) + float(
            constraint_config["constraint_tolerance"]
        )

        def turnover_at(step_value: float) -> float:
            weights = decision[:n_assets] + step_value * weight_direction
            return float(0.5 * np.abs(weights - current_values).sum())

        if turnover_at(0.0) > turnover_limit + 1.0e-12:
            raise OptimizationError(
                "secondary start is outside its turnover numerical boundary"
            )
        if turnover_at(feasible_step_cap) > turnover_limit:
            lower_step = 0.0
            upper_step = feasible_step_cap
            for _turnover_boundary_iteration in range(80):
                midpoint = 0.5 * (lower_step + upper_step)
                if turnover_at(midpoint) <= turnover_limit:
                    lower_step = midpoint
                else:
                    upper_step = midpoint
            feasible_step_cap = lower_step
    signal_limit = utility_floor - 0.25 * float(
        constraint_config["constraint_tolerance"]
    ) * max(abs(float(primary_utility)), 1.0)
    utility_at_start = signal_utility(
        decision[:n_assets],
        benchmark.to_numpy(dtype=float),
        signal_score.to_numpy(dtype=float),
    )
    utility_direction = float(
        signal_score.to_numpy(dtype=float) @ weight_direction
    )
    utility_at_cap = utility_at_start + feasible_step_cap * utility_direction
    if utility_at_cap < signal_limit:
        if utility_direction >= 0.0:
            raise OptimizationError("secondary line search cannot preserve signal utility floor")
        feasible_step_cap = min(
            feasible_step_cap,
            max(0.0, (utility_at_start - signal_limit) / (-utility_direction)),
        )
    offset = decision[:n_assets] - reference
    derivative_at_zero = float(
        linear_objective @ direction
        + 2.0 * stability * (offset @ weight_direction)
    )
    curvature = stability * float(weight_direction @ weight_direction)
    if curvature > 0.0:
        step = float(
            np.clip(-derivative_at_zero / (2.0 * curvature), 0.0, feasible_step_cap)
        )
    else:
        step = feasible_step_cap if derivative_at_zero < 0.0 else 0.0
    decision = decision + step * direction
    if with_turnover:
        assert current_values is not None
        decision[n_assets:] = np.abs(decision[:n_assets] - current_values)
    final_utility = signal_utility(
        decision[:n_assets],
        benchmark.to_numpy(dtype=float),
        signal_score.to_numpy(dtype=float),
    )
    feasibility_tolerance = max(
        float(constraint_config["constraint_tolerance"]), 1.0e-9
    )
    if final_utility < utility_floor - feasibility_tolerance:
        raise OptimizationError(
            "secondary solution violates signal utility floor: "
            f"{final_utility:.12g} < {utility_floor:.12g}"
        )
    objective_value = float(
        linear_objective @ decision
        + stability * np.square(decision[:n_assets] - reference).sum()
    )
    final_gap = max(0.0, objective_value - lower_bound)
    return StageSolution(
        decision=decision,
        objective_value=objective_value,
        iterations=total_iterations,
        status=0,
        message=(
            "linear-relaxation lower bound and TE-truncated exact line search; "
            f"step={step:.12g}, bound={final_gap:.12g}"
        ),
        optimality_gap=final_gap,
    )


def _cvxpy_risk_expression(cp: Any, risk: PortfolioRisk, active: Any) -> Any:
    if risk.form == "asset_covariance":
        assert risk.asset_covariance is not None
        return cp.quad_form(
            active, cp.psd_wrap(risk.asset_covariance.to_numpy(dtype=float))
        )
    assert risk.exposures is not None
    assert risk.factor_covariance is not None
    assert risk.specific_variance is not None
    x = risk.exposures.to_numpy(dtype=float)
    f = risk.factor_covariance.to_numpy(dtype=float)
    d = risk.specific_variance.to_numpy(dtype=float)
    return cp.quad_form(x.T @ active, cp.psd_wrap(f)) + cp.sum(
        cp.multiply(d, cp.square(active))
    )


def _solve_cvxpy(
    *,
    signal_score: pd.Series,
    risk: PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
    linear_cost_bps: float,
) -> tuple[StageSolution, StageSolution, float, float]:
    import cvxpy as cp

    n_assets = len(benchmark)
    a_eq, b_eq, a_ub, b_ub, bounds, with_turnover = _linear_problem(
        benchmark,
        current,
        sectors,
        exposures,
        candidate_mask,
        tradable,
        constraint_config,
        require_turnover_auxiliary=True,
    )
    decision = cp.Variable(len(bounds.lb))
    weights = decision[:n_assets]
    active = weights - benchmark.to_numpy(dtype=float)
    hard_constraints: list[Any] = [
        a_eq @ decision == b_eq,
        decision >= bounds.lb,
        decision <= bounds.ub,
    ]
    if len(a_ub):
        hard_constraints.append(a_ub @ decision <= b_ub)
    if constraint_config["max_tracking_error"] is not None:
        hard_constraints.append(
            _cvxpy_risk_expression(cp, risk, active)
            <= float(constraint_config["max_tracking_error"]) ** 2
        )
    score = signal_score.to_numpy(dtype=float)
    utility = score @ active
    options = {
        "max_iter": int(optimizer_config["max_iterations"]),
        "tol_gap_abs": float(optimizer_config["ftol"]),
        "tol_feas": max(float(optimizer_config["ftol"]), 1.0e-10),
    }
    primary_problem = cp.Problem(cp.Maximize(utility), hard_constraints)
    try:
        primary_problem.solve(solver="CLARABEL", warm_start=False, verbose=False, **options)
    except Exception as exc:
        raise OptimizationError(f"CVXPY CLARABEL primary solve failed: {exc}") from exc
    if primary_problem.status != cp.OPTIMAL or decision.value is None:
        raise OptimizationError(
            f"CVXPY CLARABEL primary solve did not return optimum: {primary_problem.status}"
        )
    primary_decision = np.asarray(decision.value, dtype=float)
    primary_utility = signal_utility(primary_decision[:n_assets], benchmark, signal_score)
    floor = signal_utility_floor(
        primary_utility, float(optimizer_config["minimum_signal_capture"])
    )
    primary = StageSolution(
        decision=primary_decision,
        objective_value=-primary_utility,
        iterations=int(primary_problem.solver_stats.num_iters or 0),
        status=str(primary_problem.status),
        message=str(primary_problem.status),
    )
    reference = benchmark.to_numpy(dtype=float) if current is None else current.to_numpy(dtype=float)
    secondary_objective = float(optimizer_config["stability_penalty"]) * cp.sum_squares(
        weights - reference
    )
    if with_turnover:
        secondary_objective += 0.5 * float(linear_cost_bps) / 10000.0 * cp.sum(
            decision[n_assets:]
        )
    utility_constraint = utility >= floor
    secondary_problem = cp.Problem(
        cp.Minimize(secondary_objective), [*hard_constraints, utility_constraint]
    )
    try:
        secondary_problem.solve(solver="CLARABEL", warm_start=False, verbose=False, **options)
    except Exception as exc:
        raise OptimizationError(f"CVXPY CLARABEL secondary solve failed: {exc}") from exc
    if secondary_problem.status != cp.OPTIMAL or decision.value is None:
        raise OptimizationError(
            "CVXPY CLARABEL secondary solve did not return optimum: "
            f"{secondary_problem.status}"
        )
    secondary = StageSolution(
        decision=np.asarray(decision.value, dtype=float),
        objective_value=float(secondary_problem.value),
        iterations=int(secondary_problem.solver_stats.num_iters or 0),
        status=str(secondary_problem.status),
        message=str(secondary_problem.status),
        duals={"signal_utility_floor": float(utility_constraint.dual_value)},
    )
    return primary, secondary, primary_utility, floor


def optimize_lexicographic_signal_cost(
    *,
    expected_return: pd.Series,
    signal_score: pd.Series,
    risk_input: pd.DataFrame | PortfolioRisk,
    benchmark: pd.Series,
    current: pd.Series | None,
    sectors: pd.Series | None,
    exposures: pd.DataFrame | None,
    candidate_mask: pd.Series | None,
    tradable: pd.Series,
    optimizer_config: dict[str, Any],
    constraint_config: dict[str, Any],
    linear_cost_bps: float,
) -> Any:
    from .diagnostics import constraint_report
    from .optimizer import OptimizationResult

    original_index = expected_return.index
    index = pd.Index(sorted(original_index), name=original_index.name)
    expected_return = expected_return.reindex(index)
    signal_score = signal_score.reindex(index)
    benchmark = benchmark.reindex(index)
    current = None if current is None else current.reindex(index)
    sectors = None if sectors is None else sectors.reindex(index)
    exposures = None if exposures is None else exposures.reindex(index)
    candidate_mask = None if candidate_mask is None else candidate_mask.reindex(index)
    tradable = tradable.reindex(index)
    risk = _risk_for_index(as_portfolio_risk(risk_input), index)
    backend = select_lexicographic_backend(optimizer_config["solver_backend"])
    if backend == "cvxpy_clarabel":
        primary, secondary, primary_utility, utility_floor = _solve_cvxpy(
            signal_score=signal_score,
            risk=risk,
            benchmark=benchmark,
            current=current,
            sectors=sectors,
            exposures=exposures,
            candidate_mask=candidate_mask,
            tradable=tradable,
            optimizer_config=optimizer_config,
            constraint_config=constraint_config,
            linear_cost_bps=linear_cost_bps,
        )
    else:
        primary = solve_primary_signal_problem_scipy(
            signal_score=signal_score,
            risk=risk,
            benchmark=benchmark,
            current=current,
            sectors=sectors,
            exposures=exposures,
            candidate_mask=candidate_mask,
            tradable=tradable,
            optimizer_config=optimizer_config,
            constraint_config=constraint_config,
        )
        primary_utility = signal_utility(
            primary.decision[: len(index)], benchmark, signal_score
        )
        utility_floor = signal_utility_floor(
            primary_utility, float(optimizer_config["minimum_signal_capture"])
        )
        secondary = solve_secondary_cost_problem_scipy(
            primary=primary,
            primary_utility=primary_utility,
            utility_floor=utility_floor,
            signal_score=signal_score,
            risk=risk,
            benchmark=benchmark,
            current=current,
            sectors=sectors,
            exposures=exposures,
            candidate_mask=candidate_mask,
            tradable=tradable,
            optimizer_config=optimizer_config,
            constraint_config=constraint_config,
            linear_cost_bps=linear_cost_bps,
        )
    weights = pd.Series(
        secondary.decision[: len(index)], index=index, name="target_weight"
    )
    final_utility = signal_utility(weights, benchmark, signal_score)
    capture = signal_capture_ratio(primary_utility, final_utility)
    report = constraint_report(
        weights,
        benchmark,
        current,
        risk,
        sectors,
        exposures,
        tradable,
        constraint_config,
        candidate_mask=candidate_mask,
    )
    tolerance = max(float(constraint_config["constraint_tolerance"]), 1.0e-9)
    if not report["passed"]:
        details = ", ".join(item["constraint"] for item in report["violations"])
        raise OptimizationError(
            f"lexicographic result violates hard constraints: {details}"
        )
    if final_utility < utility_floor - tolerance:
        raise OptimizationError("lexicographic result violates minimum signal capture")
    turnover = one_way_turnover(weights, current)
    cost = linear_transaction_cost(turnover, linear_cost_bps)
    solver = {
        "backend": backend,
        "solver": "CLARABEL" if backend == "cvxpy_clarabel" else "HiGHS",
        "objective_mode": "lexicographic_signal_cost",
        "success": True,
        "status": "optimal",
        "message": "both lexicographic stages solved successfully",
        "iterations": primary.iterations + secondary.iterations,
        "primary_status": primary.status,
        "secondary_status": secondary.status,
        "primary_iterations": primary.iterations,
        "secondary_iterations": secondary.iterations,
        "primary_optimality_gap": primary.optimality_gap,
        "primary_optimality_gap_bound_type": (
            None if backend == "cvxpy_clarabel" else "HiGHS outer-inner certified gap"
        ),
        "secondary_optimality_gap": secondary.optimality_gap,
        "secondary_optimality_gap_bound_type": (
            None
            if backend == "cvxpy_clarabel"
            else "final objective minus convex linear-relaxation lower bound"
        ),
        "primary_signal_utility": primary_utility,
        "signal_utility_floor": utility_floor,
        "signal_utility_solver_floor": (
            utility_floor
            if backend == "cvxpy_clarabel"
            else utility_floor - min(
                float(constraint_config["constraint_tolerance"]),
                max(100.0 * float(optimizer_config["ftol"]), 1.0e-6),
            )
        ),
        "signal_utility_tolerance": (
            0.0
            if backend == "cvxpy_clarabel"
            else min(
                float(constraint_config["constraint_tolerance"]),
                max(100.0 * float(optimizer_config["ftol"]), 1.0e-6),
            )
        ),
        "final_signal_utility": final_utility,
        "signal_capture_ratio": capture,
        "minimum_signal_capture": float(optimizer_config["minimum_signal_capture"]),
        "objective_value": secondary.objective_value,
        "estimated_transaction_cost": cost,
        "one_way_turnover": turnover,
        "linear_cost_bps": float(linear_cost_bps),
        "stability_penalty": float(optimizer_config["stability_penalty"]),
        "risk_form": risk.form,
        "used_turnover_auxiliary_variables": current is not None,
        "turnover_constraint_form": "exact_l1_auxiliary" if current is not None else None,
        "constraint_duals": secondary.duals,
        "constraint_duals_unavailable_reason": (
            None if secondary.duals is not None else "HiGHS linear-relaxation backend does not expose stable constraint duals"
        ),
        "kkt_residual": None,
        "kkt_residual_unavailable_reason": (
            None if backend == "cvxpy_clarabel" else "HiGHS linear-relaxation backend does not expose a complete KKT certificate"
        ),
        "backend_selected_before_solve": True,
        "backend_fallback_used": False,
    }
    return OptimizationResult(
        weights=weights.reindex(original_index), solver=solver, constraints=report
    )
