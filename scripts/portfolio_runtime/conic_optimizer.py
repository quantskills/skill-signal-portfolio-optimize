from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from hashlib import sha256
from time import perf_counter
from typing import Any

import numpy as np
from scipy import sparse

from .errors import OptimizationError
from .risk import PortfolioRisk, as_portfolio_risk


@dataclass
class ConicSolveResult:
    decision: np.ndarray
    solver: dict[str, Any]


@dataclass
class _CompiledProblem:
    problem: Any
    signal: Any | None
    signal_floor: Any | None
    decision: Any
    linear_objective: Any
    equality_rhs: Any
    inequality_rhs: Any | None
    lower_bounds: Any
    upper_bounds: Any
    benchmark: Any
    current: Any | None
    tracking_error_limit: Any | None
    is_dpp: bool
    solve_count: int = 0


@dataclass
class LexicographicConicResult:
    primary_decision: np.ndarray
    secondary_decision: np.ndarray
    primary_utility: float
    utility_floor: float
    primary_objective_value: float
    secondary_objective_value: float
    primary_iterations: int
    secondary_iterations: int
    primary_status: str
    secondary_status: str
    utility_floor_dual: float | None
    diagnostics: dict[str, Any]


@dataclass
class _CompiledLexicographicProblem:
    primary_problem: Any
    secondary_problem: Any
    decision: Any
    signal: Any
    equality_rhs: Any
    inequality_rhs: Any | None
    lower_bounds: Any
    upper_bounds: Any
    benchmark: Any
    reference: Any
    utility_absolute_floor: Any
    utility_constraint: Any
    tracking_error_limit: Any | None
    is_dpp: bool
    solve_count: int = 0


_PROBLEM_CACHE: OrderedDict[str, _CompiledProblem] = OrderedDict()
_LEXICOGRAPHIC_CACHE: OrderedDict[str, _CompiledLexicographicProblem] = OrderedDict()


def clear_conic_problem_cache() -> None:
    """Clear the in-process cache used by repeated rolling solves."""
    _PROBLEM_CACHE.clear()
    _LEXICOGRAPHIC_CACHE.clear()


def _clean_long_only_numerics(
    decision: np.ndarray,
    *,
    n_assets: int,
    lower_bounds: np.ndarray,
    current: np.ndarray | None,
    tolerance: float,
) -> tuple[np.ndarray, float]:
    """Remove solver-scale negative weights without changing total stock weight."""
    cleaned = np.asarray(decision, dtype=float).copy()
    weights = cleaned[:n_assets]
    negative = weights < 0.0
    if not negative.any():
        return cleaned, 0.0
    minimum = float(weights[negative].min())
    if minimum < -float(tolerance):
        raise OptimizationError(
            f"CLARABEL returned a negative weight beyond tolerance: {minimum:.12g}"
        )

    clipped_mass = float(-weights[negative].sum())
    if clipped_mass > float(tolerance):
        raise OptimizationError(
            "CLARABEL aggregate negative weight exceeds numerical tolerance: "
            f"{clipped_mass:.12g}"
        )
    weights[negative] = 0.0
    slack = weights - np.asarray(lower_bounds[:n_assets], dtype=float)
    slack[negative] = 0.0
    donor = int(np.argmax(slack))
    if float(slack[donor]) + float(tolerance) < clipped_mass:
        raise OptimizationError(
            "cannot redistribute CLARABEL numerical clipping within lower bounds"
        )
    weights[donor] -= clipped_mass
    cleaned[:n_assets] = weights
    if len(cleaned) > n_assets and current is not None:
        cleaned[n_assets:] = np.abs(weights - np.asarray(current, dtype=float))
    return cleaned, clipped_mass


def _acceptable_status(cp: Any, status: Any) -> bool:
    return status in {cp.OPTIMAL, cp.OPTIMAL_INACCURATE}


def _import_cvxpy() -> Any:
    try:
        import cvxpy as cp
    except ImportError as exc:
        raise OptimizationError(
            "CVXPY backend is unavailable; install the dependencies from requirements.txt"
        ) from exc
    if "CLARABEL" not in set(cp.installed_solvers()):
        raise OptimizationError(
            "CLARABEL backend is unavailable; install the dependencies from requirements.txt"
        )
    return cp


def _array_digest(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = sha256()
    digest.update(str(array.shape).encode("ascii"))
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(array.tobytes())
    return digest.hexdigest()


def _sparse_digest(value: sparse.csc_matrix) -> str:
    matrix = value.tocsc()
    digest = sha256()
    digest.update(str(matrix.shape).encode("ascii"))
    digest.update(matrix.data.tobytes())
    digest.update(matrix.indices.tobytes())
    digest.update(matrix.indptr.tobytes())
    return digest.hexdigest()


def _cache_key(
    *,
    objective_mode: str,
    risk_aversion: float,
    turnover_penalty: float,
    has_current_l1: bool,
    a_eq: np.ndarray,
    a_ub: np.ndarray,
    risk_operator: sparse.csc_matrix,
    has_tracking_error: bool,
    has_signal_floor: bool = False,
) -> str:
    digest = sha256()
    for value in (
        objective_mode,
        repr(float(risk_aversion)),
        repr(float(turnover_penalty)),
        str(bool(has_current_l1)),
        str(bool(has_tracking_error)),
        str(bool(has_signal_floor)),
        _array_digest(a_eq),
        _array_digest(a_ub),
        _sparse_digest(risk_operator),
    ):
        digest.update(value.encode("ascii"))
        digest.update(b"\0")
    return digest.hexdigest()


def _build_problem(
    cp: Any,
    *,
    n_assets: int,
    objective_mode: str,
    risk_aversion: float,
    turnover_penalty: float,
    a_eq: np.ndarray,
    a_ub: np.ndarray,
    risk_operator: sparse.csc_matrix,
    has_tracking_error: bool,
    has_signal_floor: bool = False,
    has_current_l1: bool,
) -> _CompiledProblem:
    dimension = a_eq.shape[1]
    decision = cp.Variable(dimension)
    weights = decision[:n_assets]
    linear_objective = cp.Parameter(dimension)
    equality_rhs = cp.Parameter(a_eq.shape[0])
    inequality_rhs = cp.Parameter(a_ub.shape[0]) if len(a_ub) else None
    lower_bounds = cp.Parameter(dimension)
    upper_bounds = cp.Parameter(dimension)
    benchmark = cp.Parameter(n_assets)
    signal = cp.Parameter(n_assets) if has_signal_floor else None
    signal_floor = cp.Parameter() if has_signal_floor else None
    current = cp.Parameter(n_assets) if has_current_l1 else None

    equality_matrix = sparse.csc_matrix(a_eq)
    inequality_matrix = sparse.csc_matrix(a_ub)
    constraints: list[Any] = [
        equality_matrix @ decision == equality_rhs,
        decision >= lower_bounds,
        decision <= upper_bounds,
    ]
    if inequality_rhs is not None:
        constraints.append(inequality_matrix @ decision <= inequality_rhs)
    if signal is not None and signal_floor is not None:
        constraints.append(signal @ weights >= signal_floor)

    risk_weights = (
        weights
        if objective_mode == "minimum_variance"
        else weights - benchmark
    )
    risk_vector = risk_operator @ risk_weights
    objective = linear_objective @ decision
    if objective_mode in {"mean_variance", "minimum_variance", "signal_preserving_minimum_variance"}:
        objective += 0.5 * float(risk_aversion) * cp.sum_squares(risk_vector)
    if current is not None:
        objective += float(turnover_penalty) * cp.norm1(weights - current)
    if has_tracking_error:
        tracking_error_limit = cp.Parameter(nonneg=True)
        constraints.append(cp.norm(risk_vector, 2) <= tracking_error_limit)
    else:
        tracking_error_limit = None

    problem = cp.Problem(cp.Minimize(objective), constraints)
    return _CompiledProblem(
        problem=problem,
        decision=decision,
        linear_objective=linear_objective,
        equality_rhs=equality_rhs,
        inequality_rhs=inequality_rhs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        signal=signal,
        signal_floor=signal_floor,
        benchmark=benchmark,
        current=current,
        tracking_error_limit=tracking_error_limit,
        is_dpp=bool(problem.is_dpp()),
    )


def solve_clarabel_socp(
    *,
    expected_return: np.ndarray,
    objective_signal: np.ndarray,
    risk_input: PortfolioRisk | Any,
    benchmark: np.ndarray,
    current: np.ndarray | None,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    with_turnover_auxiliary: bool,
    optimizer_config: dict[str, Any],
    max_tracking_error: float | None,
    reported_backend: str = "cvxpy_clarabel_socp",
    signal_score: np.ndarray | None = None,
    signal_floor: float | None = None,
) -> ConicSolveResult:
    """Solve the convex portfolio problem without expanding factor risk."""
    cp = _import_cvxpy()
    risk = as_portfolio_risk(risk_input)
    risk_operator = risk.square_root_operator()
    n_assets = len(benchmark)
    dimension = a_eq.shape[1]
    if risk_operator.shape[1] != n_assets:
        raise OptimizationError("risk operator does not match optimization assets")

    objective_mode = str(optimizer_config["objective_mode"])
    risk_aversion = float(optimizer_config["risk_aversion"])
    turnover_penalty = float(optimizer_config["turnover_penalty"])
    has_signal_floor = objective_mode == "signal_preserving_minimum_variance"
    if has_signal_floor != (signal_score is not None and signal_floor is not None):
        raise OptimizationError(
            "signal_preserving_minimum_variance requires signal_score and signal_floor"
        )
    has_current_l1 = (
        current is not None and turnover_penalty > 0.0 and not with_turnover_auxiliary
    )
    key = _cache_key(
        objective_mode=objective_mode,
        risk_aversion=risk_aversion,
        turnover_penalty=turnover_penalty,
        has_current_l1=has_current_l1,
        a_eq=a_eq,
        has_signal_floor=has_signal_floor,
        a_ub=a_ub,
        risk_operator=risk_operator,
        has_tracking_error=max_tracking_error is not None,
    )
    cache_size = int(optimizer_config.get("conic_cache_size", 8))
    use_cache = cache_size > 0
    compiled = _PROBLEM_CACHE.pop(key, None) if use_cache else None
    cache_hit = compiled is not None
    compile_started = perf_counter()
    if compiled is None:
        compiled = _build_problem(
            cp,
            n_assets=n_assets,
            objective_mode=objective_mode,
            risk_aversion=risk_aversion,
            turnover_penalty=turnover_penalty,
            a_eq=a_eq,
            a_ub=a_ub,
            has_signal_floor=has_signal_floor,
            risk_operator=risk_operator,
            has_tracking_error=max_tracking_error is not None,
            has_current_l1=has_current_l1,
        )
    compile_seconds = perf_counter() - compile_started

    linear = np.zeros(dimension, dtype=float)
    if has_signal_floor:
        assert signal_score is not None and signal_floor is not None
    if objective_mode in {"mean_variance", "minimum_variance"}:
        linear[:n_assets] = -np.asarray(expected_return, dtype=float)
    elif objective_mode == "signal_preserving_minimum_variance":
        linear[:n_assets] = 0.0
    else:
        linear[:n_assets] = -np.asarray(objective_signal, dtype=float)
    if with_turnover_auxiliary:
        linear[n_assets:] = turnover_penalty

    compiled.linear_objective.value = linear
    compiled.equality_rhs.value = np.asarray(b_eq, dtype=float)
    if compiled.inequality_rhs is not None:
        compiled.inequality_rhs.value = np.asarray(b_ub, dtype=float)
    compiled.lower_bounds.value = np.asarray(lower_bounds, dtype=float)
    compiled.upper_bounds.value = np.asarray(upper_bounds, dtype=float)
    compiled.benchmark.value = np.asarray(benchmark, dtype=float)
    if compiled.signal is not None and compiled.signal_floor is not None:
        assert signal_score is not None and signal_floor is not None
        compiled.signal.value = np.asarray(signal_score, dtype=float)
        compiled.signal_floor.value = float(signal_floor)
    if compiled.current is not None:
        assert current is not None
        compiled.current.value = np.asarray(current, dtype=float)
    tracking_error_limit = compiled.tracking_error_limit
    if tracking_error_limit is not None:
        assert max_tracking_error is not None
        tracking_error_limit.value = float(max_tracking_error)

    if not cache_hit and current is not None:
        initial = np.zeros(dimension, dtype=float)
        initial[:n_assets] = np.asarray(current, dtype=float)
        if with_turnover_auxiliary:
            initial[n_assets:] = 0.0
        compiled.decision.value = initial

    options = {
        "max_iter": int(optimizer_config["max_iterations"]),
        "tol_gap_abs": float(optimizer_config["ftol"]),
        "tol_gap_rel": float(optimizer_config["ftol"]),
        "tol_feas": max(float(optimizer_config["ftol"]), 1.0e-10),
    }
    solve_started = perf_counter()
    try:
        compiled.problem.solve(
            solver="CLARABEL",
            warm_start=bool(optimizer_config.get("warm_start", True)),
            verbose=False,
            **options,
        )
    except Exception as exc:
        raise OptimizationError(f"CVXPY CLARABEL solve failed: {exc}") from exc
    solve_seconds = perf_counter() - solve_started
    if not _acceptable_status(cp, compiled.problem.status) or compiled.decision.value is None:
        raise OptimizationError(
            "CVXPY CLARABEL did not return an acceptable optimum: "
            f"{compiled.problem.status}"
        )
    cleaning_tolerance = max(
        100.0 * float(optimizer_config["ftol"]), 1.0e-6
    )
    cleaned_decision, clipped_weight_mass = _clean_long_only_numerics(
        np.asarray(compiled.decision.value, dtype=float),
        n_assets=n_assets,
        lower_bounds=np.asarray(lower_bounds, dtype=float),
        current=current,
        tolerance=cleaning_tolerance,
    )
    compiled.decision.value = cleaned_decision

    compiled.solve_count += 1
    if use_cache:
        _PROBLEM_CACHE[key] = compiled
        while len(_PROBLEM_CACHE) > cache_size:
            _PROBLEM_CACHE.popitem(last=False)

    stats = compiled.problem.solver_stats
    solver = {
        "backend": reported_backend,
        "signal_floor": signal_floor,
        "solver": "CLARABEL",
        "objective_mode": objective_mode,
        "success": True,
        "status": str(compiled.problem.status),
        "message": str(compiled.problem.status),
        "inaccurate_status_accepted": bool(compiled.problem.status == cp.OPTIMAL_INACCURATE),
        "iterations": int(stats.num_iters or 0),
        "objective_value": float(compiled.problem.value),
        "used_turnover_auxiliary_variables": with_turnover_auxiliary,
        "tracking_error_constraint_form": (
            f"{risk.form}_socp" if max_tracking_error is not None else None
        ),
        "risk_form": risk.form,
        "risk_operator_rows": int(risk_operator.shape[0]),
        "problem_cache_hit": cache_hit,
        "problem_cache_key": key[:16],
        "problem_cache_size": len(_PROBLEM_CACHE),
        "problem_is_dpp": compiled.is_dpp,
        "problem_compile_seconds": float(compile_seconds),
        "solver_wall_seconds": float(solve_seconds),
        "solver_setup_seconds": (
            None if stats.setup_time is None else float(stats.setup_time)
        ),
        "solver_reported_seconds": (
            None if stats.solve_time is None else float(stats.solve_time)
        ),
        "warm_start_requested": bool(optimizer_config.get("warm_start", True)),
        "numerical_clipped_weight_mass": float(clipped_weight_mass),
    }
    return ConicSolveResult(
        decision=cleaned_decision,
        solver=solver,
    )


def _lexicographic_cache_key(
    *,
    stability_penalty: float,
    linear_cost_bps: float,
    a_eq: np.ndarray,
    a_ub: np.ndarray,
    risk_operator: sparse.csc_matrix,
    has_tracking_error: bool,
) -> str:
    return _cache_key(
        objective_mode="lexicographic_signal_cost",
        risk_aversion=stability_penalty,
        turnover_penalty=linear_cost_bps,
        has_current_l1=False,
        a_eq=a_eq,
        a_ub=a_ub,
        risk_operator=risk_operator,
        has_tracking_error=has_tracking_error,
    )


def _build_lexicographic_problem(
    cp: Any,
    *,
    n_assets: int,
    stability_penalty: float,
    linear_cost_bps: float,
    a_eq: np.ndarray,
    a_ub: np.ndarray,
    risk_operator: sparse.csc_matrix,
    has_tracking_error: bool,
) -> _CompiledLexicographicProblem:
    dimension = a_eq.shape[1]
    decision = cp.Variable(dimension)
    weights = decision[:n_assets]
    signal = cp.Parameter(n_assets)
    equality_rhs = cp.Parameter(a_eq.shape[0])
    inequality_rhs = cp.Parameter(a_ub.shape[0]) if len(a_ub) else None
    lower_bounds = cp.Parameter(dimension)
    upper_bounds = cp.Parameter(dimension)
    benchmark = cp.Parameter(n_assets)
    reference = cp.Parameter(n_assets)
    utility_absolute_floor = cp.Parameter()

    hard_constraints: list[Any] = [
        sparse.csc_matrix(a_eq) @ decision == equality_rhs,
        decision >= lower_bounds,
        decision <= upper_bounds,
    ]
    if inequality_rhs is not None:
        hard_constraints.append(sparse.csc_matrix(a_ub) @ decision <= inequality_rhs)

    tracking_error_limit = None
    if has_tracking_error:
        tracking_error_limit = cp.Parameter(nonneg=True)
        hard_constraints.append(
            cp.norm(risk_operator @ (weights - benchmark), 2)
            <= tracking_error_limit
        )

    primary_problem = cp.Problem(cp.Maximize(signal @ weights), hard_constraints)
    secondary_objective = float(stability_penalty) * cp.sum_squares(
        weights - reference
    )
    if dimension > n_assets:
        secondary_objective += (
            0.5 * float(linear_cost_bps) / 10000.0
        ) * cp.sum(decision[n_assets:])
    utility_constraint = signal @ weights >= utility_absolute_floor
    secondary_problem = cp.Problem(
        cp.Minimize(secondary_objective),
        [*hard_constraints, utility_constraint],
    )
    return _CompiledLexicographicProblem(
        primary_problem=primary_problem,
        secondary_problem=secondary_problem,
        decision=decision,
        signal=signal,
        equality_rhs=equality_rhs,
        inequality_rhs=inequality_rhs,
        lower_bounds=lower_bounds,
        upper_bounds=upper_bounds,
        benchmark=benchmark,
        reference=reference,
        utility_absolute_floor=utility_absolute_floor,
        utility_constraint=utility_constraint,
        tracking_error_limit=tracking_error_limit,
        is_dpp=bool(primary_problem.is_dpp() and secondary_problem.is_dpp()),
    )


def solve_lexicographic_clarabel(
    *,
    signal_score: np.ndarray,
    risk_input: PortfolioRisk | Any,
    benchmark: np.ndarray,
    current: np.ndarray | None,
    a_eq: np.ndarray,
    b_eq: np.ndarray,
    a_ub: np.ndarray,
    b_ub: np.ndarray,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    optimizer_config: dict[str, Any],
    max_tracking_error: float | None,
    linear_cost_bps: float,
    minimum_signal_capture: float,
) -> LexicographicConicResult:
    """Solve both lexicographic stages with one cached conic structure."""
    cp = _import_cvxpy()
    risk = as_portfolio_risk(risk_input)
    risk_operator = risk.square_root_operator()
    n_assets = len(benchmark)
    if risk_operator.shape[1] != n_assets:
        raise OptimizationError("risk operator does not match optimization assets")

    stability_penalty = float(optimizer_config["stability_penalty"])
    key = _lexicographic_cache_key(
        stability_penalty=stability_penalty,
        linear_cost_bps=linear_cost_bps,
        a_eq=a_eq,
        a_ub=a_ub,
        risk_operator=risk_operator,
        has_tracking_error=max_tracking_error is not None,
    )
    cache_size = int(optimizer_config.get("conic_cache_size", 8))
    use_cache = cache_size > 0
    compiled = _LEXICOGRAPHIC_CACHE.pop(key, None) if use_cache else None
    cache_hit = compiled is not None
    compile_started = perf_counter()
    if compiled is None:
        compiled = _build_lexicographic_problem(
            cp,
            n_assets=n_assets,
            stability_penalty=stability_penalty,
            linear_cost_bps=linear_cost_bps,
            a_eq=a_eq,
            a_ub=a_ub,
            risk_operator=risk_operator,
            has_tracking_error=max_tracking_error is not None,
        )
    compile_seconds = perf_counter() - compile_started

    score = np.asarray(signal_score, dtype=float)
    benchmark_values = np.asarray(benchmark, dtype=float)
    reference = (
        benchmark_values if current is None else np.asarray(current, dtype=float)
    )
    compiled.signal.value = score
    compiled.equality_rhs.value = np.asarray(b_eq, dtype=float)
    if compiled.inequality_rhs is not None:
        compiled.inequality_rhs.value = np.asarray(b_ub, dtype=float)
    compiled.lower_bounds.value = np.asarray(lower_bounds, dtype=float)
    compiled.upper_bounds.value = np.asarray(upper_bounds, dtype=float)
    compiled.benchmark.value = benchmark_values
    compiled.reference.value = reference
    compiled.utility_absolute_floor.value = -1.0e100
    if compiled.tracking_error_limit is not None:
        assert max_tracking_error is not None
        compiled.tracking_error_limit.value = float(max_tracking_error)

    if not cache_hit:
        initial = np.zeros(len(lower_bounds), dtype=float)
        initial[:n_assets] = reference
        if len(lower_bounds) > n_assets and current is not None:
            initial[n_assets:] = 0.0
        compiled.decision.value = initial

    options = {
        "max_iter": int(optimizer_config["max_iterations"]),
        "tol_gap_abs": float(optimizer_config["ftol"]),
        "tol_gap_rel": float(optimizer_config["ftol"]),
        "tol_feas": max(float(optimizer_config["ftol"]), 1.0e-10),
    }
    warm_start = bool(optimizer_config.get("warm_start", True))
    primary_started = perf_counter()
    try:
        compiled.primary_problem.solve(
            solver="CLARABEL",
            warm_start=warm_start,
            verbose=False,
            **options,
        )
    except Exception as exc:
        raise OptimizationError(
            f"CVXPY CLARABEL primary solve failed: {exc}"
        ) from exc
    primary_seconds = perf_counter() - primary_started
    if not _acceptable_status(cp, compiled.primary_problem.status) or compiled.decision.value is None:
        raise OptimizationError(
            "CVXPY CLARABEL primary solve did not return an acceptable optimum: "
            f"{compiled.primary_problem.status}"
        )
    cleaning_tolerance = max(
        100.0 * float(optimizer_config["ftol"]), 1.0e-6
    )
    primary_decision, primary_clipped_weight_mass = _clean_long_only_numerics(
        np.asarray(compiled.decision.value, dtype=float),
        n_assets=n_assets,
        lower_bounds=np.asarray(lower_bounds, dtype=float),
        current=current,
        tolerance=cleaning_tolerance,
    )
    compiled.decision.value = primary_decision
    primary_utility = float(score @ (primary_decision[:n_assets] - benchmark_values))
    utility_floor = float(
        primary_utility
        - (1.0 - float(minimum_signal_capture)) * abs(primary_utility)
    )
    compiled.utility_absolute_floor.value = float(
        utility_floor + score @ benchmark_values
    )
    primary_stats = compiled.primary_problem.solver_stats

    secondary_started = perf_counter()
    try:
        compiled.secondary_problem.solve(
            solver="CLARABEL",
            warm_start=warm_start,
            verbose=False,
            **options,
        )
    except Exception as exc:
        raise OptimizationError(
            f"CVXPY CLARABEL secondary solve failed: {exc}"
        ) from exc
    secondary_seconds = perf_counter() - secondary_started
    if not _acceptable_status(cp, compiled.secondary_problem.status) or compiled.decision.value is None:
        raise OptimizationError(
            "CVXPY CLARABEL secondary solve did not return an acceptable optimum: "
            f"{compiled.secondary_problem.status}"
        )
    secondary_decision, secondary_clipped_weight_mass = _clean_long_only_numerics(
        np.asarray(compiled.decision.value, dtype=float),
        n_assets=n_assets,
        lower_bounds=np.asarray(lower_bounds, dtype=float),
        current=current,
        tolerance=cleaning_tolerance,
    )
    compiled.decision.value = secondary_decision
    secondary_stats = compiled.secondary_problem.solver_stats
    dual_value = compiled.utility_constraint.dual_value
    utility_floor_dual = (
        None if dual_value is None else float(np.asarray(dual_value).reshape(-1)[0])
    )

    compiled.solve_count += 1
    if use_cache:
        _LEXICOGRAPHIC_CACHE[key] = compiled
        while len(_LEXICOGRAPHIC_CACHE) > cache_size:
            _LEXICOGRAPHIC_CACHE.popitem(last=False)

    diagnostics = {
        "risk_form": risk.form,
        "risk_operator_rows": int(risk_operator.shape[0]),
        "tracking_error_constraint_form": (
            f"{risk.form}_socp" if max_tracking_error is not None else None
        ),
        "problem_cache_hit": cache_hit,
        "problem_cache_key": key[:16],
        "problem_cache_size": len(_LEXICOGRAPHIC_CACHE),
        "problem_is_dpp": compiled.is_dpp,
        "problem_compile_seconds": float(compile_seconds),
        "primary_solver_wall_seconds": float(primary_seconds),
        "secondary_solver_wall_seconds": float(secondary_seconds),
        "solver_wall_seconds": float(primary_seconds + secondary_seconds),
        "warm_start_requested": warm_start,
        "primary_numerical_clipped_weight_mass": float(
            primary_clipped_weight_mass
        ),
        "secondary_numerical_clipped_weight_mass": float(
            secondary_clipped_weight_mass
        ),
        "primary_inaccurate_status_accepted": bool(
            compiled.primary_problem.status == cp.OPTIMAL_INACCURATE
        ),
        "secondary_inaccurate_status_accepted": bool(
            compiled.secondary_problem.status == cp.OPTIMAL_INACCURATE
        ),
    }
    return LexicographicConicResult(
        primary_decision=primary_decision,
        secondary_decision=secondary_decision,
        primary_utility=primary_utility,
        utility_floor=utility_floor,
        primary_objective_value=-primary_utility,
        secondary_objective_value=float(compiled.secondary_problem.value),
        primary_iterations=int(primary_stats.num_iters or 0),
        secondary_iterations=int(secondary_stats.num_iters or 0),
        primary_status=str(compiled.primary_problem.status),
        secondary_status=str(compiled.secondary_problem.status),
        utility_floor_dual=utility_floor_dual,
        diagnostics=diagnostics,
    )
