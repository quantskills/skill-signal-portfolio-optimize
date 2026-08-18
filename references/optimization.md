# Optimization Method

## Objective

For `mean_variance` target weights `w`, benchmark weights `b`, calibrated expected returns `mu`, and annualized asset covariance `Sigma`, minimize:

```text
-(w' mu) + 0.5 * risk_aversion * (w-b)' Sigma (w-b)
          + turnover_penalty * sum(sqrt((w-current)^2 + epsilon^2))
```

This is benchmark-relative mean-variance optimization. The smoothed absolute turnover term improves numerical behavior but does not replace the exact configured turnover constraint.

For `score_max_te`, minimize negative standardized signal score plus turnover cost while
requiring an explicit tracking-error limit. This mode does not interpret a LightGBM rank or
single-factor score as an annualized expected return.

Use CVXPY with Clarabel when a quadratic tracking-error constraint is present and OSQP for a
pure quadratic program. When CVXPY is unavailable, `score_max_te` uses SciPy HiGHS with exact
L1 turnover auxiliaries. Frank-Wolfe linear oracles first find a tracking-error-feasible anchor.
Valid supporting hyperplanes then provide an outer objective bound, while protected scalar bisection from a strictly feasible
anchor to the covariance-ellipsoid boundary provides an inner feasible bound. The solver stops
only when their disclosed objective gap reaches the greater of configured `ftol` and `1e-5`, using `optimizer.max_iterations` as the certified cutting-plane budget.
This objective certificate does not relax any hard-constraint tolerance. `mean_variance`
continues to use SciPy SLSQP as the compatibility backend. Never switch backend after a
reported solver failure.

## Signal calibration

`rank_score` inputs are direction-adjusted, MAD-winsorized, converted to cross-sectional percentile ranks, standardized, and multiplied by `annualized_alpha_scale`. This scale is a research assumption, not a fitted expected return.

`expected_return` inputs are direction-adjusted and optionally winsorized, but are not standardized because that would destroy their return units. Set `signal.zscore: false`; the values must already match the annualized covariance units.

## Constraints

Supported constraints are:

- fully invested long-only weights
- maximum position weight
- maximum absolute active weight by stock
- industry active-weight ranges relative to the benchmark
- style active-exposure target ranges relative to the benchmark
- exact one-way turnover limit, defined as `0.5 * sum(abs(w-current))`
- annualized ex-ante tracking-error limit
- frozen weights for non-tradable assets

A position that becomes non-tradable after market drift cannot be sold back to its configured stock bound. The optimizer therefore fixes it at the exact current weight, reports it in `frozen_bound_exceptions`, and applies maximum-weight and maximum-active-weight checks to the remaining controllable assets. Raw portfolio maxima remain in diagnostics, so the exception is never hidden. Once the asset becomes tradable, the ordinary bounds apply again.

Each exposure accepts either `target_active+tolerance` or `min_active+max_active`. `SIZE` is mandatory in schema version 2. Legacy symmetric scalar or mapping limits remain accepted in schema version 1.

The optimization universe is the union of signal names, positive-weight benchmark constituents, and positive current holdings. Names without a signal receive zero expected return; the equal-weight signal baseline still selects only signaled names.

## Covariance repair

The runtime averages the matrix with its transpose, applies an eigenvalue floor, and reconstructs a symmetric positive-semidefinite matrix. Diagnostics disclose maximum asymmetry, minimum eigenvalues before and after repair, and the Frobenius repair norm.

## Failure behavior

The runtime raises an error instead of emitting optimized weights when:

- the solver does not report success;
- any hard constraint exceeds its tolerance;
- inputs are missing, duplicated, non-finite, or misaligned;
- bounds cannot support a fully invested portfolio;
- required current weights for frozen positions are absent.

The equal-weight signal portfolio remains a comparator and is never a hidden fallback.
