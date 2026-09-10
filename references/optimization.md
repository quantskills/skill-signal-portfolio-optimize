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


The recommended `clarabel_socp` backend writes tracking error as the second-order-cone
constraint `norm(R * (w-b), 2) <= max_tracking_error`. For asset covariance, `R`
is a covariance square root. For factor risk `Sigma = X F X' + D`, the runtime
stacks `sqrt(F) * X'` and `sqrt(D)` directly, so the solver never constructs the
dense `N x N` asset covariance. The same representation is used in the quadratic
risk term of `mean_variance`.

Clarabel `optimal_inaccurate` results are accepted only when returned weights are finite and pass the runtime's independent hard-constraint, weight-sum, and signal-capture checks; the accepted status and any clipped numerical mass are recorded in diagnostics.

CVXPY problems are DPP-compatible and cached in-process by exact risk and linear-
constraint structure. On a cache hit, the runtime updates the signal, benchmark,
current-state bounds, and right-hand sides, preserves the preceding solution, and
requests a warm start. A universe, exposure, covariance, or constraint-matrix change
creates a new cache entry. `conic_cache_size` bounds retained structures.

The shipped fast examples set `fallback_policy: error`: missing Clarabel or a failed conic solve is
reported instead of starting a potentially long compatibility solve. Set
`fallback_policy: scipy_highs` with `solver_backend: auto` only when that behavior is
intentional. The `score_max_te` fallback uses exact L1 turnover auxiliaries, a risk-
feasible anchor, and auditable HiGHS ellipsoid cuts. `max_cutting_planes` is a hard
budget; exhausting it is an explicit error. `mean_variance` uses the existing SciPy
compatibility path. Every returned portfolio is independently checked against all
hard constraints.

## Schema 7 signal-preserving active-risk objective

Schema 7 keeps the executable equal-weight Top-N portfolio as anchor a. For
signal score s and benchmark b, define active signal utility as
U(w) = transpose(s) * (w-b). The optimizer requires:

    U(w) >= U(a) - (1-minimum_signal_capture) * abs(U(a))

It then minimizes benchmark-relative factor risk plus exact L1 turnover:

    0.5 * risk_aversion * (w-b)' Sigma (w-b)
      + turnover_penalty * sum(abs(w-current))

The shipped example uses minimum_signal_capture 0.97 and turnover_penalty
0.00035. Because the L1 term is twice one-way turnover, 0.00035 represents a
7 bps one-way cost. The result reports anchor and final signal utility, signal
capture ratio, active-risk reduction, anchor reallocation, and solver timing.
Schema 7 keeps the schema 6 candidate, stock, execution, and fail-closed
requirements, but does not use blend_strength.

## Schema 6 conservative anchor blend

Schema 6 is the recommended alpha-preserving mode. Let `a` be the executable
equal-weight Top-N signal portfolio and let `m` be the minimum-total-variance
portfolio over the same anchor support, subject to long-only, full-investment,
stock-bound, frozen-position, exit-only, and candidate-weight constraints. The
reported target is:

```text
w = (1 - blend_strength) * a + blend_strength * m
```

The default `blend_strength` is `0.10`. Setting it to zero reproduces the
executable signal anchor exactly; values such as `0.10` and `0.20` provide
an explicit, interpretable risk budget. The endpoint minimizes `w' Sigma w`,
not benchmark-relative tracking error. The signal determines Top-N membership
but is deliberately not reused to rank or concentrate names inside that set.

Both endpoints satisfy the same convex position and execution constraints, so
their blend remains feasible. The final portfolio is nevertheless checked
independently. It must not have higher predicted total volatility than the
anchor beyond numerical tolerance. Diagnostics report anchor, endpoint, and
final predicted volatility, risk reduction, reallocated weight, maximum
single-name deviation, and blend strength.

Schema 6 rejects hard tracking-error, turnover, industry-neutrality, and
style-neutrality constraints. Style and industry still affect the endpoint
through the factor risk model. Realized turnover remains governed by the
execution policy, such as StockDemo-compatible `keep=0.7`. This keeps the
optimizer focused on a modest risk adjustment instead of allowing risk
constraints to dominate the upstream alpha portfolio.

## Schema 5 lexicographic objective

For score vector `s`, target `w`, and benchmark `b`, define active signal utility as `U(w) = transpose(s) * (w-b)`. Stage 1 maximizes `U` under every hard constraint. If its optimum is `U_star` and configured capture is `c`, Stage 2 requires:

```text
U(w) >= U_star - (1-c) * abs(U_star)
```

This definition remains directionally correct when `U_star` is zero or negative. The numerical SciPy inequality uses a disclosed tolerance no greater than `constraints.constraint_tolerance` below this theoretical floor; the actual value is `min(constraint_tolerance, max(100*ftol, 1e-6))`, and the final result is checked against the declared constraint tolerance. Stage 2 minimizes exact one-way linear cost `0.5 * sum(abs(w-current)) * bps / 10000` plus `stability_penalty * sum((w-reference)^2)`. The strictly positive quadratic term selects a unique deterministic portfolio. Without current weights, use the benchmark as the first-period reference and do not apply turnover or trading cost.

Clarabel solves both lexicographic stages against one cached hard-constraint structure. Stage 1 updates the score parameter and maximizes utility; Stage 2 adds a parameterized utility floor and minimizes cost plus stability while warm-starting from Stage 1. The explicit SciPy fallback retains its certified HiGHS outer-inner primary gap and conservative secondary lower bound, but the primary cutting-plane search is limited by `max_cutting_planes`. Report a solve failure without switching backend. Write unsupported duals and KKT residuals as `null` with a reason.

## Signal calibration

`rank_score` inputs are direction-adjusted and MAD-winsorized before one of three monotonic rank transforms is applied. `uniform` preserves the v0.8 centered-percentile behavior. `normal_score` maps plotting positions `(rank-0.5)/N` through the standard-normal quantile. `power` applies `sign(p-0.5) * abs(p-0.5)^rank_power`. Optional standardization follows the transform. `annualized_alpha_scale` is a research assumption, not a fitted expected return.

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
- aggregate candidate-universe portfolio-weight range

A position that becomes non-tradable after market drift cannot be sold back to its configured stock bound. The optimizer therefore fixes it at the exact current weight, reports it in `frozen_bound_exceptions`, and applies maximum-weight and maximum-active-weight checks to the remaining controllable assets. Raw portfolio maxima remain in diagnostics, so the exception is never hidden. Once the asset becomes tradable, the ordinary bounds apply again.

Each exposure accepts either `target_active+tolerance` or `min_active+max_active`. `SIZE` is mandatory in schema versions 2 through 5. Legacy symmetric scalar or mapping limits remain accepted in schema version 1. Candidate weight bounds are absolute portfolio weights between zero and one; omitted lower or upper bounds resolve to zero or one.

The exact L1 turnover auxiliary bound reserves `constraints.constraint_tolerance` below the configured maximum before HiGHS solves the linear system; final diagnostics still use the declared maximum plus that same tolerance.

The optimization universe is the union of candidates, positive-weight benchmark constituents, and positive current holdings. Under `role_aware`, candidates without a signal are errors; a tradable current-only holding without a signal is constrained not to increase, a benchmark-only non-holding receives neutral alpha, and a non-tradable holding remains frozen. These roles are written to diagnostics rather than being treated as interchangeable neutral fills.

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
