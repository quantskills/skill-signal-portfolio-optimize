class PortfolioOptimizeError(Exception):
    """Base class for expected, user-actionable runtime errors."""


class ConfigError(PortfolioOptimizeError):
    """Raised when configuration is invalid."""


class InputDataError(PortfolioOptimizeError):
    """Raised when an input file violates its contract."""


class OptimizationError(PortfolioOptimizeError):
    """Raised when the optimization is infeasible or fails validation."""


class RiskModelError(PortfolioOptimizeError):
    """Raised when structural risk estimation cannot produce a valid model."""
