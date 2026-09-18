"""Model builders. Every builder returns an unfitted sklearn-compatible
estimator whose last step is named ``"classifier"``."""

from . import boosting, forest, logistic  # noqa: F401
