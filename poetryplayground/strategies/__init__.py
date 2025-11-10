"""Strategy implementations for the Strategy Engine.

This package contains all concrete strategy implementations. Each strategy
is a "creative recipe" that orchestrates multiple generators to fulfill
a high-level creative brief.

Available strategies:
- BridgeTwoConceptsStrategy: Bridge between two words using semantic paths,
  conceptual clouds, metaphors, and line seeds
"""

from poetryplayground.strategies.bridge_two_concepts import BridgeTwoConceptsStrategy

__all__ = ["BridgeTwoConceptsStrategy"]
