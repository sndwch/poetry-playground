"""Seed management for deterministic and reproducible poetry generation.

Provides centralized seed management for all random number generators,
ensuring reproducible outputs across runs when a seed is specified.
"""

import random
import time
from typing import Optional

try:
    import numpy as np

    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    np = None

from .logger import logger


class SeedManager:
    """Manages random seeds for reproducible generation."""

    def __init__(self):
        """Initialize seed manager with no seed set."""
        self._current_seed: Optional[int] = None
        self._seed_history = []

    @property
    def current_seed(self) -> Optional[int]:
        """Get the current seed value."""
        return self._current_seed

    def generate_seed(self) -> int:
        """Generate a new random seed based on current time.

        Returns:
            Integer seed value (32-bit unsigned)
        """
        # Use time-based seed with microsecond precision
        seed = int(time.time() * 1000000) % (2**32)
        return seed

    def set_seed(self, seed: Optional[int] = None) -> int:
        """Set the random seed for all RNG sources.

        Args:
            seed: Seed value to use. If None, generates a new random seed.

        Returns:
            The seed value that was set
        """
        if seed is None:
            seed = self.generate_seed()
            logger.info(f"Generated random seed: {seed}")
        else:
            logger.info(f"Using specified seed: {seed}")

        self._current_seed = seed
        self._seed_history.append(seed)

        # Set Python's random module seed
        random.seed(seed)

        # Set numpy's random seed if available
        if NUMPY_AVAILABLE:
            np.random.seed(seed)

        return seed

    def reset(self) -> None:
        """Reset to unseeded random state."""
        self._current_seed = None
        random.seed(None)
        if NUMPY_AVAILABLE:
            np.random.seed(None)
        logger.debug("Reset to unseeded random state")

    def get_state_summary(self) -> dict:
        """Get current seed state information.

        Returns:
            Dictionary with seed state information
        """
        return {
            "current_seed": self._current_seed,
            "seed_history": self._seed_history.copy(),
            "numpy_available": NUMPY_AVAILABLE,
        }

    def format_seed_output(self, prefix: str = "🎲") -> str:
        """Format seed information for display.

        Args:
            prefix: Emoji or prefix to use

        Returns:
            Formatted string with seed information
        """
        if self._current_seed is None:
            return f"{prefix} Random seed: Not set (non-deterministic)"
        return f"{prefix} Random seed: {self._current_seed} (use --seed {self._current_seed} to reproduce)"


# Global seed manager instance
seed_manager = SeedManager()


def set_global_seed(seed: Optional[int] = None) -> int:
    """Set the global random seed.

    Args:
        seed: Seed value to use. If None, generates a new random seed.

    Returns:
        The seed value that was set

    Example:
        >>> seed = set_global_seed(42)
        >>> print(f"Using seed: {seed}")
        Using seed: 42
    """
    return seed_manager.set_seed(seed)


def get_current_seed() -> Optional[int]:
    """Get the current global seed value.

    Returns:
        Current seed or None if not set
    """
    return seed_manager.current_seed


def generate_new_seed() -> int:
    """Generate a new random seed without setting it.

    Returns:
        New seed value
    """
    return seed_manager.generate_seed()


def format_seed_message() -> str:
    """Format current seed information for display.

    Returns:
        Formatted seed message
    """
    return seed_manager.format_seed_output()
