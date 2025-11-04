"""Patch for pronouncing library to avoid pkg_resources deprecation warning."""

import sys
import warnings

# Suppress the specific deprecation warning from pronouncing
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    message=".*pkg_resources is deprecated.*",
    module="pronouncing"
)

# Alternative: monkey-patch pkg_resources before pronouncing imports it
try:
    import importlib.resources as resources

    def patch_pronouncing():
        """Monkey-patch pronouncing to use importlib.resources instead of pkg_resources."""
        import pronouncing

        # If pronouncing is using pkg_resources, we'd patch it here
        # But for now, we'll just suppress the warning
        pass

except ImportError:
    # Fall back to pkg_resources if importlib.resources not available
    pass