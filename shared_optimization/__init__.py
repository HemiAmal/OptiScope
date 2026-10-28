"""
shared_optimization: Unified optimization algorithm & mathematical foundation library.
Contains all 23 required optimization methods, core matrix/vector utilities,
and visualization routines.
"""

# Expose subpackages cleanly
import shared_optimization.core as core
import shared_optimization.algorithms as algorithms
import shared_optimization.problems as problems
import shared_optimization.visualization as visualization

__all__ = ["core", "algorithms", "problems", "visualization"]
