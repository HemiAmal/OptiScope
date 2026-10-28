"""Visualization and plotting module."""

from shared_optimization.visualization.svg_renderer import SVGRenderer
from shared_optimization.visualization.marching_squares import extract_contour_segments
from shared_optimization.visualization.contour import plot_contour_with_trajectory
from shared_optimization.visualization.model_plots import (
    plot_linear_regression_fit,
    plot_logistic_regression_boundary,
    plot_convergence_curves,
)

__all__ = [
    "SVGRenderer",
    "extract_contour_segments",
    "plot_contour_with_trajectory",
    "plot_linear_regression_fit",
    "plot_logistic_regression_boundary",
    "plot_convergence_curves",
]
