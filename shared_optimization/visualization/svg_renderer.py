"""
Pure-Python SVG vector graphics renderer for contours, trajectories, and curves.
"""

from typing import List, Optional, Tuple


class SVGRenderer:
    """
    Manages SVG drawing primitives and coordinate mapping.

    Coordinate Mapping:
        Maps mathematical coordinates (x, y) in [x_min, x_max] x [y_min, y_max]
        to screen coordinates (sx, sy) in [margin_left, width - margin_right] x [margin_top, height - margin_bottom].
        Note: SVG y-axis is oriented downwards, so screen_y is inverted.
    """
    def __init__(
        self,
        width: int = 800,
        height: int = 600,
        x_min: float = -2.0,
        x_max: float = 2.0,
        y_min: float = -2.0,
        y_max: float = 2.0,
        title: str = "Optimization Landscape Plot",
    ) -> None:
        self.width = width
        self.height = height
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.title = title

        self.margin_left = 70
        self.margin_right = 40
        self.margin_top = 60
        self.margin_bottom = 60

        self.plot_width = width - self.margin_left - self.margin_right
        self.plot_height = height - self.margin_top - self.margin_bottom

        self.elements: List[str] = []

    def math_to_screen(self, x: float, y: float) -> Tuple[float, float]:
        """Convert mathematical coordinate (x, y) to screen pixels (sx, sy)."""
        x_span = max(self.x_max - self.x_min, 1e-12)
        y_span = max(self.y_max - self.y_min, 1e-12)

        sx = self.margin_left + ((x - self.x_min) / x_span) * self.plot_width
        sy = (self.height - self.margin_bottom) - ((y - self.y_min) / y_span) * self.plot_height
        return sx, sy

    def draw_background_and_axes(self, x_label: str = "x1", y_label: str = "x2") -> None:
        """Render background canvas, plot border, tick marks, and axes labels."""
        # Canvas background
        self.elements.append(
            f'<rect width="{self.width}" height="{self.height}" fill="#181a1f" />'
        )
        # Plot area background
        self.elements.append(
            f'<rect x="{self.margin_left}" y="{self.margin_top}" '
            f'width="{self.plot_width}" height="{self.plot_height}" '
            f'fill="#21252b" stroke="#3e4451" stroke-width="1.5" />'
        )

        # Title
        self.elements.append(
            f'<text x="{self.width / 2}" y="{self.margin_top - 20}" '
            f'fill="#abb2bf" font-family="sans-serif" font-size="16" '
            f'font-weight="bold" text-anchor="middle">{self.title}</text>'
        )

        # Ticks on X-axis (5 ticks)
        for i in range(5):
            t = i / 4.0
            val_x = self.x_min + t * (self.x_max - self.x_min)
            sx, sy = self.math_to_screen(val_x, self.y_min)
            self.elements.append(
                f'<line x1="{sx}" y1="{sy}" x2="{sx}" y2="{sy + 6}" stroke="#5c6370" stroke-width="1" />'
            )
            self.elements.append(
                f'<text x="{sx}" y="{sy + 20}" fill="#828997" font-family="sans-serif" '
                f'font-size="11" text-anchor="middle">{val_x:.1f}</text>'
            )

        # Ticks on Y-axis (5 ticks)
        for i in range(5):
            t = i / 4.0
            val_y = self.y_min + t * (self.y_max - self.y_min)
            sx, sy = self.math_to_screen(self.x_min, val_y)
            self.elements.append(
                f'<line x1="{sx - 6}" y1="{sy}" x2="{sx}" y2="{sy}" stroke="#5c6370" stroke-width="1" />'
            )
            self.elements.append(
                f'<text x="{sx - 10}" y="{sy + 4}" fill="#828997" font-family="sans-serif" '
                f'font-size="11" text-anchor="end">{val_y:.1f}</text>'
            )

        # Axis Labels
        self.elements.append(
            f'<text x="{self.margin_left + self.plot_width / 2}" y="{self.height - 15}" '
            f'fill="#abb2bf" font-family="sans-serif" font-size="13" text-anchor="middle">{x_label}</text>'
        )
        self.elements.append(
            f'<text x="20" y="{self.margin_top + self.plot_height / 2}" '
            f'fill="#abb2bf" font-family="sans-serif" font-size="13" text-anchor="middle" '
            f'transform="rotate(-90 20 {self.margin_top + self.plot_height / 2})">{y_label}</text>'
        )

    def draw_line_segment(
        self,
        x1: float,
        y1: float,
        x2: float,
        y2: float,
        stroke: str = "#61afef",
        stroke_width: float = 1.0,
        opacity: float = 0.8,
    ) -> None:
        """Draw a single line segment in mathematical coordinates."""
        sx1, sy1 = self.math_to_screen(x1, y1)
        sx2, sy2 = self.math_to_screen(x2, y2)
        self.elements.append(
            f'<line x1="{sx1:.1f}" y1="{sy1:.1f}" x2="{sx2:.1f}" y2="{sy2:.1f}" '
            f'stroke="{stroke}" stroke-width="{stroke_width}" stroke-opacity="{opacity}" />'
        )

    def draw_polyline(
        self,
        points: List[Tuple[float, float]],
        stroke: str = "#e5c07b",
        stroke_width: float = 2.5,
    ) -> None:
        """Draw connected line segments representing an optimization path."""
        if len(points) < 2:
            return
        pts_str = " ".join(
            f"{self.math_to_screen(x, y)[0]:.1f},{self.math_to_screen(x, y)[1]:.1f}"
            for x, y in points
        )
        self.elements.append(
            f'<polyline points="{pts_str}" fill="none" stroke="{stroke}" '
            f'stroke-width="{stroke_width}" stroke-linecap="round" stroke-linejoin="round" />'
        )

    def draw_marker(
        self,
        x: float,
        y: float,
        color: str = "#98c379",
        radius: float = 5.0,
        label: Optional[str] = None,
    ) -> None:
        """Draw point marker with optional text annotation."""
        sx, sy = self.math_to_screen(x, y)
        self.elements.append(
            f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="{radius}" fill="{color}" stroke="#181a1f" stroke-width="1.5" />'
        )
        if label:
            self.elements.append(
                f'<text x="{sx + radius + 4:.1f}" y="{sy - 4:.1f}" fill="{color}" '
                f'font-family="sans-serif" font-size="11" font-weight="bold">{label}</text>'
            )

    def render_to_string(self) -> str:
        """Assemble all elements into a complete SVG document."""
        svg_content = [
            f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {self.width} {self.height}" '
            f'width="{self.width}" height="{self.height}">',
            *self.elements,
            "</svg>",
        ]
        return "\n".join(svg_content)

    def save(self, filepath: str) -> None:
        """Write SVG document to file."""
        import os
        os.makedirs(os.path.dirname(os.path.abspath(filepath)), exist_ok=True)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(self.render_to_string())
