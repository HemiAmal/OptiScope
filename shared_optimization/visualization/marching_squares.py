"""
Marching Squares contour extraction algorithm for generating isocontours from scalar fields.
"""

from typing import Callable, List, Sequence, Tuple

LineSegment = Tuple[Tuple[float, float], Tuple[float, float]]


def _interp_point(
    p1: Tuple[float, float], p2: Tuple[float, float], z1: float, z2: float, c: float
) -> Tuple[float, float]:
    """Linearly interpolate point between p1 and p2 where value equals c."""
    dz = z2 - z1
    t = 0.5 if abs(dz) < 1e-15 else (c - z1) / dz
    t = max(0.0, min(1.0, t))
    return (p1[0] + t * (p2[0] - p1[0]), p1[1] + t * (p2[1] - p1[1]))


def extract_contour_segments(
    x_grid: Sequence[float],
    y_grid: Sequence[float],
    Z: Sequence[Sequence[float]],
    isovalue: float,
) -> List[LineSegment]:
    """
    Extract isocontour line segments for a given isovalue using Marching Squares.

    Returns:
        List[LineSegment]: List of ((x1, y1), (x2, y2)) segments.
    """
    nx = len(x_grid)
    ny = len(y_grid)
    segments: List[LineSegment] = []

    for i in range(nx - 1):
        x0, x1 = x_grid[i], x_grid[i + 1]
        for j in range(ny - 1):
            y0, y1 = y_grid[j], y_grid[j + 1]

            z0 = Z[i][j]          # bottom-left
            z1 = Z[i + 1][j]      # bottom-right
            z2 = Z[i + 1][j + 1]  # top-right
            z3 = Z[i][j + 1]      # top-left

            p0 = (x0, y0)
            p1 = (x1, y0)
            p2 = (x1, y1)
            p3 = (x0, y1)

            bit0 = 1 if z0 >= isovalue else 0
            bit1 = 1 if z1 >= isovalue else 0
            bit2 = 1 if z2 >= isovalue else 0
            bit3 = 1 if z3 >= isovalue else 0

            case_idx = bit0 + (bit1 << 1) + (bit2 << 2) + (bit3 << 3)

            if case_idx == 0 or case_idx == 15:
                continue

            # Compute edge points if needed
            # edge 0: bottom (p0 -> p1)
            # edge 1: right  (p1 -> p2)
            # edge 2: top    (p3 -> p2)
            # edge 3: left   (p0 -> p3)
            e0 = lambda: _interp_point(p0, p1, z0, z1, isovalue)
            e1 = lambda: _interp_point(p1, p2, z1, z2, isovalue)
            e2 = lambda: _interp_point(p3, p2, z3, z2, isovalue)
            e3 = lambda: _interp_point(p0, p3, z0, z3, isovalue)

            if case_idx in (1, 14):
                segments.append((e3(), e0()))
            elif case_idx in (2, 13):
                segments.append((e0(), e1()))
            elif case_idx in (3, 12):
                segments.append((e3(), e1()))
            elif case_idx in (4, 11):
                segments.append((e1(), e2()))
            elif case_idx == 5:
                # Ambiguous saddle: average corner values
                center = 0.25 * (z0 + z1 + z2 + z3)
                if center >= isovalue:
                    segments.append((e3(), e2()))
                    segments.append((e0(), e1()))
                else:
                    segments.append((e3(), e0()))
                    segments.append((e1(), e2()))
            elif case_idx in (6, 9):
                segments.append((e0(), e2()))
            elif case_idx in (7, 8):
                segments.append((e3(), e2()))
            elif case_idx == 10:
                center = 0.25 * (z0 + z1 + z2 + z3)
                if center >= isovalue:
                    segments.append((e3(), e0()))
                    segments.append((e1(), e2()))
                else:
                    segments.append((e3(), e2()))
                    segments.append((e0(), e1()))

    return segments
