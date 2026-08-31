"""Non-self-intersecting random waypoint plans for a moving target.

A "sensible" random route for a target that flies through a sequence of points
should not cross itself. Self-intersection is a 2-D notion (a 1-D curve in 3-D
space generically never meets itself), so we enforce it on the top-down
xy-projection — the ground track must be a *simple* polyline.

This module provides:
  * ``generate_points`` — rejection-sampled points in the dome with min spacing.
  * ``order_simple_2opt`` — order points into a non-crossing (simple) open path
    via 2-opt uncrossing. Each uncross strictly shortens total length, so it
    terminates at a simple path that also looks natural (taut, short-tour-like).
  * ``order_angular`` — trivial star-shaped simple ordering (fan/spiral look).
  * ``path_is_simple`` / ``self_intersections`` — verify a polyline/ground track
    has no crossing non-adjacent segments (also usable on a *flown* trajectory).
  * ``WaypointPlan`` — an ordered set of 3-D waypoints a target advances through.

The theorem behind 2-opt: the Euclidean-shortest tour through a point set is
never self-intersecting, and every 2-opt local minimum is simple — so removing
crossings by sub-path reversal is guaranteed to reach a simple path.
"""

from __future__ import annotations

import numpy as np


# -- Geometry helpers ---------------------------------------------------------

def _ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Signed area sign of triangle abc (>0 CCW, <0 CW, 0 collinear)."""
    return (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])


def _segments_properly_cross(p1, p2, p3, p4) -> bool:
    """True if open segments p1p2 and p3p4 cross at an interior point.

    Uses the standard orientation test. Shared endpoints (adjacent segments in a
    path) are NOT counted as crossings; only proper interior intersections are.
    """
    d1 = _ccw(p3, p4, p1)
    d2 = _ccw(p3, p4, p2)
    d3 = _ccw(p1, p2, p3)
    d4 = _ccw(p1, p2, p4)
    return ((d1 > 0) != (d2 > 0)) and ((d3 > 0) != (d4 > 0))


def self_intersections(path_xy: np.ndarray) -> list[tuple[int, int]]:
    """Return index pairs (i, j) of non-adjacent segments that cross.

    Args:
        path_xy: (K, 2) polyline vertices (or the xy of a flown trajectory).

    Returns:
        List of crossing segment-index pairs; empty iff the path is simple.
    """
    pts = np.asarray(path_xy, dtype=float)[:, :2]
    n = len(pts) - 1  # number of segments
    hits = []
    for i in range(n):
        # Non-adjacent segments only: segment i shares a vertex with i-1 and i+1.
        # For an OPEN path, segment 0 and segment n-1 share no vertex, so the
        # (0, n-1) pair IS checked here (no special-casing).
        for j in range(i + 2, n):
            if _segments_properly_cross(pts[i], pts[i + 1], pts[j], pts[j + 1]):
                hits.append((i, j))
    return hits


def path_is_simple(path_xy: np.ndarray) -> bool:
    """True iff the polyline / ground track has no self-intersections."""
    return len(self_intersections(path_xy)) == 0


# -- Point sampling -----------------------------------------------------------

def generate_points(
    n: int,
    dome_size: float,
    rng: np.random.Generator,
    min_separation: float = 3.0,
    altitude_range: tuple[float, float] = (1.5, 6.0),
    max_attempts: int = 1000,
) -> np.ndarray:
    """Rejection-sample ``n`` 3-D points in the dome with a minimum spacing.

    xy is sampled in the disk of radius ``dome_size/2``; z in ``altitude_range``.
    """
    r_max = dome_size / 2.0
    pts: list[np.ndarray] = []
    for _ in range(n):
        for _ in range(max_attempts):
            ang = rng.uniform(0.0, 2.0 * np.pi)
            rad = r_max * np.sqrt(rng.uniform(0.0, 1.0))  # uniform in disk
            cand = np.array(
                [rad * np.cos(ang), rad * np.sin(ang), rng.uniform(*altitude_range)],
                dtype=np.float32,
            )
            if all(np.linalg.norm(cand[:2] - p[:2]) >= min_separation for p in pts):
                pts.append(cand)
                break
        else:
            pts.append(cand)  # accept last candidate if crowded
    return np.array(pts, dtype=np.float32)


# -- Orderings ----------------------------------------------------------------

def order_angular(points: np.ndarray) -> np.ndarray:
    """Star-shaped simple ordering: sort by polar angle about the centroid."""
    xy = np.asarray(points)[:, :2]
    c = xy.mean(axis=0)
    ang = np.arctan2(xy[:, 1] - c[1], xy[:, 0] - c[0])
    return np.argsort(ang)


def _path_length(xy: np.ndarray, order: list[int]) -> float:
    return float(sum(np.linalg.norm(xy[order[k + 1]] - xy[order[k]]) for k in range(len(order) - 1)))


def order_simple_2opt(
    points: np.ndarray,
    rng: np.random.Generator | None = None,
    max_passes: int = 1000,
) -> np.ndarray:
    """Order points into a non-self-intersecting open path via 2-opt uncrossing.

    Starts from a random order and repeatedly reverses the sub-path between any
    two crossing edges. Terminates at a simple path (guaranteed: each reversal
    strictly shortens total length). Returns the vertex order (indices).
    """
    xy = np.asarray(points, dtype=float)[:, :2]
    n = len(xy)
    order = list(range(n))
    if rng is not None:
        rng.shuffle(order)

    for _ in range(max_passes):
        improved = False
        # segments of the open path: (order[k], order[k+1]) for k in 0..n-2
        for i in range(n - 1):
            for j in range(i + 2, n - 1):
                a, b = xy[order[i]], xy[order[i + 1]]
                c, d = xy[order[j]], xy[order[j + 1]]
                if _segments_properly_cross(a, b, c, d):
                    order[i + 1 : j + 1] = order[i + 1 : j + 1][::-1]  # 2-opt reverse
                    improved = True
                    break
            if improved:
                break
        if not improved:
            break
    return np.array(order)


# -- Plan object --------------------------------------------------------------

class WaypointPlan:
    """An ordered sequence of 3-D waypoints a target advances through.

    The target moves toward ``current()``; when it gets within ``reach`` the plan
    advances to the next waypoint. When the last is reached the plan can loop or
    regenerate a fresh simple route.

    Attributes:
        waypoints: (K, 3) ordered waypoints (ground track is non-self-intersecting).
        reach: distance at which a waypoint counts as reached.
    """

    def __init__(self, waypoints: np.ndarray, reach: float = 1.5, loop: bool = True) -> None:
        self.waypoints = np.asarray(waypoints, dtype=np.float32)
        self.reach = reach
        self.loop = loop
        self._i = 0

    @classmethod
    def sample(
        cls,
        n: int,
        dome_size: float,
        rng: np.random.Generator,
        min_separation: float = 3.0,
        altitude_range: tuple[float, float] = (1.5, 6.0),
        method: str = "2opt",
        reach: float = 1.5,
        loop: bool = True,
        spawn_radius: float = 0.0,
    ) -> "WaypointPlan":
        """Sample points and order them into a simple route (default 2-opt).

        When ``spawn_radius`` > 0 the FIRST waypoint (the target's spawn/start) is
        sampled within that small disk while the remaining waypoints spread over the
        full ``dome_size`` disk. This decouples where the chase starts from how far
        the target roams: a close start, then a route that leads far away.
        """
        def _order(p: np.ndarray) -> np.ndarray:
            if method == "angular":
                return order_angular(p)
            if method == "2opt":
                return order_simple_2opt(p, rng)
            raise ValueError(f"unknown ordering method {method!r}")

        if spawn_radius and spawn_radius > 0.0 and n > 1:
            first = generate_points(1, 2.0 * spawn_radius, rng, min_separation, altitude_range)
            rest = generate_points(n - 1, dome_size, rng, min_separation, altitude_range)
            pts = np.vstack([first, rest[_order(rest)]]).astype(np.float32)
            return cls(pts, reach=reach, loop=loop)

        pts = generate_points(n, dome_size, rng, min_separation, altitude_range)
        return cls(pts[_order(pts)], reach=reach, loop=loop)

    def current(self) -> np.ndarray:
        """Current target waypoint (3,)."""
        return self.waypoints[self._i].copy()

    def advance_if_reached(self, position: np.ndarray) -> bool:
        """Advance to the next waypoint if ``position`` is within ``reach``.

        Returns True if the plan advanced (a waypoint was reached).
        """
        if np.linalg.norm(np.asarray(position)[:2] - self.waypoints[self._i][:2]) < self.reach:
            if self._i < len(self.waypoints) - 1:
                self._i += 1
            elif self.loop:
                self._i = 0
            return True
        return False

    @property
    def ground_track_simple(self) -> bool:
        """Whether the planned polyline's ground track is non-self-intersecting."""
        return path_is_simple(self.waypoints)
