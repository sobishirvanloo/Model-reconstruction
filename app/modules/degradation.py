"""Controlled degradation of point clouds, for robustness experiments.

Every perturbation is reproducible (seeded) and sized relative to the
model (% of the bounding-box diagonal), so the same settings mean the same
thing for any model:

* position noise  - Gaussian noise on x y z, sigma in % of the diagonal
* normal noise    - each normal tilted by a random angle ~ N(0, sigma deg)
* holes           - points inside random spheres removed
* non-uniform     - keep-probability falls linearly from 100% to
                    ``min_keep_pct`` along one axis

``Degradation.apply`` runs them in a fixed order: holes -> non-uniform
sampling -> position noise -> normal noise.  Pure numpy (no Qt).
"""

from dataclasses import dataclass, replace

import numpy as np

from modules.geometry_utils import bbox_diagonal, normalize

AXES = "XYZ"


# =========================
# Single perturbations
# =========================
def position_noise(points, normals, sigma_pct, rng):
    sigma = sigma_pct / 100.0 * bbox_diagonal(points)
    return points + rng.normal(0.0, sigma, points.shape), normals


def normal_noise(points, normals, sigma_deg, rng):
    """Tilt every normal by an angle ~ N(0, sigma) in a random direction."""
    if normals is None:
        return points, normals
    random = rng.normal(size=normals.shape)
    tangent = random - np.einsum("ij,ij->i", random, normals)[:, None] * normals
    tangent = normalize(tangent)
    theta = np.radians(rng.normal(0.0, sigma_deg, len(normals)))[:, None]
    return points, normalize(np.cos(theta) * normals + np.sin(theta) * tangent)


def cut_holes(points, normals, count, radius_pct, rng):
    """Remove the points inside ``count`` spheres centred on random points."""
    if count <= 0 or len(points) == 0:
        return points, normals
    radius = radius_pct / 100.0 * bbox_diagonal(points)
    centres = points[rng.choice(len(points), min(count, len(points)), replace=False)]
    keep = np.ones(len(points), dtype=bool)
    for centre in centres:
        keep &= np.linalg.norm(points - centre, axis=1) > radius
    return points[keep], (normals[keep] if normals is not None else None)


def nonuniform_sampling(points, normals, min_keep_pct, axis, rng):
    """Keep-probability 100% at the low end of ``axis``, ``min_keep_pct``
    at the high end (a density gradient)."""
    if min_keep_pct >= 100 or len(points) == 0:
        return points, normals
    coord = points[:, axis]
    span = coord.max() - coord.min()
    t = (coord - coord.min()) / span if span > 0 else np.zeros_like(coord)
    keep_probability = 1.0 - (1.0 - min_keep_pct / 100.0) * t
    keep = rng.random(len(points)) < keep_probability
    return points[keep], (normals[keep] if normals is not None else None)


# =========================
# Combined settings
# =========================
@dataclass(frozen=True)
class Degradation:
    noise_pct: float = 0.0        # position noise sigma, % of diagonal
    normal_deg: float = 0.0       # normal noise sigma, degrees
    holes: int = 0                # number of spherical holes
    hole_radius_pct: float = 5.0  # hole radius, % of diagonal
    min_keep_pct: float = 100.0   # non-uniform sampling (100 = off)
    axis: int = 1                 # 0/1/2 = X/Y/Z for non-uniform sampling
    seed: int = 0

    @property
    def is_identity(self):
        return (self.noise_pct <= 0 and self.normal_deg <= 0
                and self.holes <= 0 and self.min_keep_pct >= 100)

    def apply(self, points, normals):
        """Return degraded copies of ``points`` / ``normals``."""
        points = np.asarray(points, dtype=float)
        normals = None if normals is None else np.asarray(normals, dtype=float)
        if self.is_identity:
            return points.copy(), None if normals is None else normals.copy()
        rng = np.random.default_rng(self.seed)
        points, normals = cut_holes(
            points, normals, self.holes, self.hole_radius_pct, rng)
        points, normals = nonuniform_sampling(
            points, normals, self.min_keep_pct, self.axis, rng)
        if self.noise_pct > 0:
            points, normals = position_noise(points, normals, self.noise_pct, rng)
        if self.normal_deg > 0:
            points, normals = normal_noise(points, normals, self.normal_deg, rng)
        return points, normals

    def describe(self):
        if self.is_identity:
            return "clean"
        parts = []
        if self.noise_pct > 0:
            parts.append(f"noise σ={self.noise_pct:g}%")
        if self.normal_deg > 0:
            parts.append(f"normals σ={self.normal_deg:g}°")
        if self.holes > 0:
            parts.append(f"{self.holes} hole{'s' * (self.holes > 1)} "
                         f"r={self.hole_radius_pct:g}%")
        if self.min_keep_pct < 100:
            parts.append(f"keep→{self.min_keep_pct:g}% along {AXES[self.axis]}")
        return " · ".join(parts)


# =========================
# Sweeps (one perturbation, several levels)
# =========================
# key -> (label, Degradation field, default levels)
SWEEP_KINDS = {
    "noise": ("Position noise σ (% of diagonal)", "noise_pct",
              (0.0, 0.1, 0.25, 0.5, 1.0)),
    "normals": ("Normal noise σ (degrees)", "normal_deg",
                (0.0, 5.0, 15.0, 30.0, 60.0)),
    "holes": ("Holes (count, radius 5%)", "holes",
              (0, 1, 3, 6, 10)),
    "nonuniform": ("Non-uniform sampling: min keep (%)", "min_keep_pct",
                   (100.0, 50.0, 25.0, 10.0, 5.0)),
}


def sweep_settings(kind, level, seed=0, base=None):
    """The ``Degradation`` for one level of a sweep."""
    _, field_name, _ = SWEEP_KINDS[kind]
    base = base or Degradation()
    value = int(round(level)) if field_name == "holes" else float(level)
    return replace(base, seed=seed, **{field_name: value})


def parse_levels(text, kind):
    """'0, 0.5, 1' -> sorted unique levels (validated for the kind)."""
    _, field_name, _ = SWEEP_KINDS[kind]
    try:
        levels = [float(part) for part in text.replace(";", ",").split(",")
                  if part.strip()]
    except ValueError:
        raise ValueError("levels must be numbers separated by commas")
    if not levels:
        raise ValueError("enter at least one level")
    if any(level < 0 for level in levels):
        raise ValueError("levels cannot be negative")
    if field_name == "min_keep_pct" and any(level > 100 for level in levels):
        raise ValueError("min keep must be between 0 and 100")
    # "min keep" degrades as it DEcreases: order levels clean -> worst
    return sorted(set(levels), reverse=(field_name == "min_keep_pct"))
