"""Beam quantities at a diagnostic plane: current, rms emittance and Twiss parameters."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import constants

from beamtracer.solver.tracer import PlaneHits


@dataclass(frozen=True)
class BeamMoments:
    """Current-weighted second moments of a beam in one transverse plane.

    ``emittance`` is the rms emittance ``sqrt(<x^2><x'^2> - <x x'>^2)`` [m rad],
    ``normalized_emittance`` that times ``beta gamma`` (from the mean axial velocity), and
    ``alpha``, ``beta`` [m], ``gamma`` [1/m] the Twiss parameters of the rms ellipse.
    """

    current: float
    rms_size: float
    rms_angle: float
    correlation: float
    emittance: float
    normalized_emittance: float
    alpha: float
    beta: float
    gamma: float
    mean_energy: float
    count: int

    def as_dict(self) -> dict[str, float]:
        """Return the moments as a dictionary."""
        return dict(self.__dict__)


def plane_moments(
    hits: PlaneHits,
    *,
    cylindrical: bool,
    mass: np.ndarray | float,
    species: int | None = None,
    forward_only: bool = True,
) -> BeamMoments:
    """Return the :class:`BeamMoments` of the trajectories crossing a plane.

    In cylindrical geometry each trajectory is a ring, and the moments are those of its
    projection on ``x``-``x'`` (the ring rotated about the axis, averaged over all angles): ``<x^2> = <r^2>/2``, ``<x'^2> = <r'^2 + a'^2>/2`` and ``<x x'> = <r r'>/2``,
    with ``r' = v_r / v_x`` and ``a' = v_theta / v_x``. In planar geometry the moments are
    those of ``y`` and ``y' = v_y / v_x`` about their means. ``mass`` [kg] is one value per
    species (indexed by ``hits.species``) or a single value; only the given ``species`` and,
    with ``forward_only``, crossings in the ``+x`` direction count.
    """
    keep = np.ones(hits.current.size, dtype=bool)
    if species is not None:
        keep &= hits.species == species
    if forward_only:
        keep &= hits.direction > 0
    w = np.abs(hits.current[keep])
    point = hits.point[:, keep]
    vel = hits.velocity[:, keep]
    masses = np.asarray(mass, dtype=float)
    m = masses[hits.species[keep]] if masses.ndim else np.full(w.size, float(masses))
    total = float(w.sum())
    if w.size == 0 or total == 0.0:
        nan = float("nan")
        return BeamMoments(0.0, nan, nan, nan, nan, nan, nan, nan, nan, nan, 0)
    vx = vel[0]
    if cylindrical:
        r = np.hypot(point[1], point[2])
        with np.errstate(invalid="ignore", divide="ignore"):
            cy = np.where(r > 0, point[1] / r, 1.0)
            cz = np.where(r > 0, point[2] / r, 0.0)
        vr = vel[1] * cy + vel[2] * cz
        vt = vel[2] * cy - vel[1] * cz
        rp = vr / vx
        ap = vt / vx
        xx = np.sum(w * r * r) / (2.0 * total)
        pp = np.sum(w * (rp * rp + ap * ap)) / (2.0 * total)
        xp = np.sum(w * r * rp) / (2.0 * total)
    else:
        y = point[1]
        yp = vel[1] / vx
        my = np.sum(w * y) / total
        myp = np.sum(w * yp) / total
        xx = np.sum(w * (y - my) ** 2) / total
        pp = np.sum(w * (yp - myp) ** 2) / total
        xp = np.sum(w * (y - my) * (yp - myp)) / total
    emittance = float(np.sqrt(max(xx * pp - xp * xp, 0.0)))
    beta_gamma = float(np.sum(w * vx) / total) / constants.c
    energy = 0.5 * m * np.sum(vel * vel, axis=0) / constants.e
    alpha = -xp / emittance if emittance > 0 else float("nan")
    beta = xx / emittance if emittance > 0 else float("nan")
    gamma = pp / emittance if emittance > 0 else float("nan")
    return BeamMoments(
        current=total,
        rms_size=float(np.sqrt(xx)),
        rms_angle=float(np.sqrt(pp)),
        correlation=float(xp),
        emittance=emittance,
        normalized_emittance=emittance * beta_gamma,
        alpha=float(alpha),
        beta=float(beta),
        gamma=float(gamma),
        mean_energy=float(np.sum(w * energy) / total),
        count=int(w.size),
    )


def mismatch(alpha: float, beta: float, alpha0: float, beta0: float) -> float:
    """Return the mismatch parameter ``M`` of a beam ``(alpha, beta)``.

    Against the design ``(alpha0, beta0)``, as defined in Midttun et al. (IPAC 2017)
    eq. (1).
    """
    gamma = (1.0 + alpha**2) / beta
    gamma0 = (1.0 + alpha0**2) / beta0
    delta = (alpha - alpha0) ** 2 - (gamma - gamma0) * (beta - beta0)
    return float(np.sqrt(1.0 + (delta + np.sqrt(delta * (delta + 4.0))) / 2.0) - 1.0)
