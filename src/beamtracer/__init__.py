"""Steady ion beams in a self-consistent field: extraction, acceleration, charge changes.

:class:`BeamOptics` traces a beam self-consistently (a Vlasov iteration of field
solve, trajectory tracing and space-charge deposition) in planar or cylindrical
geometry, from the plasma it is extracted from through electrodes and solenoids;
:class:`BeamOpticsConfig` describes the run. :mod:`beamtracer.neutralizer` couples the
traced beam to the plasma of a neutralizer, and :mod:`beamtracer.solver` holds the
numerics.
"""

from importlib.metadata import PackageNotFoundError, version

import scope_profiler as sp

from beamtracer._profiling import PROFILE_TAGS
from beamtracer.config import (
    AxialField,
    BeamOpticsConfig,
    BeamSpecies,
    ChargeChangingParams,
    ExtractionPlasma,
    NegativeExtractionPlasma,
    ParticleBeam,
    Solenoid,
    SpaceChargeCompensation,
)
from beamtracer.model import BeamOptics
from beamtracer.sections import (
    FieldBCs,
    FieldBoundaryCondition,
    GridParams,
    ObjectParams,
    ShapeParams,
    SymbolicSpaceFunction,
)

try:
    __version__ = version("beamtracer")
except PackageNotFoundError:  # not installed, e.g. run from a source tree
    __version__ = "0.0.0+unknown"

# The decorated functions pick the tag up, so registering after the imports is in time.
sp.register_opt_in_tag(*PROFILE_TAGS)

__all__ = [
    "AxialField",
    "BeamOptics",
    "BeamOpticsConfig",
    "BeamSpecies",
    "ChargeChangingParams",
    "ExtractionPlasma",
    "FieldBCs",
    "FieldBoundaryCondition",
    "GridParams",
    "NegativeExtractionPlasma",
    "ObjectParams",
    "ParticleBeam",
    "ShapeParams",
    "Solenoid",
    "SpaceChargeCompensation",
    "SymbolicSpaceFunction",
]
