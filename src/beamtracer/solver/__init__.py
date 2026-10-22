"""Steady-state ion optics in planar or cylindrical geometry.

- :mod:`.mesh`: the node mesh, electrodes and wall distances,
- :mod:`.poisson`: Poisson's equation with a Boltzmann plasma species (electrons for
  positive-ion extraction, positive ions for negative-ion extraction),
- :mod:`.fields`: the staggered electric field and axisymmetric magnetic fields,
- :mod:`.tracer`: beams, trajectory tracing, space charge and diagnostic planes,
- :mod:`.diagnostics`: emittance and Twiss parameters,
- :mod:`.vlasov`: the Vlasov iteration that ties them together.
"""

from beamtracer.solver.diagnostics import BeamMoments, mismatch, plane_moments
from beamtracer.solver.fields import (
    AxisymmetricMagneticField,
    StaggeredElectricField,
    solenoid_on_axis,
    surface_field,
)
from beamtracer.solver.mesh import Electrode, FaceCondition, IonOpticsMesh
from beamtracer.solver.poisson import (
    BoltzmannElectrons,
    BoltzmannIons,
    IonOpticsPoisson,
)
from beamtracer.solver.tracer import (
    STATUS_NAMES,
    BeamParticles,
    BeamTracer,
    PlaneHits,
    TraceResult,
    beam_with_energy,
)
from beamtracer.solver.vlasov import (
    Compensation,
    NegativeIonPlasmaModel,
    PlasmaModel,
    VlasovState,
    apply_compensation,
    vlasov_iterate,
)

__all__ = [
    "STATUS_NAMES",
    "AxisymmetricMagneticField",
    "BeamMoments",
    "BeamParticles",
    "BeamTracer",
    "BoltzmannElectrons",
    "BoltzmannIons",
    "Compensation",
    "Electrode",
    "FaceCondition",
    "IonOpticsMesh",
    "IonOpticsPoisson",
    "NegativeIonPlasmaModel",
    "PlaneHits",
    "PlasmaModel",
    "StaggeredElectricField",
    "TraceResult",
    "VlasovState",
    "apply_compensation",
    "beam_with_energy",
    "mismatch",
    "plane_moments",
    "solenoid_on_axis",
    "surface_field",
    "vlasov_iterate",
]
