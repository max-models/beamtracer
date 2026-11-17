"""BeamOptics under MPI: two ranks give the serial beam (skipped without mpirun and mpi4py).

With ``field_backend="petsc"`` the ranks share the field solve (skipped without petsc4py).
"""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

from beamtracer import BeamOptics
from beamtracer.tests.unit.test_beam_optics import _extraction
from beamtracer.tests.unit.test_compiled_tracer import _planar

pytestmark = pytest.mark.skipif(
    shutil.which("mpirun") is None or importlib.util.find_spec("mpi4py") is None,
    reason="needs mpirun and mpi4py",
)
PETSC = pytest.mark.skipif(
    importlib.util.find_spec("petsc4py") is None, reason="needs petsc4py"
)

SCRIPT = """
import json, sys
from maybempi import MPI
from beamtracer import BeamOptics
from beamtracer.tests.unit.test_beam_optics import _extraction
from beamtracer.tests.unit.test_compiled_tracer import _planar
case, backend, plane = sys.argv[1], sys.argv[2], float(sys.argv[3])
config = (_planar() if case == "planar" else _extraction(1.0)).replace(field_backend=backend)
model = BeamOptics(config).run()
m = model.beam_moments(plane)
if MPI.COMM_WORLD.Get_rank() == 0:
    print(json.dumps({"current": m.current, "rms_size": m.rms_size,
                      "stats": model.stats,
                      "lookup": -1 if model.tracer._lookup is None else int(model.tracer._lookup[1].size),
                      "potential": float(abs(model.potential).max())}))
"""

CASES = {"planar": (_planar, 0.115), "extraction": (lambda: _extraction(1.0), 7.5e-3)}


def _ranks(case: str, backend: str) -> dict:
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
    plane = str(CASES[case][1])
    done = subprocess.run(
        ["mpirun", "--oversubscribe", "-n", "2", sys.executable, "-c", SCRIPT, case, backend, plane],
        capture_output=True, text=True, env=env, timeout=900, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2000:]
    return json.loads(done.stdout.strip().splitlines()[-1])


@pytest.mark.parametrize(
    ("case", "backend", "tolerance"),
    [
        ("planar", "lu", 1e-12),
        pytest.param("planar", "petsc", 1e-6, marks=PETSC),
        pytest.param("extraction", "petsc", 1e-6, marks=PETSC),
    ],
)
def test_two_ranks_give_the_serial_beam(case, backend, tolerance) -> None:
    config, plane = CASES[case]
    serial = BeamOptics(config()).run()
    result = _ranks(case, backend)
    expected = serial.beam_moments(plane)
    assert result["current"] == pytest.approx(expected.current, rel=tolerance)
    assert result["rms_size"] == pytest.approx(expected.rms_size, rel=tolerance)
    assert result["potential"] == pytest.approx(abs(serial.potential).max(), rel=1e-6)
    if tolerance < 1e-9:
        assert result["stats"] == serial.stats
        lookup = serial.tracer._lookup
        assert result["lookup"] == (-1 if lookup is None else lookup[1].size)
