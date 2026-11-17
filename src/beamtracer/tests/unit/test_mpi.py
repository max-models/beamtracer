"""BeamOptics under MPI: two ranks give the serial beam (skipped without mpirun and mpi4py)."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys

import pytest

from beamtracer import BeamOptics
from beamtracer.tests.unit.test_compiled_tracer import _planar

pytestmark = pytest.mark.skipif(
    shutil.which("mpirun") is None or importlib.util.find_spec("mpi4py") is None,
    reason="needs mpirun and mpi4py",
)

SCRIPT = """
import json
from maybempi import MPI
from beamtracer import BeamOptics
from beamtracer.tests.unit.test_compiled_tracer import _planar
model = BeamOptics(_planar()).run()
m = model.beam_moments(0.115)
if MPI.COMM_WORLD.Get_rank() == 0:
    print(json.dumps({"current": m.current, "rms_size": m.rms_size,
                      "stats": model.stats, "lookup": int(model.tracer._lookup[1].size)}))
"""


def test_two_ranks_give_the_serial_beam() -> None:
    serial = BeamOptics(_planar()).run()
    env = {
        **os.environ,
        "OMP_NUM_THREADS": "1",
        "OMPI_ALLOW_RUN_AS_ROOT": "1",
        "OMPI_ALLOW_RUN_AS_ROOT_CONFIRM": "1",
    }
    done = subprocess.run(
        ["mpirun", "--oversubscribe", "-n", "2", sys.executable, "-c", SCRIPT],
        capture_output=True, text=True, env=env, timeout=600, check=False,
    )  # fmt: skip
    assert done.returncode == 0, done.stderr[-2000:]
    result = json.loads(done.stdout.strip().splitlines()[-1])
    expected = serial.beam_moments(0.115)
    assert result["current"] == pytest.approx(expected.current, rel=1e-12)
    assert result["rms_size"] == pytest.approx(expected.rms_size, rel=1e-9)
    assert result["stats"] == serial.stats
    assert result["lookup"] == serial.tracer._lookup[1].size
