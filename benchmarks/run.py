"""Run the IBSimu benchmarks: each case in both codes, compared and timed.

Usage::

    python benchmarks/run.py                       # all cases, both codes
    python benchmarks/run.py vlasov2d plasmacyl    # some cases
    python benchmarks/run.py --scale 2 --repeats 3 --threads 4 --integrator boris

Every code runs a case in its own process (``--worker``), under a scope-profiler session
that writes ``<out>/<case>_<code>.h5``; its potential and exit beam go to
``<out>/<case>_<code>.npz``. The comparison and the times are printed as Markdown and
written to ``<out>/results.md`` and ``<out>/results.json``. The exit status is 1 when a
compared quantity is outside its tolerance.
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import common3d  # noqa: E402
from common import (  # noqa: E402
    Case,
    Exit,
    Outcome,
    away_from_electrodes,
    compare,
    run_beamtracer,
    run_ibsimu,
)

CASES = (
    "vlasov2d",
    "lens_2d",
    "lens_cyl",
    "two_stage",
    "iter_nbi",
    "plasma2d",
    "plasmacyl",
    "solenoid",
    "sector_magnet",
    "quad3d",
    "solenoid3d",
    "extraction3d",
)
CODES = ("ibsimu", "beamtracer")


def load(name: str, iterations: int | None = None) -> Case | common3d.Case3D:
    """Return the case ``name`` of ``benchmarks/cases``, with ``iterations`` if given."""
    case = importlib.import_module(f"cases.{name}").CASE
    if iterations:
        case = dataclasses.replace(case, iterations=iterations)
    return case


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("cases", nargs="*", default=list(CASES), help="cases to run")
    parser.add_argument("--codes", nargs="+", default=list(CODES), choices=CODES)
    parser.add_argument(
        "--scale", type=int, default=1, help="divide the cells by this factor"
    )
    parser.add_argument(
        "--particles", type=int, default=None, help="trajectories (default: the case's)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=None,
        help="Vlasov iterations (default: the case's)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1, help="runs per code; best time"
    )
    parser.add_argument("--threads", type=int, default=1, help="threads of both codes")
    parser.add_argument(
        "--ibsimu-threads",
        type=int,
        default=None,
        help="IBSimu threads (default --threads)",
    )
    parser.add_argument("--integrator", default="rk4", choices=("rk4", "boris"))
    parser.add_argument(
        "--field-backend",
        default="lu",
        choices=("lu", "amg", "multigrid", "petsc", "auto"),
        help="beamtracer's field solve (field_backend)",
    )
    parser.add_argument("--out", type=Path, default=Path("benchmark_results"))
    parser.add_argument(
        "--worker", nargs=2, metavar=("CASE", "CODE"), help=argparse.SUPPRESS
    )
    return parser.parse_args(argv)


def worker(args: argparse.Namespace) -> None:
    """Run one case in one code, profiled, and save what it gives."""
    import scope_profiler as sp

    name, code = args.worker
    case = load(name, args.iterations)
    nodes, h = case.mesh(args.scale)
    particles = args.particles or case.particles
    stem = args.out / f"{name}_{code}"

    def region(part: str) -> Any:
        return sp.region(f"bench.{part}")

    importlib.import_module(code)  # imports are not timed
    outcome = None
    with sp.session(
        output=str(stem.with_suffix(".h5")), enable_tags=["beamtracer"], verbose=False
    ):
        for _ in range(args.repeats):
            with sp.region("bench.total"):
                if isinstance(case, common3d.Case3D):
                    run = (
                        common3d.run_ibsimu
                        if code == "ibsimu"
                        else common3d.run_beamtracer
                    )
                    setting = (
                        args.ibsimu_threads if code == "ibsimu" else args.integrator
                    )
                    outcome = run(case, nodes, h, particles, setting, region)  # type: ignore[arg-type]
                elif code == "ibsimu":
                    outcome = run_ibsimu(
                        case, nodes, h, particles, args.ibsimu_threads, region
                    )
                else:
                    outcome = run_beamtracer(
                        case, nodes, h, particles, args.integrator, region
                    )
    assert outcome is not None
    beam = outcome.exit
    if isinstance(case, common3d.Case3D):
        away = common3d.away_from_electrodes(case, nodes, h)  # type: ignore[arg-type]
        arrays = {"y": beam.y, "yp": beam.yp, "z": beam.z, "zp": beam.zp}
    else:
        away = away_from_electrodes(case, nodes, h)
        arrays = {"u": beam.u, "up": beam.up}
    np.savez(
        stem.with_suffix(".npz"),
        potential=outcome.potential,
        away=away,
        current=beam.current,
        energy=beam.energy,
        **arrays,
    )


def outcome_of(path: Path) -> Outcome | common3d.Outcome3D:
    """Read what a worker saved."""
    data = np.load(path)
    if "z" in data:
        return common3d.Outcome3D(
            data["potential"],
            common3d.Exit3D(
                data["y"],
                data["yp"],
                data["z"],
                data["zp"],
                data["current"],
                data["energy"],
            ),
            data["away"],
        )
    return Outcome(
        data["potential"],
        Exit(data["u"], data["up"], data["current"], data["energy"]),
        data["away"],
    )


def times(path: Path, code: str) -> dict[str, float | None]:
    """Return the best total and setup times and the mean solve and trace times of a run."""
    import scope_profiler as sp

    statistics = sp.collect_region_statistics(sp.read_profile(str(path)))
    regions = statistics["files"][0]["region_statistics"]

    def get(name: str, key: str) -> float | None:
        return regions[name][key] if name in regions else None

    library = "bench" if code == "ibsimu" else "beamtracer"
    return {
        "total": get("bench.total", "min_duration_seconds"),
        "setup": get("bench.setup", "min_duration_seconds"),
        "field_solve": get(f"{library}.field_solve", "average_duration_seconds"),
        "trace": get(f"{library}.trace", "average_duration_seconds"),
    }


def run_case(name: str, args: argparse.Namespace) -> dict[str, Any]:
    """Run a case in the codes, each in a worker process, and compare them."""
    env = dict(
        os.environ,
        OMP_NUM_THREADS=str(args.threads),
        OPENBLAS_NUM_THREADS="1",
        BENCHMARK_FIELD_BACKEND=args.field_backend,
    )
    for code in args.codes:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--worker",
            name,
            code,
            "--scale",
            str(args.scale),
            "--repeats",
            str(args.repeats),
            "--threads",
            str(args.threads),
            "--ibsimu-threads",
            str(args.ibsimu_threads),
            "--integrator",
            args.integrator,
            "--out",
            str(args.out),
        ]
        if args.particles:
            command += ["--particles", str(args.particles)]
        if args.iterations:
            command += ["--iterations", str(args.iterations)]
        if subprocess.run(command, env=env).returncode != 0:
            return {"case": name, "error": f"{code} failed", "times": {}}
    result: dict[str, Any] = {
        "case": name,
        "times": {c: times(args.out / f"{name}_{c}.h5", c) for c in args.codes},
    }
    if set(args.codes) == set(CODES):
        outcomes = {c: outcome_of(args.out / f"{name}_{c}.npz") for c in CODES}
        case = load(name)
        compare_case = (
            common3d.compare if isinstance(case, common3d.Case3D) else compare
        )
        result["comparison"] = compare_case(
            case,  # type: ignore[arg-type]
            outcomes["ibsimu"],  # type: ignore[arg-type]
            outcomes["beamtracer"],  # type: ignore[arg-type]
        )
    return result


def _seconds(value: float | None) -> str:
    """Format a time."""
    return "-" if value is None else f"{value:.3g} s"


def _number(value: float | None) -> str:
    """Format a value."""
    return "-" if value is None else f"{value:.5g}"


def report(results: list[dict[str, Any]], args: argparse.Namespace) -> str:
    """Return the results as Markdown."""
    lines = [
        "# IBSimu benchmarks",
        "",
        f"Scale {args.scale}, {args.threads} thread(s) (IBSimu {args.ibsimu_threads}), "
        f"beamtracer {args.integrator} ({args.field_backend} field solve), best of {args.repeats}.",
        "",
        "| Case | Agrees | IBSimu | IBSimu setup | beamtracer | Speedup | Without IBSimu setup |",
        "|---|---|---|---|---|---|---|",
    ]
    for result in results:
        t = result["times"]
        ok = result.get("comparison")
        agrees = (
            "-" if ok is None else ("yes" if all(r["ok"] for r in ok) else "**no**")
        )
        if "error" in result:
            agrees = f"**{result['error']}**"
        ib, bt = t.get("ibsimu", {}), t.get("beamtracer", {})
        speedup = without = "-"
        if ib.get("total") and bt.get("total"):
            speedup = f"{ib['total'] / bt['total']:.1f}x"
            without = f"{(ib['total'] - (ib.get('setup') or 0.0)) / bt['total']:.1f}x"
        lines.append(
            f"| {result['case']} | {agrees} | {_seconds(ib.get('total'))} | "
            f"{_seconds(ib.get('setup'))} | {_seconds(bt.get('total'))} | {speedup} | {without} |"
        )
    for result in results:
        if "comparison" not in result:
            continue
        case = load(result["case"])
        lines += [
            "",
            f"## {case.name}: {case.title}",
            "",
            f"IBSimu example `{case.source}`. {case.notes}".rstrip(),
            "",
            "| Quantity | IBSimu | beamtracer | Difference | Tolerance | |",
            "|---|---|---|---|---|---|",
        ]
        for row in result["comparison"]:
            tolerance = "-" if row["tolerance"] is None else f"{row['tolerance']:.2g}"
            mark = (
                "" if row["tolerance"] is None else ("ok" if row["ok"] else "**fails**")
            )
            lines.append(
                f"| {row['quantity']} | {_number(row['ibsimu'])} | {_number(row['beamtracer'])} | "
                f"{row['difference']:.2e} | {tolerance} | {mark} |"
            )
        t = result["times"]
        lines += [
            "",
            "| Code | Total | Setup | Field solve (mean) | Trace (mean) |",
            "|---|---|---|---|---|",
        ]
        for code, value in t.items():
            lines.append(
                f"| {code} | {_seconds(value['total'])} | {_seconds(value['setup'])} | "
                f"{_seconds(value['field_solve'])} | {_seconds(value['trace'])} |"
            )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    """Run the benchmarks; return 1 if a case does not agree."""
    args = arguments(argv)
    if args.ibsimu_threads is None:
        args.ibsimu_threads = args.threads
    args.out.mkdir(parents=True, exist_ok=True)
    if args.worker:
        worker(args)
        return 0
    unknown = sorted(set(args.cases) - set(CASES))
    if unknown:
        raise SystemExit(f"unknown cases {unknown}; known: {', '.join(CASES)}")
    results = [run_case(name, args) for name in args.cases]
    text = report(results, args)
    print(text)
    (args.out / "results.md").write_text(text)
    (args.out / "results.json").write_text(json.dumps(results, indent=2))
    failed = any("error" in res for res in results)
    return int(
        failed or any(not r["ok"] for res in results for r in res.get("comparison", []))
    )


if __name__ == "__main__":
    raise SystemExit(main())
