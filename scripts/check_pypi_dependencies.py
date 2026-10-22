"""Check that every dependency in pyproject.toml can be installed from PyPI.

PyPI rejects an upload whose metadata has a direct reference such as
``cunumpy @ git+https://github.com/org/cunumpy.git@devel`` ("400 Can't have direct
dependency"), and the release fails after the version is merged. This check catches that,
and its sibling, in a pull request instead:

* a requirement with a URL (``name @ git+https://...``, ``name @ file:///...``);
* a project that PyPI does not know;
* a version range that no release on PyPI satisfies, e.g. ``cunumpy>=0.7`` before 0.7.0 is
  published, or only yanked releases do.

The project's own name is skipped, so the ``name[docs,test]`` entry of a ``dev`` extra is
fine. Requirement markers (``; python_version < "3.11"``) are not evaluated: a dependency is
checked whether or not the marker applies. ``[dependency-groups]`` are not part of the
published metadata and are not checked.

Usage::

    python scripts/check_pypi_dependencies.py [pyproject.toml]

Needs ``packaging`` (``uv run --no-project --with packaging python scripts/...``). Exits with
status 1 and prints GitHub Actions annotations when something fails.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path

import tomllib
from packaging.requirements import InvalidRequirement, Requirement
from packaging.utils import canonicalize_name
from packaging.version import InvalidVersion, Version

PYPI_URL = "https://pypi.org/pypi/{name}/json"

#: ``{version: [file entries]}`` of a project on PyPI, or ``None`` if PyPI does not know it.
Releases = dict[str, list[dict]]
Fetch = Callable[[str], Releases | None]


class PyPIUnreachableError(RuntimeError):
    """PyPI could not be queried, so the check can say nothing."""


def fetch_releases(name: str, attempts: int = 3) -> Releases | None:
    """Return the releases of ``name`` on PyPI, or ``None`` if there is no such project."""
    url = PYPI_URL.format(name=name)
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(url, timeout=20) as response:  # noqa: S310
                return json.load(response)["releases"]
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None
            failure = f"HTTP {error.code}"
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as error:
            failure = str(error)
        if attempt < attempts:
            time.sleep(2**attempt)
    raise PyPIUnreachableError(f"could not query {url}: {failure}")


def requirement_strings(pyproject: dict) -> list[tuple[str, str]]:
    """Return ``(where, requirement)`` for the dependencies and every optional dependency."""
    project = pyproject.get("project", {})
    found = [("dependencies", text) for text in project.get("dependencies", [])]
    for extra, texts in project.get("optional-dependencies", {}).items():
        found += [(f"optional-dependencies.{extra}", text) for text in texts]
    return found


def is_installable(releases: Releases, requirement: Requirement) -> bool:
    """Whether some release on PyPI that is not yanked satisfies the requirement."""
    for text, files in releases.items():
        if not files or all(entry.get("yanked", False) for entry in files):
            continue
        try:
            version = Version(text)
        except InvalidVersion:
            continue
        if requirement.specifier.contains(version, prereleases=None):
            return True
    return False


def check(pyproject: dict, fetch: Fetch = fetch_releases) -> list[str]:
    """Return one message per dependency that PyPI would reject or that cannot be installed."""
    project = pyproject.get("project", {})
    own_name = canonicalize_name(project.get("name", ""))
    problems = []
    if "dependencies" in project.get("dynamic", []):
        problems.append(
            "dependencies are dynamic, so they cannot be read from pyproject.toml: "
            "check the built metadata (Requires-Dist) instead"
        )
    cache: dict[str, Releases | None] = {}
    for where, text in requirement_strings(pyproject):
        try:
            requirement = Requirement(text)
        except InvalidRequirement as error:
            problems.append(f"[{where}] {text!r} is not a valid requirement: {error}")
            continue
        if requirement.url:
            problems.append(
                f"[{where}] {text!r} is a direct reference ({requirement.url}); PyPI rejects "
                f"uploads with one. Use a version range of a release on PyPI, e.g. "
                f'"{requirement.name}>=<version>", and keep the git URL for local development in '
                f"[tool.uv.sources] or a dependency group."
            )
            continue
        name = canonicalize_name(requirement.name)
        if name == own_name:
            continue
        if name not in cache:
            cache[name] = fetch(name)
        releases = cache[name]
        if releases is None:
            problems.append(
                f"[{where}] {text!r}: no project {requirement.name!r} on PyPI"
            )
        elif not is_installable(releases, requirement):
            newest = max(
                (v for v in releases if releases[v]),
                key=_version_key,
                default="none",
            )
            problems.append(
                f"[{where}] {text!r}: no release on PyPI satisfies it "
                f"(newest: {newest}); publish the dependency first, or loosen the range"
            )
    return problems


def _version_key(text: str) -> Version:
    try:
        return Version(text)
    except InvalidVersion:
        return Version("0")


def main(argv: list[str] | None = None) -> int:
    """Run the check on a pyproject.toml; return the exit status."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("pyproject", nargs="?", default="pyproject.toml", type=Path)
    args = parser.parse_args(argv)

    with args.pyproject.open("rb") as handle:
        pyproject = tomllib.load(handle)
    try:
        problems = check(pyproject)
    except PyPIUnreachableError as error:
        print(f"::error file={args.pyproject}::{error}")
        return 1

    total = len(requirement_strings(pyproject))
    if problems:
        for problem in problems:
            print(f"::error file={args.pyproject}::{problem}")
        print(
            f"\n{len(problems)} of {total} dependencies cannot be installed from PyPI."
        )
        return 1
    print(f"All {total} dependencies in {args.pyproject} can be installed from PyPI.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
