"""The PyPI dependency check (scripts/check_pypi_dependencies.py), against a fake PyPI."""

import importlib.util
import sys
from pathlib import Path

import pytest

# the script needs packaging, and tomllib (Python 3.11+); the workflow runs it on 3.12
pytest.importorskip("packaging")
tomllib = pytest.importorskip("tomllib")

ROOT = Path(__file__).resolve().parents[4]
SCRIPT = ROOT / "scripts" / "check_pypi_dependencies.py"

spec = importlib.util.spec_from_file_location("check_pypi_dependencies", SCRIPT)
checker = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = checker
spec.loader.exec_module(checker)

#: what the fake PyPI knows: project -> {version: files}
RELEASES = {
    "numpy": {"1.26.0": [{}], "2.0.0": [{}], "2.1.0rc1": [{}]},
    "cunumpy": {"0.6.2": [{}], "0.6.3": [{}]},
    "yankedonly": {"1.0": [{"yanked": True}]},
    "empty": {"1.0": []},
}


def fake_fetch(name):
    return RELEASES.get(name)


def run(project, **tables):
    return checker.check({"project": {"name": "demo", **project}, **tables}, fake_fetch)


def test_plain_dependencies_pass():
    assert run({"dependencies": ["numpy", "cunumpy>=0.6.3"]}) == []


@pytest.mark.parametrize(
    "requirement",
    [
        "cunumpy @ git+https://github.com/max-models/cunumpy.git@devel",
        "cunumpy @ https://example.org/cunumpy-0.6.3.tar.gz",
        "cunumpy @ file:///tmp/cunumpy",
    ],
)
def test_direct_references_are_rejected(requirement):
    problems = run({"dependencies": [requirement]})
    assert len(problems) == 1
    assert "direct reference" in problems[0]
    assert "cunumpy" in problems[0]


def test_direct_references_in_extras_are_rejected_too():
    problems = run(
        {"optional-dependencies": {"dev": ["cunumpy @ git+https://x/y.git"]}}
    )
    assert len(problems) == 1
    assert problems[0].startswith("[optional-dependencies.dev]")


def test_unknown_project():
    problems = run({"dependencies": ["not-on-pypi"]})
    assert problems == [
        "[dependencies] 'not-on-pypi': no project 'not-on-pypi' on PyPI"
    ]


def test_range_no_release_satisfies():
    problems = run({"dependencies": ["cunumpy>=0.7"]})
    assert len(problems) == 1
    assert "no release on PyPI satisfies" in problems[0]
    assert "newest: 0.6.3" in problems[0]


def test_yanked_and_file_less_releases_do_not_count():
    assert len(run({"dependencies": ["yankedonly"]})) == 1
    assert len(run({"dependencies": ["empty"]})) == 1


def test_prereleases_count_only_when_asked_for():
    assert run({"dependencies": ["numpy>=2.1"]}) != []
    assert run({"dependencies": ["numpy>=2.1.0rc1"]}) == []


def test_the_projects_own_name_is_skipped_with_extras():
    assert (
        run({"optional-dependencies": {"dev": ["demo[docs,test]"], "docs": ["numpy"]}})
        == []
    )
    assert run({"optional-dependencies": {"dev": ["Demo[docs]"]}}) == []


def test_own_name_with_a_direct_reference_is_still_rejected():
    assert len(run({"dependencies": ["demo @ git+https://x/y.git"]})) == 1


def test_markers_are_checked_regardless_of_the_environment():
    assert run({"dependencies": ['numpy; python_version < "3.9"']}) == []
    assert len(run({"dependencies": ['not-on-pypi; python_version < "3.9"']})) == 1


def test_invalid_requirement():
    problems = run({"dependencies": ["numpy>>=1"]})
    assert len(problems) == 1
    assert "not a valid requirement" in problems[0]


def test_dynamic_dependencies_are_reported():
    problems = run({"dynamic": ["dependencies"]})
    assert len(problems) == 1
    assert "dynamic" in problems[0]


def test_each_project_is_looked_up_once():
    calls = []

    def counting_fetch(name):
        calls.append(name)
        return RELEASES.get(name)

    pyproject = {
        "project": {
            "name": "demo",
            "dependencies": ["numpy"],
            "optional-dependencies": {"a": ["numpy>=1"], "b": ["NumPy"]},
        }
    }
    assert checker.check(pyproject, counting_fetch) == []
    assert calls == ["numpy"]


def test_main_exit_status(tmp_path, capsys):
    good = tmp_path / "good.toml"
    good.write_text('[project]\nname = "demo"\ndependencies = []\n')
    assert checker.main([str(good)]) == 0
    assert "All 0 dependencies" in capsys.readouterr().out

    bad = tmp_path / "bad.toml"
    bad.write_text(
        '[project]\nname = "demo"\ndependencies = ["x @ git+https://y/z.git"]\n'
    )
    assert checker.main([str(bad)]) == 1
    out = capsys.readouterr().out
    assert out.startswith("::error file=")
    assert "1 of 1 dependencies" in out


def test_this_projects_pyproject_has_only_pypi_dependencies():
    """No network: the requirements are valid and none is a direct reference."""
    with (ROOT / "pyproject.toml").open("rb") as handle:
        pyproject = tomllib.load(handle)
    for where, text in checker.requirement_strings(pyproject):
        requirement = checker.Requirement(text)
        assert requirement.url is None, f"[{where}] {text} is a direct reference"
