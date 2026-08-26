"""
test_workflows.py -- the CI workflows must be able to run the code they invoke.

WHY THIS EXISTS
---------------
quote-maintenance.yml installed only `groq`, on the reasoning that quotes.py
needs nothing else. But maintain_pool.py imports PHILOSOPHER_QUOTES from
fetcher.py, which imports `requests` at module scope for the image fetchers.
Every scheduled run from 2026-08-09 to 2026-08-23 died with
ModuleNotFoundError, and because the report step read a stale committed
pool_status.json it still announced "Pool healthy" -- so three weeks of
failures looked like noise in the inbox.

These tests walk the real import graph of each script the workflows run and
assert the declared dependencies cover it. Adding an import to anything in the
maintain_pool -> quotes -> fetcher chain without updating the install step
fails here instead of silently on a Sunday.
"""
import ast
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORKFLOWS = ROOT / ".github" / "workflows"

# Import name -> pip distribution name, where they differ.
_DIST_ALIASES = {
    "yaml": "pyyaml",
    "PIL": "pillow",
    "dateutil": "python-dateutil",
}

# Modules that ship with CPython and never need installing. Checked against the
# real stdlib list so this cannot drift.
_STDLIB = set(sys.stdlib_module_names)


def _local_modules() -> set[str]:
    """Top-level .py files in the repo -- importable without pip."""
    return {p.stem for p in ROOT.glob("*.py")}


def _top_level_imports(path: Path) -> set[str]:
    """Every module name imported at any depth in one file."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    found: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            found.update(a.name.split(".")[0] for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            # level > 0 is a relative import; this repo is flat, so ignore.
            if node.level == 0 and node.module:
                found.add(node.module.split(".")[0])
    return found


def import_closure(entry: Path) -> set[str]:
    """Transitively resolve imports, following local modules into their files."""
    local = _local_modules()
    seen_files: set[Path] = set()
    pending = [entry]
    external: set[str] = set()
    while pending:
        current = pending.pop()
        if current in seen_files or not current.exists():
            continue
        seen_files.add(current)
        for name in _top_level_imports(current):
            if name in _STDLIB:
                continue
            if name in local:
                pending.append(ROOT / f"{name}.py")
            else:
                external.add(name)
    return external


def declared_packages(workflow: Path) -> set[str]:
    """pip install targets declared anywhere in a workflow file."""
    text = workflow.read_text(encoding="utf-8")
    out: set[str] = set()
    for line in re.findall(r"pip install ([^\n|]+)", text):
        for token in line.split():
            if token.startswith("-"):
                continue
            # strip version pins
            out.add(re.split(r"[<>=!~\[]", token)[0].strip().lower())
    return out


# --- the regression --------------------------------------------------------

def test_maintenance_workflow_installs_its_import_chain():
    """The 2026-08-09 breakage: maintain_pool needs requests via fetcher."""
    wf = WORKFLOWS / "quote-maintenance.yml"
    assert wf.exists()

    needed = import_closure(ROOT / "scripts" / "maintain_pool.py")
    declared = declared_packages(wf)
    if re.search(r"pip install .*-r\s+\S*requirements\.txt", wf.read_text(encoding="utf-8")):
        pytest.skip("workflow installs requirements.txt wholesale")

    missing = {
        m for m in needed
        if _DIST_ALIASES.get(m, m).lower() not in declared
    }
    assert not missing, (
        f"quote-maintenance.yml runs maintain_pool.py, which imports {sorted(missing)}, "
        f"but only installs {sorted(declared)}. Add them to the install step."
    )


def test_maintenance_chain_actually_needs_requests():
    """Guards the test above from silently passing if the chain changes."""
    needed = import_closure(ROOT / "scripts" / "maintain_pool.py")
    assert "requests" in needed
    assert "groq" in needed


def test_import_closure_follows_local_modules():
    """maintain_pool imports nothing third-party directly; it inherits it."""
    direct = _top_level_imports(ROOT / "scripts" / "maintain_pool.py")
    assert "requests" not in direct, "if this becomes direct, the test still holds"
    assert "requests" in import_closure(ROOT / "scripts" / "maintain_pool.py")


# --- staleness guard -------------------------------------------------------

def test_status_file_is_removed_before_regeneration():
    """A crashed run must not be reported healthy from last week's status file."""
    text = (WORKFLOWS / "quote-maintenance.yml").read_text(encoding="utf-8")
    topup = text.index("Top up the quote pool")
    report = text.index("Report pool problems")
    assert "rm -f runs/pool_status.json" in text[topup:report], (
        "pool_status.json must be deleted before maintenance runs, or a stale "
        "committed copy makes a failed run look healthy."
    )


def test_missing_status_file_notifies_and_fails():
    """Absent status file means the job crashed; it must not exit 0 quietly."""
    text = (WORKFLOWS / "quote-maintenance.yml").read_text(encoding="utf-8")
    block = text[text.index("if [ ! -f runs/pool_status.json ]"):]
    end = re.search(r"^\s*fi\s*$", block, re.MULTILINE)
    block = block[:end.start()] if end else block
    assert "notify" in block, "a crashed run must open an issue"
    assert "exit 1" in block, "a crashed run must fail the job"


# --- both workflows --------------------------------------------------------

@pytest.mark.parametrize("name", ["pipeline.yml", "quote-maintenance.yml"])
def test_workflow_parses(name):
    yaml = pytest.importorskip("yaml")
    data = yaml.safe_load((WORKFLOWS / name).read_text(encoding="utf-8"))
    assert data["jobs"], f"{name} declares no jobs"


def test_earned_state_is_committed_back():
    """CI must commit runs/ or every ledger and pool row dies with the runner."""
    for name in ("pipeline.yml", "quote-maintenance.yml"):
        text = (WORKFLOWS / name).read_text(encoding="utf-8")
        assert re.search(r"git add [^\n]*runs/", text), (
            f"{name} does not commit runs/ back; earned state would be lost."
        )
