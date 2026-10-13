from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import shutil
import sys


def _modules_available() -> bool:
    return (
        importlib.util.find_spec('sumolib') is not None
        and importlib.util.find_spec('traci') is not None
    )


def _candidate_tools_dirs() -> list[Path]:
    candidates: list[Path] = []

    sumo_home = os.environ.get('SUMO_HOME')
    if sumo_home:
        home = Path(sumo_home).expanduser()
        candidates.extend((home / 'tools', home))

    # If sumo/sumo-gui is already on PATH, infer <SUMO_HOME>/tools from <SUMO_HOME>/bin/sumo.exe.
    for executable in ('sumo', 'sumo.exe', 'sumo-gui', 'sumo-gui.exe'):
        resolved = shutil.which(executable)
        if not resolved:
            continue
        exe = Path(resolved).resolve()
        if exe.parent.name.lower() == 'bin':
            candidates.append(exe.parent.parent / 'tools')
        candidates.append(exe.parent / 'tools')

    if os.name == 'nt':
        roots = [
            os.environ.get('ProgramFiles(x86)'),
            os.environ.get('ProgramFiles'),
            os.environ.get('LOCALAPPDATA'),
        ]
        relative = [
            Path('Eclipse') / 'Sumo' / 'tools',
            Path('SUMO') / 'tools',
        ]
        for root in roots:
            if not root:
                continue
            base = Path(root)
            for suffix in relative:
                candidates.append(base / suffix)

    # Preserve order while removing duplicates.
    result: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        key = str(candidate).lower() if os.name == 'nt' else str(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
    return result


def ensure_sumo_python_tools() -> Path | None:
    """Make SUMO's Python tools importable without requiring Spyder PYTHONPATH edits.

    Returns the tools directory that was added, or ``None`` when ``sumolib`` and
    ``traci`` were already importable (for example from pip/conda).
    """
    if _modules_available():
        return None

    checked: list[Path] = []
    for tools_dir in _candidate_tools_dirs():
        checked.append(tools_dir)
        if not tools_dir.is_dir():
            continue
        # A normal SUMO tools directory contains both packages/directories.
        if not (tools_dir / 'sumolib').exists() or not (tools_dir / 'traci').exists():
            continue
        text = str(tools_dir)
        if text not in sys.path:
            sys.path.insert(0, text)
        if _modules_available():
            # Also populate SUMO_HOME when we inferred the installation from
            # PATH/common Windows locations. sumolib.checkBinary('sumo') then
            # resolves <SUMO_HOME>/bin/sumo(.exe) without requiring PATH edits.
            home = tools_dir.parent if tools_dir.name.lower() == 'tools' else tools_dir
            if (home / 'bin').is_dir():
                os.environ.setdefault('SUMO_HOME', str(home))
            return tools_dir

    checked_text = '\n'.join(f'  - {path}' for path in checked) or '  (no candidate paths were available)'
    raise ModuleNotFoundError(
        "SUMO Python tools ('sumolib' and 'traci') were not found.\n"
        "The clean runner tried SUMO_HOME/tools, the directory inferred from sumo.exe on PATH, "
        "and common Windows SUMO install locations.\n\n"
        "Fix one of the following and rerun the Spyder script:\n"
        "  1) Set environment variable SUMO_HOME to your SUMO installation folder, e.g.\n"
        "     C:\\Program Files (x86)\\Eclipse\\Sumo\n"
        "     then restart Spyder; or\n"
        "  2) Put SUMO's bin directory on PATH; or\n"
        "  3) Install the Python packages in the same interpreter used by Spyder:\n"
        "     python -m pip install sumolib traci\n\n"
        "Candidate SUMO tools directories checked:\n"
        f"{checked_text}"
    )
