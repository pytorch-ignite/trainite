import dataclasses
import sys
from pathlib import Path

import tyro


@dataclasses.dataclass
class IRun:
    """Run a script under the PuDB visual debugger.

    Launches the specified Python script in PuDB's interactive
    step-through debugger. All additional arguments are forwarded
    to the script via sys.argv.

    Example usage::

        trainite i-run main.py config.yaml

    Args:
        script: The Python script to debug (e.g., main.py).
        script_args: Arguments forwarded to the script (e.g., config.yaml).
    """

    script: tyro.conf.Positional[str]
    script_args: tyro.conf.Positional[tuple[str, ...]] = ()


def start_interactive_run(config: IRun) -> None:
    """Launch a script under PuDB.

    Args:
        config: Configuration for the interactive run.
    """
    main_script = str(Path(config.script).resolve())
    sys.argv = [main_script] + list(config.script_args)

    import importlib

    pudb = importlib.import_module("pudb")
    pudb.runscript(main_script)
