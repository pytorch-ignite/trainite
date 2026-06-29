import sys

import tyro

from trainite.cli.i_run import IRun, start_interactive_run
from trainite.cli.init import Init, init_project, run_interactive_mode


def main() -> None:
    if sys.argv[1:] == ["init"]:
        run_interactive_mode()
        return

    cmd = tyro.cli(Init | IRun, prog="trainite")

    if isinstance(cmd, Init):
        init_project(cmd)
    elif isinstance(cmd, IRun):
        start_interactive_run(cmd)
