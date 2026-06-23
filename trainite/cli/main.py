import sys
from typing import Sequence

import tyro

from trainite.cli.init import Init, init_project, run_interactive_mode


def main(argv: Sequence[str] | None = None) -> None:
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if not args_list:
        print("Usage: trainite <subcommand> [options]", file=sys.stderr)
        print("\nAvailable subcommands:", file=sys.stderr)
        print("  init      Initialize a new trainite project", file=sys.stderr)
        print("\nFor help, run: trainite --help or trainite -h", file=sys.stderr)
        sys.exit(1)

    if args_list == ["init"]:
        run_interactive_mode()
    else:
        if args_list[0] == "init":
            args_list = args_list[1:]

        cmd = tyro.cli(Init, args=args_list)
        init_project(cmd)
