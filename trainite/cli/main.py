import sys
from typing import Sequence

import tyro

from trainite import __version__
from trainite.cli.init import Init, init_project, run_interactive_mode


def main(argv: Sequence[str] | None = None) -> None:
    args_list = list(argv) if argv is not None else sys.argv[1:]

    commands = {
        "init": "Initialize a new trainite project",
    }

    if args_list and args_list[0] in ("--version", "-V"):
        print("Trainite, https://github.com/pytorch-ignite/trainite/")
        print(f"Version: {__version__}")
        return

    if not args_list or args_list[0] not in commands:
        print("Usage: trainite <subcommand> [options]", file=sys.stderr)
        print("\nAvailable subcommands:", file=sys.stderr)
        for cmd, desc in commands.items():
            print(f"  {cmd:<10}{desc}", file=sys.stderr)
        print("\nFor help, run: trainite init --help or trainite -h", file=sys.stderr)
        sys.exit(1)

    subcommand = args_list[0]
    if subcommand == "init":
        if len(args_list) == 1:
            run_interactive_mode()
        else:
            cmd = tyro.cli(Init, args=args_list[1:])
            init_project(cmd)
