import sys
from pathlib import Path
from typing import Sequence

import tyro

from trainite.cli.init import Init, init_project, run_interactive_mode


def start_interactive_run(script: str, args: list[str]) -> None:
    main_script = str(Path(script).resolve())
    sys.argv = [main_script] + args

    import pudb

    pudb.runscript(main_script)


def main(argv: Sequence[str] | None = None) -> None:
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if not args_list or args_list[0] not in ("init", "i_run"):
        print("Usage: trainite <subcommand> [options]", file=sys.stderr)
        print("\nAvailable subcommands:", file=sys.stderr)
        print("  init      Initialize a new trainite project", file=sys.stderr)
        print("  i_run     Run a python script under the visual debugger", file=sys.stderr)
        print("\nFor help, run: trainite <subcommand> --help", file=sys.stderr)
        sys.exit(1)

    if args_list[0] == "init":
        if len(args_list) == 1:
            run_interactive_mode()
        else:
            args_list = args_list[1:]
            cmd = tyro.cli(Init, args=args_list)
            init_project(cmd)
    elif args_list[0] == "i_run":
        if len(args_list) > 1 and args_list[1] in ("--help", "-h"):
            print("Usage: trainite i_run <script> [args...]")
            print("\nRun a python script under the PuDB visual debugger (or pdb fallback).")
            print("\nArguments:")
            print("  script    The python script to run (e.g., main.py)")
            print("  args      Arguments to pass to the script (e.g., config.yaml)")
            sys.exit(0)

        if len(args_list) < 2:
            print("Error: Please specify the script to run.", file=sys.stderr)
            print("Usage: trainite i_run <script> [args...]", file=sys.stderr)
            sys.exit(1)

        script = args_list[1]
        script_args = args_list[2:]
        start_interactive_run(script, script_args)
