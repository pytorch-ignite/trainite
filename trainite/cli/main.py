import sys
from typing import Annotated, Sequence, Union

import tyro

from trainite import __version__
from trainite.cli.init import Init, init_project, run_interactive_mode
from trainite.cli.sky import SkyInit, init_sky

TrainiteCLI = Union[
    Annotated[Init, tyro.conf.subcommand(name="init", description="Initialize a new trainite project")],
    Annotated[
        SkyInit,
        tyro.conf.subcommand(
            name="add:sky",
            description="Add SkyPilot cloud configuration to the current experiment",
        ),
    ],
]


def main(argv: Sequence[str] | None = None) -> None:
    args_list = list(argv) if argv is not None else sys.argv[1:]

    if args_list and args_list[0] in ("--version", "-V"):
        print("Trainite, https://github.com/pytorch-ignite/trainite/")
        print(f"Version: {__version__}")
        return

    # Trigger interactive mode when `trainite init` is called with no further arguments
    if args_list == ["init"]:
        run_interactive_mode()
        return

    cmd = tyro.cli(TrainiteCLI, args=args_list)

    if isinstance(cmd, Init):
        init_project(cmd)
    elif isinstance(cmd, SkyInit):
        init_sky(cmd)
