from trainite.cli.mocker import mock_dependencies

# Mock torch and ignite if they are not installed, to prevent ImportErrors during CLI loading
mock_dependencies("torch", "ignite")

from trainite.cli.init import init_project
from trainite.cli.main import main

__all__ = ["init_project", "main"]
