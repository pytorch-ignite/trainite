from trainite.shared.mocker import mock_dependencies

# Mock torch and ignite if they are not installed, to prevent ImportErrors during CLI loading
mock_dependencies("torch", "ignite")

__version__ = "0.0.1"
