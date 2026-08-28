from pathlib import Path
import sys
from pydantic import BaseModel
import tomlkit
import yaml
from trainite.cli.init import parse_dependencies, PROJECT_ROOT, PACKAGE_ROOT


class SkyInit(BaseModel):
    """Add a SkyPilot configuration (sky.yaml) and cloud dependencies to the current Trainite experiment.

    Run this command inside an existing Trainite experiment directory.

    Args:
        force: Overwrite existing sky.yaml if it already exists.
    """

    force: bool = False


def get_skypilot_dependency() -> str:
    """Get the canonical skypilot dependency string from Trainite's pyproject.toml."""
    toml_path = PACKAGE_ROOT / "pyproject.toml"
    if not toml_path.exists():
        toml_path = PROJECT_ROOT / "pyproject.toml"
    if toml_path.exists():
        _, other_deps = parse_dependencies(toml_path)
        if "skypilot" in other_deps:
            return other_deps["skypilot"]
    return "skypilot"


def _add_sky_dependency(pyproject_path: Path) -> bool:
    """Add skypilot to project.dependencies in pyproject.toml."""
    try:
        content = pyproject_path.read_text()
        doc = tomlkit.parse(content)

        project = doc.setdefault("project", tomlkit.table())
        dependencies = project.setdefault("dependencies", tomlkit.array())

        # Check if already present
        has_sky = any(isinstance(dep, str) and dep.startswith("skypilot") for dep in dependencies)
        if not has_sky:
            sky_dep = get_skypilot_dependency()
            dependencies.append(sky_dep)
            pyproject_path.write_text(tomlkit.dumps(doc))
            return True
        return False
    except Exception as e:
        print(f"Warning: Could not update pyproject.toml: {e}", file=sys.stderr)
        return False


def init_sky(config: SkyInit) -> None:
    current_dir = Path.cwd()
    config_yaml_path = current_dir / "config.yaml"
    main_py_path = current_dir / "main.py"
    pyproject_path = current_dir / "pyproject.toml"

    # Validation: Must be inside an existing Trainite project
    if not (config_yaml_path.exists() and main_py_path.exists()):
        print(
            f"Error: '{current_dir.name}' is not a valid Trainite project directory "
            "(missing config.yaml and main.py).\n"
            "Please navigate into your Trainite experiment directory before running 'trainite add:sky'.",
            file=sys.stderr,
        )
        sys.exit(1)

    target_file = current_dir / "sky.yaml"
    if target_file.exists() and not config.force:
        print(
            f"Error: '{target_file.name}' already exists in {current_dir.name}.\n"
            "Pass '--force' to overwrite it: trainite add:sky --force",
            file=sys.stderr,
        )
        sys.exit(1)

    # Extract project_name from config.yaml if available, else folder name
    project_name = current_dir.name
    try:
        data = yaml.safe_load(config_yaml_path.read_text()) or {}
        if isinstance(data, dict) and "project_name" in data:
            project_name = data["project_name"]
    except Exception:
        pass

    # Render template
    template_path = PACKAGE_ROOT / "templates" / "integrations" / "sky.yaml"
    content = template_path.read_text().replace("{{project_name}}", project_name)
    target_file.write_text(content)

    print(f"✔ Generated sky.yaml for '{project_name}'")

    # Update pyproject.toml
    if pyproject_path.exists():
        if _add_sky_dependency(pyproject_path):
            print("✔ Added 'skypilot' to pyproject.toml dependencies")

    print("\nNext steps to run on the cloud with SkyPilot:")
    print("  1. Update environment:      pip install -e .  (or: uv sync)")
    print("  2. Check cloud setup:       sky check")
    print("  3. Launch experiment:       sky launch sky.yaml")
    print("  4. View status / logs:      sky queue / sky logs <cluster_name>")
    print("  5. Teardown cluster:        sky down <cluster_name>")
