import questionary
from pathlib import Path


def serve_input(directory: str, key: str):
    dir_path = Path(directory)

    # Check if directory exists
    if not dir_path.exists() or not dir_path.is_dir():
        print(f"Error: The directory '{directory}' does not exist.")
        return None

    files = [
        f.name for f in dir_path.iterdir() if f.is_file() and f.name.startswith(key)
    ]

    if not files:
        print(f"Warning No files found inside the '{directory}' directory.")
        return None

    selected_file: str | None = questionary.select(
        "Which event file would you like to process?",
        choices=files,
        instruction="(Use arrow keys to navigate, Enter to select)",
    ).ask()

    if selected_file:
        return dir_path / selected_file
    return None
