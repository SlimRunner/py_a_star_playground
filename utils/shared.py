import questionary
from a_star.typing import heur_cb_type
from pathlib import Path


def select_heuristic(options: dict[str, heur_cb_type]) -> heur_cb_type:
    key: str = questionary.select(
        "Select a heuristic:",
        choices=list(options.keys()),
        instruction="(Use arrow keys to navigate, Enter to select)",
    ).ask()
    return options[key]


def read_int_matrix(path: Path):
    out: list[list[int]] = []
    with open(path) as file:
        line = file.readline()
        while line:
            if line.startswith("#"):
                print(line.lstrip("#").lstrip(" "))
                line = file.readline()
                continue
            out.append([int(s) for s in line.split(",")])
            line = file.readline()
    return out
