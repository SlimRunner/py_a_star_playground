import numpy as np
import numpy.typing as npt

from a_star.algorithms import (
    find_paths,
)
from a_star.typing import (
    mixed_item_type,
    np_state_type,
)
from rich import print
from weird_maze_2.typing import (
    jump_policy_cb,
    tile_search_cb,
)
from utils.visualizers import (animate_board, prettyState)
from utils.file_ops import serve_input


class tile_type:
    goal_tile = 0x1
    white_tile = 0x2
    green_tile = 0x4
    purple_tile = 0x8
    abyss_hole = 0x10
    bridge = 0x20
    trap = 0x40
    abyss_bridge = bridge | abyss_hole
    abyss_trap = trap | abyss_hole
    occupied = 0x80

    @staticmethod
    def contains(value: int, flag: int):
        return value & flag == flag

    @staticmethod
    def matches(value: int, flag: int):
        return value == flag

    @staticmethod
    def remap(v: int):
        p1_mask = tile_type.occupied if (v > 9) else 0
        match (v % 10, p1_mask):
            case (0, _):
                return p1_mask | tile_type.goal_tile
            case (1, _):
                return p1_mask | tile_type.white_tile
            case (2, _):
                return p1_mask | tile_type.green_tile
            case (3, _):
                return p1_mask | tile_type.purple_tile
            case (4, 0):
                return tile_type.abyss_hole
            case (5, 0):
                return tile_type.abyss_bridge
            case (6, 0):
                return tile_type.abyss_trap
            case _:
                raise ValueError(f"Invalid tile type: {v}")

    @staticmethod
    def to_string(v: int, s: int):
        pad = " " * (s - 1)
        tt = tile_type
        match v:
            case a if tt.matches(a, tt.abyss_hole):
                return pad + "█"
            case a if tt.matches(a, tt.abyss_bridge):
                return pad + "■"
            case a if tt.matches(a, tt.abyss_trap):
                return pad + "▓"
            case a if tt.contains(a, tt.occupied):
                return pad + "▲"
            case a if tt.contains(a, tt.goal_tile):
                return pad + "┼"
            case a if tt.contains(a, tt.green_tile):
                return pad + "·"
            case a if tt.contains(a, tt.purple_tile):
                return pad + "·"
            case _:
                return pad + "·"


def get_location(state: np_state_type):
    result = np.argwhere(state & tile_type.occupied)
    assert result.shape == (1, 2), f"Location should be exactly 1: {result.shape}"
    r: int = result[0, 0]
    c: int = result[0, 1]
    return (r, c)


def find_tile(state: np_state_type, cb: tile_search_cb):
    result = np.argwhere(cb(state))
    assert result.shape == (1, 2), f"Location should be exactly 1: {result.shape}"
    r: int = result[0, 0]
    c: int = result[0, 1]
    return (r, c)


def goal_test(state: np_state_type):
    loc = get_location(state)
    return tile_type.contains(state[loc], tile_type.goal_tile)


def try_move(
    state: np_state_type,
    position: tuple[int, int],
    delta: tuple[int, int],
    policy: jump_policy_cb,
):
    tt = tile_type
    rows, cols = state.shape
    y1, x1 = position
    yd, xd = delta
    y2 = y1 + yd
    x2 = x1 + xd
    if not (0 <= y2 < rows and 0 <= x2 < cols):
        return None
    match state[y1, x1], state[y2, x2]:
        case (a, b) if tt.contains(b, tt.abyss_hole):
            if tt.contains(b, tt.abyss_trap):
                return None

            y3 = y2 + yd
            x3 = x2 + xd
            if not (0 <= y3 < rows and 0 <= x3 < cols):
                return None
            c = state[y3, x3]
            if tt.contains(c, tt.abyss_hole):
                return None

            n_jump = 0
            if tt.contains(a, tt.green_tile):
                n_jump += 1
            if tt.contains(b, tt.abyss_bridge):
                n_jump += 1
            if tt.contains(c, tt.purple_tile):
                n_jump += 1

            if policy(n_jump):
                st_next = state.copy()
                st_next[y1, x1] = state[y1, x1] & ~tt.occupied
                st_next[y3, x3] = state[y3, x3] | tt.occupied
                return st_next
            else:
                return None

        case (a, b) if not tt.contains(b, tt.abyss_hole):
            st_next = state.copy()
            st_next[y1, x1] = state[y1, x1] & ~tt.occupied
            st_next[y2, x2] = state[y2, x2] | tt.occupied
            return st_next

        case _:
            raise RuntimeError("Impossible state transition")


def gen_successor_cb(policy: jump_policy_cb):
    def next_states(state: npt.NDArray[np.uint8]):
        loc = get_location(state)
        neighbors: list[np_state_type] = []
        for delta in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            st_next = try_move(state, loc, delta, policy)
            if st_next is not None:
                neighbors.append(st_next)

        return tuple(neighbors)

    return next_states


def heuristic(state: np_state_type) -> mixed_item_type:
    y1, x1 = get_location(state)
    y2, x2 = find_tile(state, lambda x: x & tile_type.goal_tile)
    x3 = abs(x2 - x1)
    y3 = abs(y2 - y1)
    return y3 + x3


def solve_maze(jump_policy: int, anim: bool):
    puzzle_path = serve_input("./puzzles", "weird_maze_2")
    if puzzle_path is None:
        raise ValueError(f"File selection was canceled")

    maze_raw: list[list[int]] = []
    with open(puzzle_path) as file:
        line = file.readline()
        while line:
            maze_raw.append([int(s) for s in line.split(",")])
            line = file.readline()

    maze: npt.NDArray[np.uint8] = np.array(
        [[tile_type.remap(c) for c in r] for r in maze_raw]

    path, node_generated, node_expanded = find_paths(
        maze,
        goal_test,
        gen_successor_cb(lambda x: x == jump_policy),
        heuristic,
    )

    stats = {
        "nodes generated": node_generated,
        "nodes expanded": node_expanded,
        "path length": len(path) - 1 if path else None,
    }

    if path is None:
        print("[yellow]no solution found[/yellow]")
        print(stats)
        return

    print(prettyState(path[0], tile_type.to_string))

    if anim:
        animate_board(path, tile_type.to_string)
    print(stats)
