import numpy as np
import numpy.typing as npt

from a_star.algorithms import (
    find_paths,
)
from a_star.typing import (
    mixed_item_type,
    np_item_type,
    np_state_type,
    goal_cb_type,
    next_cb_type,
    heur_cb_type,
)
from rich import print
from weird_maze_2.typing import state_cb_to_text
from weird_maze_2.visualizer import animate_board


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
    def jump_policy(n: int):
        return True

    @staticmethod
    def contains(value: int, flag: int):
        return value & flag == flag

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
            case a if tt.contains(a, tt.abyss_hole):
                return pad + "█"
            case a if tt.contains(a, tt.occupied):
                return pad + "▲"
            case _:
                return pad + "░"


def get_location(state: np_state_type):
    result = np.argwhere(state & tile_type.occupied)
    assert result.shape == (1, 2), f"Location should be exactly 1: {result.shape}"
    r: int = result[0, 0]
    c: int = result[0, 1]
    return (r, c)


def goal_test(state: np_state_type):
    loc = get_location(state)
    return tile_type.contains(state[loc], tile_type.goal_tile)


def try_move(state: np_state_type, position: tuple[int, int], delta: tuple[int, int]):
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

            n_jump = 0
            if tt.contains(a, tt.green_tile):
                n_jump += 1
            if tt.contains(b, tt.abyss_bridge):
                n_jump += 1
            if tt.contains(c, tt.purple_tile):
                n_jump += 1

            if tt.jump_policy(n_jump):
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


def next_states(state: npt.NDArray[np.uint8]):
    loc = get_location(state)
    neighbors: list[np_state_type] = []
    for delta in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
        st_next = try_move(state, loc, delta)
        if st_next is not None:
            neighbors.append(st_next)

    return tuple(neighbors)


def solve_maze():
    maze_raw = [
        [1, 1, 1, 1, 1, 2, 2, 1, 1, 2, 5, 3, 1, 1, 2],
        [1, 11, 1, 4, 6, 5, 4, 5, 6, 5, 4, 6, 4, 1, 1],
        [3, 1, 1, 5, 2, 1, 3, 4, 3, 2, 1, 2, 5, 3, 1],
        [4, 6, 1, 4, 4, 6, 4, 4, 5, 4, 6, 1, 4, 1, 1],
        [1, 1, 2, 5, 3, 4, 1, 1, 2, 3, 4, 2, 4, 1, 2],
        [2, 1, 1, 4, 2, 4, 3, 1, 1, 1, 4, 6, 4, 4, 5],
        [5, 6, 6, 4, 4, 4, 4, 4, 4, 5, 4, 1, 1, 1, 3],
        [3, 1, 3, 5, 2, 1, 2, 3, 5, 3, 5, 3, 1, 1, 1],
        [1, 1, 1, 4, 4, 4, 4, 1, 4, 4, 4, 1, 1, 1, 2],
        [1, 1, 1, 1, 3, 4, 3, 2, 5, 1, 4, 6, 4, 4, 5],
        [3, 1, 1, 2, 1, 4, 4, 6, 4, 1, 2, 1, 1, 1, 3],
        [4, 5, 6, 5, 6, 4, 2, 2, 4, 4, 5, 6, 4, 6, 5],
        [1, 1, 5, 3, 1, 1, 1, 1, 6, 2, 3, 6, 3, 1, 3],
        [2, 1, 6, 4, 5, 4, 1, 2, 4, 1, 1, 4, 1, 0, 1],
        [3, 1, 1, 1, 3, 5, 2, 1, 6, 3, 2, 5, 3, 1, 1],
    ]
    maze: npt.NDArray[np.uint8] = np.array(
        [[tile_type.remap(c) for c in r] for r in maze_raw]
    )
    path, node_generated, node_expanded = find_paths(
        maze, goal_test, next_states, lambda x: 0
    )

    animate_board(path, tile_type.to_string)

    print(
        {
            "nodes generated": node_generated,
            "nodes expanded": node_expanded,
        }
    )
