import curses
import numpy as np
import time

from a_star.typing import (
    np_item_type,
    np_state_type,
)
from weird_maze_2.typing import (
    state_cb_to_text,
    list_of_states,
)


# stringify a state
def prettyState(state: np_state_type, state_parser: state_cb_to_text):
    span = 2
    lines = []
    row = state.shape[0]
    col = state.shape[1]
    lines.append("")
    lines.append(" " + "▁" * col * span)
    for i in range(row):
        lines.append("")
        lines[-1] += "▕"
        for j in range(col):
            lines[-1] += state_parser(state[i, j], span)
        lines[-1] += "▏"
    lines.append(" " + "▔" * col * span)
    return "\n".join(lines)


def curses_callback(
    stdscr: curses.window, states: list_of_states, state_parser: state_cb_to_text
):
    curses.curs_set(0)
    if states is None:
        raise ValueError("non solvable game")

    stdscr.clear()
    text = prettyState(states[0], state_parser)
    text += "\niter: 0"
    stdscr.addstr(0, 0, text)
    stdscr.refresh()
    time.sleep(1)

    for i, state in enumerate(states):
        stdscr.clear()
        text = prettyState(state, state_parser)
        text += f"\niter: {i + 1}"
        stdscr.addstr(0, 0, text)
        stdscr.refresh()
        time.sleep(0.4)
    stdscr.addstr(0, 0, "Press any key to end")
    stdscr.getch()


def animate_board(state: list_of_states, state_parser: state_cb_to_text):
    curses.wrapper(curses_callback, state, state_parser)
