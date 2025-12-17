import numpy as np  # indirect use for type hinting (don't remove)

from a_star.typing import *
from queue import PriorityQueue


class path_node:
    def __init__(
        self,
        state: np_state_type,
        parent: "path_node | None",
        cost: mixed_item_type,
        evaluation: mixed_item_type,
    ):
        """
        :param state: the current state
        :param parent: the previous node (PathNode)
        :param cost: the cost from the start state to the current state i.e. g(n)
        :param evaluation: the state value f(n) = g(n) + h(n)
        """
        self.state: tuple[np_item_type, ...] = ()
        row = state.shape[0]
        col = state.shape[1]
        for i in range(row):
            for j in range(col):
                self.state = self.state + (state[i, j],)
        self.state1 = state
        self.parent = parent
        self.cost = cost
        self.evaluation = evaluation

    def __lt__(self, other):
        if self.evaluation < other.evaluation:
            return True
        else:
            return False


def a_star_search(
    start_state: np_state_type,
    goal_test: goal_cb_type,
    next_states: next_cb_type,
    heuristic: heur_cb_type,
):
    """
    :param start_state:
    :param goal_test: a function, return true only when the input is the goal state
    :param next_states: a function, return a list of all successor states
    :param heuristic: a function, return the heuristic function value of the given state
    :return: goal node plus traversal information
    """
    pq: PriorityQueue[path_node] = PriorityQueue()
    initial_node = path_node(start_state, None, 0, heuristic(start_state))
    pq.put(initial_node)
    explored: explore_map_type = dict()

    node_generated = 1
    node_expanded = 0

    while pq.qsize() > 0:
        node = pq.get()
        if goal_test(node.state1):
            return node, node_generated, node_expanded
        old_cost = explored.get(node.state)
        if old_cost is not None and old_cost <= node.cost:
            continue
        explored[node.state] = node.cost
        all_successors = next_states(node.state1)
        node_expanded += 1
        for s in all_successors:
            new_cost = node.cost + 1
            new_node = path_node(s, node, new_cost, new_cost + heuristic(s))
            node_generated += 1
            pq.put(new_node)

    return None, node_generated, node_expanded


def find_paths(
    start_state: np_state_type,
    goal_test: goal_cb_type,
    successors: next_cb_type,
    heuristic: heur_cb_type,
):
    goal_node, node_generated, node_expanded = a_star_search(
        start_state, goal_test, successors, heuristic
    )
    if goal_node:
        node = goal_node
        path = [node.state1]
        while node.parent:
            node = node.parent
            path.append(node.state1)
        path.reverse()

        return path, node_generated, node_expanded
    else:
        return None, node_generated, node_expanded
