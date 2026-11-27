import numpy as np
import numpy.typing as npt
from utils.file_ops import serve_input
from utils.shared import select_heuristic, read_int_matrix
from rich import print
from utils.visualizers import animate_board, prettyState


from a_star.algorithms import (
    find_paths,
)
from a_star.typing import (
    heur_cb_type,
    np_state_type,
    mixed_item_type,
)


class tile_type:
    blank_tile = 0x1
    wall = 0x2
    box = 0x4
    star_tile = 0x8
    boxstar = box | star_tile
    keeper = 0x10
    keeperstar = keeper | star_tile

    @staticmethod
    def is_free_star(value: int):
        match (value & ~tile_type.star_tile):
            case 0 | tile_type.keeper:
                return True
            case _:
                return False

    @staticmethod
    def contains(value: int, flag: int):
        return value & flag == flag

    @staticmethod
    def matches(value: int, flag: int):
        return value == flag

    @staticmethod
    def remap(v: int):
        match (v % 10):
            case 0:
                return tile_type.blank_tile
            case 1:
                return tile_type.wall
            case 2:
                return tile_type.box
            case 3:
                return tile_type.keeper
            case 4:
                return tile_type.star_tile
            case 5:
                return tile_type.boxstar
            case 6:
                return tile_type.keeperstar
            case _:
                raise ValueError(f"Invalid tile type: {v}")

    @staticmethod
    def to_string(v: int, s: int):
        pad = " " * (s - 1)
        tt = tile_type
        match v:
            case a if tt.matches(a, tt.blank_tile):
                return pad + "·"
            case a if tt.matches(a, tt.wall):
                return pad + "▓"
            case a if tt.matches(a, tt.box):
                return pad + "□"
            case a if tt.matches(a, tt.keeper):
                return pad + "▲"
            case a if tt.matches(a, tt.star_tile):
                return pad + "◇"
            case a if tt.matches(a, tt.boxstar):
                return pad + "■"
            case a if tt.matches(a, tt.keeperstar):
                return pad + "◬"
            case _:
                return pad + "·"


def getFreq(s: np.ndarray):
    flat = s.ravel()
    counts = np.bincount(flat)
    return {i: counts[i] for i in range(len(counts)) if counts[i] > 0}


def goal_test(s: np_state_type):
    assert type(s) == np.ndarray
    countmap = getFreq(s)

    # spec gurantees there will be at least one box and one goal and
    # that boxes <= goals. Therefore this test is exhaustive.
    hasNoBoxes = countmap.get(tile_type.box, 0) == 0

    return hasNoBoxes


def getKeeperPosition(s: np_state_type):
    row = s.shape[0]
    col = s.shape[1]
    for i in range(row):
        for j in range(col):
            if tile_type.contains(s[i, j], tile_type.keeper):
                return i, j


def tsum(a: tuple[int, int], b: tuple[int, int]):
    return a[0] + b[0], a[1] + b[1]


def in_range(s: np_state_type, pos: tuple[int, int]):
    rows, cols = s.shape
    return 0 <= pos[0] < rows and 0 <= pos[1] < cols


def promote_pairs(src: int, dst: int):
    # logic map of valid moves from source to destination
    tt = tile_type
    mappings = {
        tt.keeper: {
            tt.blank_tile: (tt.blank_tile, tt.keeper),
            tt.star_tile: (tt.blank_tile, tt.keeperstar),
        },
        tt.keeperstar: {
            tt.blank_tile: (tt.star_tile, tt.keeper),
            tt.star_tile: (tt.star_tile, tt.keeperstar),
        },
    }
    # use logic map to reduce move dependencies
    srcMap = mappings.get(src, None)
    if srcMap is None:
        return None
    dstMap = srcMap.get(dst, None)
    if dstMap is None:
        return None
    return dstMap


def promote_triads(src: int, mid: int, dst: int):
    # logic map of valid pushes from source to box to destination
    tt = tile_type
    mappings = {
        tt.keeper: {
            tt.box: {
                tt.blank_tile: (tt.blank_tile, tt.keeper, tt.box),
                tt.star_tile: (tt.blank_tile, tt.keeper, tt.boxstar),
            },
            tt.boxstar: {
                tt.blank_tile: (tt.blank_tile, tt.keeperstar, tt.box),
                tt.star_tile: (tt.blank_tile, tt.keeperstar, tt.boxstar),
            },
        },
        tt.keeperstar: {
            tt.box: {
                tt.blank_tile: (tt.star_tile, tt.keeper, tt.box),
                tt.star_tile: (tt.star_tile, tt.keeper, tt.boxstar),
            },
            tt.boxstar: {
                tt.blank_tile: (tt.star_tile, tt.keeperstar, tt.box),
                tt.star_tile: (tt.star_tile, tt.keeperstar, tt.boxstar),
            },
        },
    }
    # use logic map to reduce move dependencies
    srcMap = mappings.get(src, None)
    if srcMap is None:
        return None
    midMap = srcMap.get(mid, None)
    if midMap is None:
        return None
    dstMap = midMap.get(dst, None)
    if dstMap is None:
        return None
    return dstMap


def try_walk(s: np_state_type, here: tuple[int, int], move: tuple[int, int]):
    sn = s.copy()

    if here is None:
        return None

    # add move to here to get new coordinate
    pNext = tsum(here, move)

    if not in_range(sn, pNext):
        return None

    # if keeper can move get new values, None otherwise.
    next_pairs = promote_pairs(sn[here], sn[pNext])
    if next_pairs is not None:
        # apply change
        sn[here], sn[pNext] = next_pairs
        return sn

    if tile_type.contains(sn[pNext], tile_type.box):
        pAhead = tsum(pNext, move)
        if not in_range(sn, pAhead):
            return None

        # if keeper can push get new values, None otherwise
        next_triads = promote_triads(sn[here], sn[pNext], sn[pAhead])

        if next_triads is None:
            return None

        # apply change
        sn[here], sn[pNext], sn[pAhead] = next_triads
        return sn

    return None


def next_states(s: np_state_type):
    # bruh, why does the sekelton mixes snake_case and camelCase?
    here = getKeeperPosition(s)
    assert here is not None

    #       (-1, 0)
    # ( 0,-1)  *  ( 0, 1)
    #       ( 1, 0)

    neighbors: list[np_state_type] = []
    for delta in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        st_next = try_walk(s, here, delta)
        if st_next is not None:
            neighbors.append(st_next)

    return tuple(neighbors)


def misplaced_boxes_heuristic(s: np_state_type) -> mixed_item_type:
    # get frequency of each distinct state as a dictionary
    countmap = getFreq(s)
    # return the frequency of box
    return countmap.get(tile_type.box, 0)


def in_range_w_dims(size: tuple[int, int], pos: tuple[int, int]):
    """
    determines if position is in bounds of numpy 2d array but using only
    its shape

    :param size: 2-tuple representing height and width of sokoban map
    :param pos: position to check bounds for
    :return: `True` if position is in-bound; `False` otherwise
    """
    return 0 <= pos[0] < size[0] and 0 <= pos[1] < size[1]


def sideWalled(s: np_state_type, here: tuple[int, int], wall: tuple[int, int]):
    """
    :param s: 2d numpy array of integers representing a sokoban state
    :param here: location of box to be probed
    :param wall: location of wall immediately next to the box
    :return: `True` if box is stranded (i.e. cannot reach a goal).
        `False` otherwise

    Below is the pseudocode of this algorithm.

    - for each misplaced box
        - check each cardinal spot around; store it if it is a wall
        - for each wall found
            - get a slice of the entire map that is tangent to the box
            - this slice is two tiles wide and includes the box and the
              wall
            - respectively refered to as box-aisle and wall-aisle
            - run DFS on the slice (1D DFS) where the seed is the
              box-wall pair
            - mark seed as visited push to stack
            - do until stack is empty
                - pop stack into
                - return `False` if you find a non-wall in the
                  wall-asile
                - continue without expanding node if you find a wall in
                  the box-aisle
                - return `False` if you find a goal in the box-aisle
                - otherwise expand node if next tile is not visited and
                  in-bounds
            - return `True` because the wall-aisle is solid wall and
              box-aisle has no goals hence the game can no longer be
              won.
    """
    size = s.shape
    deltaRow = wall[0] - here[0]
    deltaCol = wall[1] - here[1]
    tt = tile_type

    match deltaRow, deltaCol:
        case 0, 0:
            # if this happens this is a caller error
            assert False
        case 0, c:
            # aisle is vertical which means wall is left or right

            wallAisle = () if in_range_w_dims(size, wall) else None
            stack = [here]
            visited = {here}

            while stack:
                node = stack.pop()
                boxAisle: int = s[node]

                # None means wall-aisle is out of bounds
                if wallAisle is not None:
                    wallAisle = s[node[0], node[1] + c]

                # this if is not elif on-purpose
                if tt.is_free_star(boxAisle):
                    return False

                elif tt.matches(boxAisle, tt.wall):
                    continue

                # this enforces that out-of-bounds is solid wall
                elif wallAisle is not None and not tt.matches(wallAisle, tt.wall):
                    return False

                else:
                    nextNode = tsum(node, (1, 0))
                    if in_range_w_dims(size, nextNode) and nextNode not in visited:
                        visited.add(nextNode)
                        stack.append(nextNode)
                    nextNode = tsum(node, (-1, 0))
                    if in_range_w_dims(size, nextNode) and nextNode not in visited:
                        visited.add(nextNode)
                        stack.append(nextNode)

            return True
        case r, 0:
            # aisle is horizontal which means wall is above or below

            wallAisle = () if in_range_w_dims(size, wall) else None
            stack = [here]
            visited = {here}

            while stack:
                node = stack.pop()
                boxAisle = s[node]

                # None means wall-aisle is out of bounds
                if wallAisle is not None:
                    wallAisle = s[node[0] + r, node[1]]

                # this if is not elif on-purpose
                if tt.is_free_star(boxAisle):
                    return False
                elif tt.matches(boxAisle, tt.wall):
                    continue

                # this enforces that out-of-bounds is solid wall
                elif wallAisle is not None and not tt.matches(wallAisle, tt.wall):
                    return False
                else:
                    nextNode = tsum(node, (0, 1))
                    if in_range_w_dims(size, nextNode) and nextNode not in visited:
                        visited.add(nextNode)
                        stack.append(nextNode)
                    nextNode = tsum(node, (0, -1))
                    if in_range_w_dims(size, nextNode) and nextNode not in visited:
                        visited.add(nextNode)
                        stack.append(nextNode)

            return True
        case _:
            # This is a catch all for user error
            return False

    assert False


def isTrapped(s: np_state_type, here: tuple[int, int]):
    """
    :param s: 2d numpy array of integers representing a sokoban state
    :param here: location of box to be probed
    :return: `True` if box is trapped (i.e. cannot reach a goal).
             `False` otherwise

    This functions perform three different tests

    ### Check for L-traps and T-traps
    These patterns look like these (and their symmetries)
    ```
    . . .        . . .
    x B .   OR   x B x
    . x .        . x .
    ```
    Where `B` is a misplaced box, `.` an empty space, and `x` a wall.
    Since walls are immovable, we do not need to verify dependencies, so
    if this pattern is found the game can no longer be won.

    ### Check for 2x2 clusters
    These patterns look like this
    ```
    . . .
    x B .
    x x .
    ```
    Here we can safely assume `x` are either any type of boxes or walls
    because a 2x2 cluster of boxes causes each box in the cluster to
    become stranded no matter the context. Note that this will not
    mistakenly flag four boxes on goals because this function is only
    called with a misplaced box as center of the 3x3. There is no way to
    push two boxes. Hence, the game can no longer be won.

    ### Check for Sidewalled Boxes
    Boxes that are flush against a wall such that it cannot leave the
    aisle next to the wall and there are no goals in that aisle; thus,
    the box is stranded, and the game can no longer be won.

    Example
    ```
    . . . . .
    . . . . .
    . . * . B
    . . . . .
    . . . . .
    ```
    The box `B` cannot reach star `*`. for extended description read
    `sideWalled` docstring.
    """

    # Map of coordinates
    # (-1,-1) (-1, 0) (-1, 1)
    # ( 0,-1)    *    ( 0, 1)
    # ( 1,-1) ( 1, 0) ( 1, 1)

    # Map of bitfield values
    # 016 001 064
    # 004  *  008
    # 128 002 032

    # List of cases. Note wall checks are the first items in the tuple
    # so they are zeroed out for the corners.
    nbors = [
        (1, 1, tsum(here, (-1, 0))),
        (2, 2, tsum(here, (1, 0))),
        (4, 4, tsum(here, (0, -1))),
        (8, 8, tsum(here, (0, 1))),
        (0, 16, tsum(here, (-1, -1))),
        (0, 32, tsum(here, (1, 1))),
        (0, 64, tsum(here, (-1, 1))),
        (0, 128, tsum(here, (1, -1))),
    ]

    tt = tile_type

    # create bitfields for a 3x3 of wall and mixed obstacles
    wallObstacle = 0
    mixedObstacle = 0
    for i, j, pos in nbors:
        inBound = in_range(s, pos)
        if not inBound:
            wallObstacle += i
            mixedObstacle += j
            continue
        _isnotwall = not tt.matches(s[pos], tt.wall)
        _isbox = tt.contains(s[pos], tt.box)
        wallObstacle += 0 if _isnotwall else i
        mixedObstacle += 0 if _isnotwall and not _isbox else j

    # using the above values we can hard code the list of L-traps and
    # T-traps. This check considers only walls
    # S = {5, 6, 10, 9, 7, 14, 11, 13, 15}
    # if value of current 3x3 is in S
    #   then we have a box-trap
    blocked_by_wall = wallObstacle in {5, 6, 10, 9, 7, 14, 11, 13, 15}
    if blocked_by_wall:
        return True

    # using the above values we can use the following numbers and
    # boolean expresion to match 2x2 clusters. This checks considers
    # misplaced boxes or walls.
    # S = {21, 42, 73, 134} <- all possible 2x2 clusters in a 3x3
    # if any s in S where value of current 3x3 & s == s
    #   then we have a cluster-trap
    if (
        (21 & mixedObstacle) == 21
        or (42 & mixedObstacle) == 42
        or (73 & mixedObstacle) == 73
        or (134 & mixedObstacle) == 134
    ):
        return True

    # check if sidewalled in the four cardinal directions. If sidewalled
    # we have a stranded box.
    if (
        wallObstacle & 1
        and sideWalled(s, here, nbors[0][2])
        or wallObstacle & 2
        and sideWalled(s, here, nbors[1][2])
        or wallObstacle & 4
        and sideWalled(s, here, nbors[2][2])
        or wallObstacle & 8
        and sideWalled(s, here, nbors[3][2])
    ):
        return True

    return False


def manDist(a: tuple[int, int], b: tuple[int, int]):
    """
    Computes the standard Manhattan distance where `a` and `b` are
    positions in a grid.

    :param a: 2-tuple of source position
    :param b: 2-tuple of target position
    :return: integer representing Manhattan distance
    """
    dx = a[0] - b[0]
    dy = a[1] - b[1]
    vert = abs(dx)
    horz = abs(dy)
    return vert + horz


def custom_heuristic(s):
    """
    :param s: 2d numpy array of integers representing a sokoban state
    :return: and admissible score of the game or infinity if the game is
             deemed unwinnable.

    This function performs four heuristics
    - sum of manhattan distances from each misplaced box to closest goal
    - checks if at least one box is trapped by wall-only with L-traps,
      T-traps, or cross-traps (insoluble)
    - checks if at least one box forms part of an cluster trap (2x2). To
      make it easier to implement boxes on goals are ignored for this
      check. See `isTrapped` for an extended explanation.
    - checks if a box if flush against a wall and it is optimistically
      impossible to push away from it. Meaning, there is only wall
      parallel to the box and there are no free goals next to the wall.
      It uses DFS just along the wall strip and the corridor next to it
      in all cardinal directions that have a wall in it.
    """
    rows = s.shape[0]
    cols = s.shape[1]
    boxes = []
    goals = []
    tt = tile_type

    # aggregate all goals and boxes (ignore boxes in goals). Also, count
    # boxes that are trapped
    for i in range(rows):
        for j in range(cols):
            ss = s[i, j]
            if tt.is_free_star(ss):
                goals.append((i, j))
            if tile_type.matches(ss, tile_type.box):
                # check for l-traps, t-traps, and cluster traps (2*2
                # blocks of boxes)
                if isTrapped(s, (i, j)):
                    # no point in continuing
                    return float("inf")
                boxes.append((i, j))

    distSum = 0

    # sum of shortest manhattan distances per box to closest goal.
    # Squared time complexity.
    for b in boxes:
        closest = float("inf")
        for g in goals:
            closest = min(closest, manDist(b, g))
        distSum += closest

    return distSum


def solve_sokoban(anim: bool):
    puzzle_path = serve_input("./puzzles", "sokoban")
    if puzzle_path is None:
        raise ValueError(f"File selection was canceled")

    maze_raw = read_int_matrix(puzzle_path)
    maze: np_state_type = np.array([[tile_type.remap(c) for c in r] for r in maze_raw])

    heuristic: heur_cb_type = select_heuristic(
        {
            "compound (fastest)": custom_heuristic,
            "misplaced boxes": misplaced_boxes_heuristic,
            "none": lambda x: 0,
        }
    )

    path, node_generated, node_expanded = find_paths(
        maze,
        goal_test,
        next_states,
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

    print(stats)
    if anim:
        animate_board(path, tile_type.to_string)
