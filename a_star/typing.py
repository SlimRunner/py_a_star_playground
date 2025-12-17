import numpy.typing as npt
import numpy as np

from typing import (
    Callable,
    Dict,
    TypeAlias,
)


payload_type: TypeAlias = int | float
# np_item_type: TypeAlias = np.int_ | np.float64 | np.uint8
np_item_type: TypeAlias = np.uint8
mixed_item_type: TypeAlias = payload_type | np_item_type
np_state_type: TypeAlias = npt.NDArray[np_item_type]
goal_cb_type: TypeAlias = Callable[[np_state_type], bool]
next_cb_type: TypeAlias = Callable[[np_state_type], tuple[np_state_type, ...]]
heur_cb_type: TypeAlias = Callable[[np_state_type], mixed_item_type]
explore_map_type: TypeAlias = Dict[tuple[np_item_type, ...], mixed_item_type]
