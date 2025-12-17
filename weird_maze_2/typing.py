from a_star.typing import np_state_type
from typing import (
    Callable,
    TypeAlias,
)

agent_cb_locator: TypeAlias = Callable[[np_state_type], tuple[int, int]]
state_cb_to_text: TypeAlias = Callable[[int, int], str]
list_of_states: TypeAlias = list[np_state_type] | None
