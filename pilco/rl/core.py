"""Core RL functions and classes."""
import typing

__all__ = ["StepInfo"]


class StepInfo(typing.NamedTuple):
    """Information about one step in an environment."""

    observation: typing.Any
    action: typing.Any
    next_observation: typing.Any
    reward: float
    done: bool
    info: typing.Any = None
    state: typing.Any = None
    next_state: typing.Any = None
