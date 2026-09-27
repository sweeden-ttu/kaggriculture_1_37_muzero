from typing import Any, Callable, Dict, List, Optional, Union
from . import errors, utils
from .agent import Agent
from .core import Environment, environments, evaluate, make, register, register_lazy

__version__: str

__all__ = [
    "Agent",
    "Environment",
    "environments",
    "errors",
    "evaluate",
    "make",
    "register",
    "register_lazy",
    "utils",
    "__version__",
    "get_episode_replay",
    "list_episodes",
    "list_episodes_for_team",
    "list_episodes_for_submission",
]

def get_episode_replay(episode_id: int) -> Dict[str, Any]: ...
def list_episodes(team_id: Optional[int] = None, submission_id: Optional[int] = None) -> List[Dict[str, Any]]: ...
def list_episodes_for_team(team_id: int) -> List[Dict[str, Any]]: ...
def list_episodes_for_submission(submission_id: int) -> List[Dict[str, Any]]: ...
