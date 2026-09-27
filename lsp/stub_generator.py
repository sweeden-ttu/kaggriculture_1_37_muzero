"""Dynamic stub generator for kaggle_environments and kaggriculture.

Generates PEP 561 / PEP 484 type stubs in typings/kaggle_environments
by inspecting the live runtime objects and specification schemas.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List

import kaggle_environments
import kaggle_environments.agent
import kaggle_environments.core
import kaggle_environments.errors
import kaggle_environments.utils

try:
    import kaggle_environments.envs.kaggriculture.kaggriculture as kagg_module
except Exception:
    kagg_module = None


def generate_kaggle_environments_stubs(target_dir: str | Path | None = None) -> List[Path]:
    """Generate comprehensive .pyi stubs for kaggle_environments and kaggriculture."""
    if target_dir is None:
        target_dir = Path(__file__).resolve().parent.parent / "typings" / "kaggle_environments"
    else:
        target_dir = Path(target_dir)

    target_dir.mkdir(parents=True, exist_ok=True)
    generated_files: List[Path] = []

    # 1. py.typed (PEP 561 marker)
    py_typed = target_dir / "py.typed"
    py_typed.write_text("# Marker file for PEP 561 compliance.\n", encoding="utf-8")
    generated_files.append(py_typed)

    # 2. utils.pyi
    utils_pyi = target_dir / "utils.pyi"
    utils_pyi.write_text(
        '''from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

class Struct(dict[str, Any]):
    """Dynamic dict supporting attribute-style dot notation (e.g., struct.reward)."""
    action: Any
    info: Any
    observation: Struct | Dict[str, Any]
    reward: Optional[float]
    status: str
    seed: Optional[int]

    def __init__(self, **entries: Any) -> None: ...
    def __getattr__(self, attr: str) -> Any: ...
    def __setattr__(self, attr: str, value: Any) -> None: ...
    def __getitem__(self, key: str) -> Any: ...
    def __setitem__(self, key: str, value: Any) -> None: ...
    def __contains__(self, key: object) -> bool: ...
    def get(self, key: str, default: Any = None) -> Any: ...

def structify(obj: Any) -> Any: ...
def unstructify(obj: Any) -> Any: ...

envs_path: str
schemas: Dict[str, Any]
''',
        encoding="utf-8",
    )
    generated_files.append(utils_pyi)

    # 3. errors.pyi
    errors_pyi = target_dir / "errors.pyi"
    errors_pyi.write_text(
        '''class EnvironmentError(Exception): ...
class DeadlineExceeded(EnvironmentError): ...
class FailedPrecondition(EnvironmentError): ...
class InvalidArgument(EnvironmentError): ...
''',
        encoding="utf-8",
    )
    generated_files.append(errors_pyi)

    # 4. agent.pyi
    agent_pyi = target_dir / "agent.pyi"
    agent_pyi.write_text(
        '''from typing import Any, Callable, Dict, Optional

class Agent:
    raw: Any
    environment: Any
    name: str

    def __init__(self, raw: Any, environment: Any = None) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def act(self, observation: Dict[str, Any], configuration: Optional[Dict[str, Any]] = None) -> Any: ...
''',
        encoding="utf-8",
    )
    generated_files.append(agent_pyi)

    # 5. core.pyi
    core_pyi = target_dir / "core.pyi"
    core_pyi.write_text(
        '''from typing import Any, Callable, Dict, List, Optional, Tuple, Union
from .agent import Agent
from .utils import Struct

class Environment:
    name: str
    id: str
    specification: Struct
    configuration: Struct
    info: Struct
    steps: List[List[Struct]]
    state: List[Struct]
    logs: List[List[Dict[str, Any]]]
    debug: bool
    agents: Dict[str, Callable[..., Any]]
    done: bool

    def __init__(
        self,
        specification: Optional[Dict[str, Any]] = None,
        configuration: Optional[Dict[str, Any]] = None,
        info: Optional[Dict[str, Any]] = None,
        steps: Optional[List[List[Dict[str, Any]]]] = None,
        logs: Optional[List[List[Dict[str, Any]]]] = None,
        agents: Optional[Dict[str, Callable[..., Any]]] = None,
        interpreter: Optional[Callable[..., Any]] = None,
        renderer: Optional[Callable[..., Any]] = None,
        html_renderer: Optional[Callable[..., Any]] = None,
        debug: bool = False,
        state: Optional[Any] = None,
    ) -> None: ...

    def reset(self, num_agents: Optional[int] = None) -> List[Struct]: ...
    def step(self, actions: List[Any], logs: Optional[List[Dict[str, Any]]] = None) -> List[Struct]: ...
    def run(self, agents: List[Union[str, Callable[..., Any], Agent]]) -> List[List[Struct]]: ...
    def render(self, mode: str = "human", **kwargs: Any) -> Optional[str]: ...
    def play(self, agents: Optional[List[Union[str, Callable[..., Any], Agent]]] = None, **kwargs: Any) -> None: ...
    def train(self, agents: Optional[List[Union[str, Callable[..., Any], Agent]]] = None) -> Any: ...
    def clone(self) -> Environment: ...
    def toJSON(self) -> Dict[str, Any]: ...

def make(
    environment: Union[str, Environment, Callable[..., Any]],
    configuration: Optional[Dict[str, Any]] = None,
    info: Optional[Dict[str, Any]] = None,
    steps: Optional[List[List[Dict[str, Any]]]] = None,
    logs: Optional[List[List[Dict[str, Any]]]] = None,
    debug: bool = False,
    state: Optional[List[Dict[str, Any]]] = None,
) -> Environment: ...

def evaluate(
    environment: Union[str, Environment],
    agents: Optional[List[Union[str, Callable[..., Any], Agent]]] = None,
    configuration: Optional[Dict[str, Any]] = None,
    steps: Optional[List[List[Dict[str, Any]]]] = None,
    num_episodes: int = 1,
    debug: bool = False,
    state: Optional[List[Dict[str, Any]]] = None,
) -> List[List[Optional[float]]]: ...

def register(name: str, environment: Dict[str, Any]) -> None: ...
def register_lazy(name: str, loader: Callable[[], None]) -> None: ...

environments: Dict[str, Dict[str, Any]]
''',
        encoding="utf-8",
    )
    generated_files.append(core_pyi)

    # 6. __init__.pyi
    init_pyi = target_dir / "__init__.pyi"
    init_pyi.write_text(
        '''from typing import Any, Callable, Dict, List, Optional, Union
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
''',
        encoding="utf-8",
    )
    generated_files.append(init_pyi)

    # 7. Subpackage stubs: envs and kaggriculture
    envs_dir = target_dir / "envs"
    envs_dir.mkdir(parents=True, exist_ok=True)
    envs_init_pyi = envs_dir / "__init__.pyi"
    envs_init_pyi.write_text("# Stubs for kaggle_environments.envs\n", encoding="utf-8")
    generated_files.append(envs_init_pyi)

    kagg_dir = envs_dir / "kaggriculture"
    kagg_dir.mkdir(parents=True, exist_ok=True)
    kagg_init_pyi = kagg_dir / "__init__.pyi"
    kagg_init_pyi.write_text("from .kaggriculture import *\n", encoding="utf-8")
    generated_files.append(kagg_init_pyi)

    # 8. kaggriculture.pyi
    kagg_pyi = kagg_dir / "kaggriculture.pyi"
    kagg_pyi.write_text(
        '''from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

ANIMALS: Dict[str, Dict[str, Any]]
CROPS: Dict[str, Dict[str, Any]]
PRODUCTS: List[str]
SHOPS: Dict[str, List[str]]
LAND_PRICES: List[int]
LAND_ORDER: List[str]
FARMER_MOVES: List[str]
FARM_HAND_COST_MULT: int
HINGE_GAIN: float
MAX_SHOP_INSTANCES: int
PRICE_FLOOR: float
MARKET_I0: int
MARKET_PARAMS: Dict[str, Dict[str, Any]]
TOWN_CENTER_PRODUCTS: List[str]

agents: Dict[str, Callable[..., Any]]
html_renderer: Optional[Callable[..., Any]]
interpreter: Callable[..., Any]
renderer: Callable[..., Any]
specification: Dict[str, Any]

class KaggricultureObservation(TypedDict, total=False):
    player: int
    step: int
    remainingOverageTime: float
    farms: List[Dict[str, Any]]
    private: Dict[str, Any]
    market: Dict[str, Dict[str, float]]
    town: Dict[str, Any]
    day: int
    hour: int

class KaggricultureConfiguration(TypedDict, total=False):
    episodeSteps: int
    actTimeout: float
    boardSize: int
    startingMoney: int
    maxMarketOrdersPerTurn: int
    turnsPerDay: int
    shedCapacity: int
    weedSpawnChance: float
    townShopUnlockInterval: int
    townShopSellInterval: int
    townCenterSellInterval: int
    seed: Optional[int]
    farmHandCostMult: int
    marketParams: Dict[str, Any]

class KaggricultureAction(TypedDict, total=False):
    farmer: List[Any]
    hands: List[List[Any]]
    market: List[List[Any]]
''',
        encoding="utf-8",
    )
    generated_files.append(kagg_pyi)

    return generated_files


if __name__ == "__main__":
    files = generate_kaggle_environments_stubs()
    print(f"Generated {len(files)} stub files in {files[0].parent}")
