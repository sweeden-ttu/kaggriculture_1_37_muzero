from typing import Any, Callable, Dict, List, Optional, Tuple, Union
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
