from typing import Any, Callable, Dict, Optional

class Agent:
    raw: Any
    environment: Any
    name: str

    def __init__(self, raw: Any, environment: Any = None) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...
    def act(self, observation: Dict[str, Any], configuration: Optional[Dict[str, Any]] = None) -> Any: ...
