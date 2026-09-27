from typing import Any, Callable, Dict, List, Optional, Tuple, TypedDict

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
