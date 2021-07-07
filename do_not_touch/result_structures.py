# Pi(s,a)
from dataclasses import dataclass
from typing import Dict

Policy = Dict[int, Dict[int, float]]

# V(s)
ValueFunction = Dict[int, float]

# Q(s,a)
ActionValueFunction = Dict[int, Dict[int, float]]


# Pi(s,a) and V(s)
@dataclass
class PolicyAndValueFunction:
    pi: Policy
    v: ValueFunction


# Pi(s,a) and Q(s,a)
@dataclass
class PolicyAndActionValueFunction:
    pi: Policy
    q: ActionValueFunction
