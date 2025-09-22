# Dataclass model definitions for SSP models and scaling parameters
from dataclasses import dataclass

@dataclass
class ModelParams:
    s: float = 0.3
    alpha: float = 0.3
    delta: float = 0.1
    tas0: float = 0.0
    pr0: float = 0.0
    k_tas1: float = 0.0
    k_tas2: float = 0.0
    k_pr1: float = 0.0
    k_pr2: float = 0.0
    tfp_tas1: float = 0.0
    tfp_tas2: float = 0.0
    tfp_pr1: float = 0.0
    tfp_pr2: float = 0.0
    y_tas1: float = 0.0
    y_tas2: float = 0.0
    y_pr1: float = 0.0
    y_pr2: float = 0.0

@dataclass
class ScalingParams:
    scaling_name: str = ""
    scale_factor: float = 1.0
    k_tas1: float = 0.0
    k_tas2: float = 0.0
    k_pr1: float = 0.0
    k_pr2: float = 0.0
    tfp_tas1: float = 0.0
    tfp_tas2: float = 0.0
    tfp_pr1: float = 0.0
    tfp_pr2: float = 0.0
    y_tas1: float = 0.0
    y_tas2: float = 0.0
    y_pr1: float = 0.0
    y_pr2: float = 0.0
