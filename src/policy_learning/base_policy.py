import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Optional, Union


class BasePolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def train(self) -> None:
        raise NotImplementedError

    def eval(self) -> None:
        raise NotImplementedError

    def select_action(
        self, obs: Union[np.ndarray, torch.Tensor], deterministic: bool = False
    ) -> np.ndarray:
        raise NotImplementedError

    def learn(self, batch: Dict, preference_batch: Optional[Dict]) -> Dict[str, float]:
        raise NotImplementedError
