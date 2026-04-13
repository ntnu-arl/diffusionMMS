"""Exponential Moving Average of model parameters.

Usage:
    ema = ModelEMA(model, decay=0.9999)
    for batch in loader:
        loss.backward()
        optimizer.step()
        ema.update(model)
    # At eval time:
    ema.apply(model)       # copy EMA weights into model
    evaluate(model)
    ema.restore(model)     # restore training weights
"""

import torch
from copy import deepcopy
from utils.logger import get_root_logger

logger = get_root_logger()


class ModelEMA:
    """Maintains an exponential moving average of model parameters.

    Args:
        model: The model whose parameters to track.
        decay: EMA decay factor (higher = smoother, typical 0.999-0.9999).
    """

    def __init__(self, model, decay=0.9999):
        self.decay = decay
        # Deep copy of initial parameters (detached, no grad)
        self.shadow = deepcopy(
            {k: v.detach().clone() for k, v in model.state_dict().items()}
        )
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters with current model parameters."""
        for k, v in model.state_dict().items():
            if v.is_floating_point():
                self.shadow[k].lerp_(v.detach(), 1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def apply(self, model):
        """Copy EMA weights into model (save training weights for restore)."""
        self._backup = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    def restore(self, model):
        """Restore training weights after eval."""
        if self._backup is not None:
            model.load_state_dict(self._backup)
            self._backup = None

    def state_dict(self):
        return {"decay": self.decay, "shadow": self.shadow}

    def load_state_dict(self, state):
        self.decay = state["decay"]
        self.shadow = state["shadow"]
