from abc import ABC, abstractmethod
import torch as th

from torch.distributions import Categorical
from torch import nn
from typing import Any, Dict, List, Optional, Tuple, Union

class Distribution(ABC):
    """Abstract base class for distributions."""

    def __init__(self):
        super(Distribution, self).__init__()
        self.distribution = None

    @abstractmethod
    def log_prob(self, x: th.Tensor) -> th.Tensor:
        """
        Returns the log likelihood
        :param x: the taken action
        :return: The log likelihood of the distribution
        """
class CategoricalDistribution(Distribution):
    """
    Categorical distribution for discrete actions.
    :param action_dim: Number of discrete actions
    """

    def __init__(self, action_dim: int):
        super(CategoricalDistribution, self).__init__()
        self.action_dim = action_dim

    def proba_distribution_net(self, latent_dim: int) -> nn.Module:
        """
        Create the layer that represents the distribution:
        it will be the logits of the Categorical distribution.
        You can then get probabilities using a softmax.
        :param latent_dim: Dimension of the last layer
            of the policy network (before the action layer)
        :return:
        """
        action_logits = nn.Linear(latent_dim, self.action_dim)
        return action_logits

    def proba_distribution(self, action_logits: th.Tensor) -> "CategoricalDistribution":
        self.distribution = Categorical(logits=action_logits)
        return self

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        return self.distribution.log_prob(actions)

    def entropy(self) -> th.Tensor:
        return self.distribution.entropy()


