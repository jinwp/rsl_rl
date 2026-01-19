# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.networks import EmpiricalNormalization, MLP
from rsl_rl.utils import resolve_nn_activation


class QNetwork(nn.Module):
    """Q-network for discrete-action value estimation."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        obs_normalization: bool = False,
        hidden_dims: tuple[int] | list[int] = (256, 256),
        activation: str = "relu",
        dueling: bool = False,
        **kwargs: dict[str, object],
    ) -> None:
        if kwargs:
            print("QNetwork.__init__ got unexpected arguments, which will be ignored: " + str(list(kwargs.keys())))
        super().__init__()

        self.obs_groups = obs_groups
        num_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The QNetwork only supports 1D observations."
            num_obs += obs[obs_group].shape[-1]

        self.dueling = dueling
        if self.dueling:
            if len(hidden_dims) == 0:
                raise ValueError("Dueling QNetwork requires at least one hidden layer.")
            activation_mod = resolve_nn_activation(activation)
            hidden_dims_processed = [num_obs if dim == -1 else dim for dim in hidden_dims]
            layers = [nn.Linear(num_obs, hidden_dims_processed[0]), activation_mod]
            for layer_index in range(len(hidden_dims_processed) - 1):
                layers.append(nn.Linear(hidden_dims_processed[layer_index], hidden_dims_processed[layer_index + 1]))
                layers.append(activation_mod)
            self.feature = nn.Sequential(*layers)
            self.value_head = nn.Linear(hidden_dims_processed[-1], 1)
            self.adv_head = nn.Linear(hidden_dims_processed[-1], num_actions)
            print(f"Dueling QNetwork: {self.feature}")
        else:
            self.net = MLP(num_obs, num_actions, hidden_dims, activation)
            print(f"QNetwork MLP: {self.net}")

        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(num_obs)
        else:
            self.obs_normalizer = nn.Identity()

    def get_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def forward(self, obs: TensorDict) -> torch.Tensor:
        obs = self.obs_normalizer(self.get_obs(obs))
        if self.dueling:
            features = self.feature(obs)
            value = self.value_head(features)
            advantage = self.adv_head(features)
            return value + advantage - advantage.mean(dim=-1, keepdim=True)
        return self.net(obs)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            self.obs_normalizer.update(self.get_obs(obs))
