# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from tensordict import TensorDict


class ReplayBuffer:
    """Replay buffer for off-policy algorithms."""

    def __init__(
        self,
        capacity: int,
        obs: TensorDict,
        actions_shape: tuple[int] | list[int],
        device: str = "cpu",
    ) -> None:
        self.capacity = capacity
        self.device = device
        self.actions_shape = tuple(actions_shape)

        self.observations = {
            key: torch.zeros((capacity, *value.shape[1:]), device=device, dtype=value.dtype)
            for key, value in obs.items()
        }
        self.next_observations = {
            key: torch.zeros((capacity, *value.shape[1:]), device=device, dtype=value.dtype)
            for key, value in obs.items()
        }

        self.actions = torch.zeros((capacity, *self.actions_shape), device=device, dtype=torch.long)
        self.rewards = torch.zeros((capacity, 1), device=device)
        self.dones = torch.zeros((capacity, 1), device=device)

        self.ptr = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        obs: TensorDict,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        next_obs: TensorDict,
    ) -> None:
        batch_size = actions.shape[0]
        indices = (torch.arange(batch_size, device=self.device) + self.ptr) % self.capacity

        for key, value in obs.items():
            self.observations[key][indices].copy_(value)
        for key, value in next_obs.items():
            self.next_observations[key][indices].copy_(value)

        actions = actions.view(batch_size, *self.actions_shape).to(dtype=torch.long)
        self.actions[indices].copy_(actions)
        self.rewards[indices].copy_(rewards.view(batch_size, 1))
        self.dones[indices].copy_(dones.view(batch_size, 1))

        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size: int) -> tuple[TensorDict, torch.Tensor, torch.Tensor, torch.Tensor, TensorDict]:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)

        obs_batch = TensorDict(
            {key: value[indices] for key, value in self.observations.items()},
            batch_size=[batch_size],
            device=self.device,
        )
        next_obs_batch = TensorDict(
            {key: value[indices] for key, value in self.next_observations.items()},
            batch_size=[batch_size],
            device=self.device,
        )

        actions_batch = self.actions[indices]
        rewards_batch = self.rewards[indices]
        dones_batch = self.dones[indices]

        return obs_batch, actions_batch, rewards_batch, dones_batch, next_obs_batch
