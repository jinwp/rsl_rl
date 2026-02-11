# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from tensordict import TensorDict

from rsl_rl.modules.q_network import QNetwork
from rsl_rl.storage.replay_buffer import ReplayBuffer


class DQN:
    policy: QNetwork
    target_policy: QNetwork

    def __init__(
        self,
        policy: QNetwork,
        target_policy: QNetwork,
        storage: ReplayBuffer,
        num_actions: int,
        gamma: float = 0.99,
        learning_rate: float = 1e-3,
        batch_size: int = 256,
        min_buffer_size: int = 10000,
        target_update_interval: int = 1000,
        target_update_tau: float | None = None,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay_steps: int = 100000,
        update_every: int = 1,
        num_gradient_steps: int = 1,
        max_grad_norm: float | None = None,
        double_q: bool = False,
        device: str = "cpu",
        multi_gpu_cfg: dict | None = None,
    ) -> None:
        self.device = device
        self.policy = policy.to(self.device)
        self.target_policy = target_policy.to(self.device)
        self.target_policy.load_state_dict(self.policy.state_dict())
        self.target_policy.eval()

        self.storage = storage
        self.num_actions = num_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.min_buffer_size = min_buffer_size
        self.target_update_interval = target_update_interval
        self.target_update_tau = target_update_tau
        self.update_every = update_every
        self.num_gradient_steps = num_gradient_steps
        self.max_grad_norm = max_grad_norm
        self.double_q = double_q

        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_steps = max(1, epsilon_decay_steps)
        self.epsilon = epsilon_start

        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        self.learning_rate = learning_rate

        self.total_steps = 0
        self.num_updates = 0
        self.action_std = torch.zeros(1, device=self.device)

        self.last_obs: TensorDict | None = None
        self.last_actions: torch.Tensor | None = None
        self.last_q_values: torch.Tensor | None = None
        self.intrinsic_rewards = None

        self.has_started_training = False

        self.is_multi_gpu = multi_gpu_cfg is not None
        if multi_gpu_cfg is not None:
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_world_size = 1

    def _update_epsilon(self) -> None:
        progress = min(1.0, self.total_steps / self.epsilon_decay_steps)
        self.epsilon = self.epsilon_start + progress * (self.epsilon_end - self.epsilon_start)
        self.action_std.fill_(self.epsilon)

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.last_obs = obs
        self._update_epsilon()

        with torch.no_grad():
            q_values = self.policy(obs)
            # Cache for runner-side logging without extra forward passes.
            self.last_q_values = q_values
            greedy_actions = torch.argmax(q_values, dim=-1)

        random_mask = torch.rand(greedy_actions.shape, device=self.device) < self.epsilon
        random_actions = torch.randint(0, self.num_actions, greedy_actions.shape, device=self.device)
        actions = torch.where(random_mask, random_actions, greedy_actions)

        actions = actions.view(-1, 1)
        self.last_actions = actions
        return actions

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        with torch.no_grad():
            q_values = self.policy(obs)
            self.last_q_values = q_values
            actions = torch.argmax(q_values, dim=-1)
        return actions.view(-1, 1)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        if self.last_obs is None or self.last_actions is None:
            raise RuntimeError("process_env_step called before act.")

        done_flags = dones.float()
        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(done_flags.device).bool().view(-1)

            # If the environment auto-resets before returning observations, `obs` for time-outs may
            # correspond to a reset state. Use the pre-reset terminal observation if provided.
            terminal_obs = extras.get("terminal_observation", None)
            terminal_ids = extras.get("terminal_observation_ids", None)
            if terminal_obs is not None:
                if terminal_ids is None:
                    terminal_ids = time_outs.nonzero(as_tuple=False).squeeze(-1)
                if isinstance(terminal_ids, torch.Tensor) and terminal_ids.numel() > 0:
                    for key in obs.keys():
                        if key in terminal_obs:
                            obs[key][terminal_ids] = terminal_obs[key].to(device=obs[key].device)

            # Only zero out done flag if episode was truncated (timeout), not terminated.
            done_flags = torch.where(time_outs, torch.zeros_like(done_flags), done_flags)

        # Update normalization using the (potentially corrected) next observations.
        self.policy.update_normalization(obs)

        self.storage.add(self.last_obs, self.last_actions, rewards, done_flags, obs)
        self.last_obs = None
        self.last_actions = None
        self.total_steps += rewards.shape[0]

    def update(self) -> dict[str, float]:
        if len(self.storage) < self.min_buffer_size:
            return {"q": 0.0}

        if not self.has_started_training:
            print(f"Replay buffer filled ({len(self.storage)} >= {self.min_buffer_size}). Training started!")
            self.has_started_training = True

        if self.total_steps % self.update_every != 0:
            return {"q": 0.0}

        mean_loss = 0.0
        # Debug stats for diagnosing reward/target scaling issues.
        debug_stats: dict[str, float] = {}
        for _ in range(self.num_gradient_steps):
            obs_batch, actions_batch, rewards_batch, dones_batch, next_obs_batch = self.storage.sample(
                self.batch_size
            )

            with torch.no_grad():
                if self.double_q:
                    next_actions = self.policy(next_obs_batch).argmax(dim=-1, keepdim=True)
                    target_q = self.target_policy(next_obs_batch).gather(1, next_actions)
                else:
                    target_q = self.target_policy(next_obs_batch).max(dim=-1, keepdim=True)[0]
                targets = rewards_batch + (1.0 - dones_batch) * self.gamma * target_q

            q_values = self.policy(obs_batch)
            q_sa = q_values.gather(1, actions_batch.long())

            # Huber loss is commonly more stable for DQN than MSE.
            loss = nn.functional.smooth_l1_loss(q_sa, targets)

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            if self.max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_loss += loss.item()
            self.num_updates += 1

            # Store latest batch stats (useful even with num_gradient_steps > 1).
            debug_stats = {
                "Debug/rewards_batch_mean": float(rewards_batch.mean().item()),
                "Debug/rewards_batch_min": float(rewards_batch.min().item()),
                "Debug/rewards_batch_max": float(rewards_batch.max().item()),
                "Debug/dones_batch_mean": float(dones_batch.float().mean().item()),
                "Debug/target_q_mean": float(target_q.mean().item()),
                "Debug/targets_mean": float(targets.mean().item()),
                "Debug/q_sa_mean": float(q_sa.mean().item()),
                "Debug/gamma": float(self.gamma),
            }

            if self.target_update_interval > 0 and self.num_updates % self.target_update_interval == 0:
                if self.target_update_tau is None:
                    self._update_target()
                else:
                    self._soft_update(self.target_update_tau)

        mean_loss /= self.num_gradient_steps
        out = {"q": mean_loss}
        out.update(debug_stats)
        return out

    def _update_target(self) -> None:
        self.target_policy.load_state_dict(self.policy.state_dict())

    def _soft_update(self, tau: float) -> None:
        for target_param, param in zip(self.target_policy.parameters(), self.policy.parameters()):
            target_param.data.mul_(1.0 - tau).add_(param.data, alpha=tau)

    def broadcast_parameters(self) -> None:
        if not self.is_multi_gpu:
            return
        model_params = [self.policy.state_dict(), self.target_policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])
        self.target_policy.load_state_dict(model_params[1])

    def reduce_parameters(self) -> None:
        grads = [param.grad.view(-1) for param in self.policy.parameters() if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in self.policy.parameters():
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel
