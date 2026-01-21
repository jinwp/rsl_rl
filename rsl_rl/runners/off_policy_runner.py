# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
import time
import torch
import warnings
from tensordict import TensorDict

from rsl_rl.algorithms import DQN
from rsl_rl.env import VecEnv
from rsl_rl.modules import QNetwork
from rsl_rl.storage import ReplayBuffer
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger


class OffPolicyRunner:
    """Off-policy runner for training and evaluation of discrete-action algorithms."""

    def __init__(self, env: VecEnv, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        self.cfg = train_cfg
        self.policy_cfg = train_cfg["policy"]
        self.cfg["algorithm"].setdefault("rnd_cfg", None)
        self.alg_cfg = dict(train_cfg["algorithm"])
        self.alg_cfg.pop("rnd_cfg", None)

        self.device = device
        self.env = env

        self._configure_multi_gpu()

        obs = self.env.get_observations()
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], self._get_default_obs_sets())

        self.alg = self._construct_algorithm(obs)

        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )

        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if init_at_random_ep_len:
            self.env.episode_length_buf = torch.randint_like(
                self.env.episode_length_buf, high=int(self.env.max_episode_length)
            )

        obs = self.env.get_observations().to(self.device)
        self.train_mode()

        if self.is_distributed:
            print(f"Synchronizing parameters for rank {self.gpu_global_rank}...")
            self.alg.broadcast_parameters()

        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        for it in range(start_it, total_it):
            start = time.time()
            with torch.inference_mode():
                for _ in range(self.cfg["num_steps_per_env"]):
                    actions = self.alg.act(obs)
                    env_actions = actions.squeeze(-1) if actions.dim() == 2 and actions.shape[-1] == 1 else actions
                    obs, rewards, dones, extras = self.env.step(env_actions.to(self.env.device))
                    obs, rewards, dones = (obs.to(self.device), rewards.to(self.device), dones.to(self.device))
                    self.alg.process_env_step(obs, rewards, dones, extras)
                    self.logger.process_env_step(rewards, dones, extras, None)

                stop = time.time()
                collect_time = stop - start
                start = stop

            loss_dict = self.alg.update()

            stop = time.time()
            learn_time = stop - start
            self.current_learning_iteration = it

            self.logger.log(
                it=it,
                start_it=start_it,
                total_it=total_it,
                collect_time=collect_time,
                learn_time=learn_time,
                loss_dict=loss_dict,
                learning_rate=self.alg.learning_rate,
                action_std=self.alg.action_std,
                rnd_weight=None,
            )

            if it % self.cfg["save_interval"] == 0:
                self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))  # type: ignore

        if self.logger.log_dir is not None and not self.logger.disable_logs:
            self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))

    def save(self, path: str, infos: dict | None = None) -> None:
        saved_dict = {
            "model_state_dict": self.alg.policy.state_dict(),
            "target_model_state_dict": self.alg.target_policy.state_dict(),
            "optimizer_state_dict": self.alg.optimizer.state_dict(),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        torch.save(saved_dict, path)
        self.logger.save_model(path, self.current_learning_iteration)

    def load(self, path: str, load_optimizer: bool = True, map_location: str | None = None) -> dict:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        self.alg.policy.load_state_dict(loaded_dict["model_state_dict"])
        if "target_model_state_dict" in loaded_dict:
            self.alg.target_policy.load_state_dict(loaded_dict["target_model_state_dict"])
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict["infos"]

    def get_inference_policy(self, device: str | None = None) -> callable:
        self.eval_mode()
        if device is not None:
            self.alg.policy.to(device)
        return self.alg.act_inference

    def train_mode(self) -> None:
        self.alg.policy.train()
        self.alg.target_policy.eval()

    def eval_mode(self) -> None:
        self.alg.policy.eval()
        self.alg.target_policy.eval()

    def add_git_repo_to_log(self, repo_file_path: str) -> None:
        self.logger.git_status_repos.append(repo_file_path)

    def _get_default_obs_sets(self) -> list[str]:
        return []

    def _configure_multi_gpu(self) -> None:
        self.gpu_world_size = int(os.getenv("WORLD_SIZE", "1"))
        self.is_distributed = self.gpu_world_size > 1

        if not self.is_distributed:
            self.gpu_local_rank = 0
            self.gpu_global_rank = 0
            self.multi_gpu_cfg = None
            return

        self.gpu_local_rank = int(os.getenv("LOCAL_RANK", "0"))
        self.gpu_global_rank = int(os.getenv("RANK", "0"))

        self.multi_gpu_cfg = {
            "global_rank": self.gpu_global_rank,
            "local_rank": self.gpu_local_rank,
            "world_size": self.gpu_world_size,
        }

        if self.device != f"cuda:{self.gpu_local_rank}":
            raise ValueError(
                f"Device '{self.device}' does not match expected device for local rank '{self.gpu_local_rank}'."
            )
        if self.gpu_local_rank >= self.gpu_world_size:
            raise ValueError(
                f"Local rank '{self.gpu_local_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )
        if self.gpu_global_rank >= self.gpu_world_size:
            raise ValueError(
                f"Global rank '{self.gpu_global_rank}' is greater than or equal to world size '{self.gpu_world_size}'."
            )

        torch.distributed.init_process_group(backend="nccl", rank=self.gpu_global_rank, world_size=self.gpu_world_size)
        torch.cuda.set_device(self.gpu_local_rank)

    def _construct_algorithm(self, obs: TensorDict) -> DQN:
        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `obs_normalization` as part of the "
                "`policy` configuration instead.",
                DeprecationWarning,
            )
            if self.policy_cfg.get("obs_normalization") is None:
                self.policy_cfg["obs_normalization"] = self.cfg["empirical_normalization"]

        policy_class = resolve_callable(self.policy_cfg.pop("class_name"))
        policy: QNetwork = policy_class(obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg).to(
            self.device
        )
        target_policy: QNetwork = policy_class(
            obs, self.cfg["obs_groups"], self.env.num_actions, **self.policy_cfg
        ).to(self.device)

        replay_buffer_size = self.alg_cfg.pop("replay_buffer_size")
        storage = ReplayBuffer(replay_buffer_size, obs, [1], self.device)

        alg_class = resolve_callable(self.alg_cfg.pop("class_name"))
        alg: DQN = alg_class(
            policy,
            target_policy,
            storage,
            num_actions=self.env.num_actions,
            device=self.device,
            multi_gpu_cfg=self.multi_gpu_cfg,
            **self.alg_cfg,
        )

        return alg
