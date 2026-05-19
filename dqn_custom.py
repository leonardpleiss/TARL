import warnings
from typing import Any, ClassVar, Dict, List, Optional, Tuple, Type, TypeVar, Union

import numpy as np
import torch as th
from torch.nn import functional as F

from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import MaybeCallback, Schedule
from stable_baselines3.dqn.policies import CnnPolicy, DQNPolicy, MlpPolicy, MultiInputPolicy, QNetwork
from stable_baselines3.dqn.dqn import DQN
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import GymEnv

SelfDQN = TypeVar("SelfDQN", bound="DQN")


class CustomDQN(DQN):

    policy_aliases: ClassVar[Dict[str, Type[BasePolicy]]] = {
        "MlpPolicy": MlpPolicy,
        "CnnPolicy": CnnPolicy,
        "MultiInputPolicy": MultiInputPolicy,
    }
    # Linear schedule will be defined in `_setup_model()`
    exploration_schedule: Schedule
    q_net: QNetwork
    q_net_target: QNetwork
    policy: DQNPolicy

    def __init__(
        self,
        policy: Union[str, Type[DQNPolicy]],
        env: Union[GymEnv, str],
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,
        learning_starts: int = 100,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        stats_window_size: int = 100,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
        replay_buffer_log_path: str = "",
        target = "DQN",
        oversampling_ratio = 1.,
        td_scale = False,
    ) -> None:
        super().__init__(
            policy,
            env,
            learning_rate,
            buffer_size,
            learning_starts,
            batch_size,
            tau,
            gamma,
            train_freq,
            gradient_steps,
            replay_buffer_class,
            replay_buffer_kwargs,
            optimize_memory_usage,
            target_update_interval,
            exploration_fraction,
            exploration_initial_eps,
            exploration_final_eps,
            max_grad_norm,
            stats_window_size,
            tensorboard_log,
            policy_kwargs,
            verbose,
            seed,
            device,
            _init_setup_model,
            replay_buffer_log_path,
        )

        self.target = target
        self.oversampling_ratio = oversampling_ratio
        self.td_scale = td_scale

        print(f"Initalized new policy, {target=}, {oversampling_ratio=}, {td_scale=}")

    def train(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self._update_learning_rate(self.policy.optimizer)
        
        losses = []

        for gradient_step in range(gradient_steps):

            # Oversampling based on oversampling ratio
            replay_data = self.replay_buffer.sample(int(batch_size*self.oversampling_ratio), env=self._vec_normalize_env)               # KEY CHANGE

            # Get current Q-values estimates
            current_q_values = self.q_net(replay_data.observations)
            current_q_values = th.gather(current_q_values, dim=1, index=replay_data.actions.long())

            with th.no_grad():

                # Next online and offline Q values
                all_next_q_values_online = self.q_net(replay_data.next_observations)
                all_next_q_values_offline = self.q_net_target(replay_data.next_observations)

                next_actions_online = all_next_q_values_online.argmax(dim=1, keepdim=True)
                next_actions_offline = all_next_q_values_offline.argmax(dim=1, keepdim=True)

                # Get online Target
                next_q_values_online = all_next_q_values_online.gather(1, next_actions_online).reshape(-1, 1)
                target_q_values_online = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_online

                # Get offline target                                                                                        
                if self.target == "DQN":                                                                                    
                    next_q_values_offline = all_next_q_values_offline.gather(1, next_actions_offline).reshape(-1, 1)        
                elif self.target == "DDQN":                                                                                 
                    next_q_values_offline = all_next_q_values_offline.gather(1, next_actions_online).reshape(-1, 1)         
                else:                                                                                                       
                    raise ValueError(f"Target not implemented - expects 'DQN' or 'DDQN', provided {self.target}.")          
                target_q_values_offline = replay_data.rewards + (1 - replay_data.dones) * self.gamma * next_q_values_offline
                                                                                        
                # Get alignment metric
                offline_err = target_q_values_offline - current_q_values                                                                # KEY CHANGE
                online_err = target_q_values_online - current_q_values                                                                  # KEY CHANGE

                new_online_err = target_q_values_online - (current_q_values + offline_err)                                              # KEY CHANGE
                online_error_change =  (th.abs(online_err) / (th.abs(online_err) + th.abs(new_online_err) + 1e-8)).detach()             # KEY CHANGE
                online_more_extreme = ((offline_err * online_err) > 0) & (th.abs(online_err) > th.abs(offline_err))                     # KEY CHANGE
                online_error_change = th.where(online_more_extreme, 1, online_error_change)                                             # KEY CHANGE

                # Scale alignment with TD-Error if specified                                                                            
                if self.td_scale:                                                                                                       # KEY CHANGE
                    online_error_change *= th.abs(offline_err)                                                                          # KEY CHANGE
                                                      
                # Get alignment-corrected batch                                                                                         
                _, batch_idxes = th.topk(online_error_change.squeeze(), k=batch_size)                                                   # KEY CHANGE
                
                # Logging
                avg_ratio_before_discard = online_error_change.mean()
                average_ratio_after_discard = online_error_change[batch_idxes].mean()
                max_ratio_before_discard = online_error_change.max()
                max_ratio_after_discard = online_error_change[batch_idxes].max()
                self.logger.record("xustom/avg_ratio_before_discard", avg_ratio_before_discard.item(), exclude="tensorboard")
                self.logger.record("xustom/average_ratio_after_discard", average_ratio_after_discard.item(), exclude="tensorboard")
                self.logger.record("xustom/max_ratio_before_discard", max_ratio_before_discard.item(), exclude="tensorboard")
                self.logger.record("xustom/max_ratio_after_discard", max_ratio_after_discard.item(), exclude="tensorboard")        

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values[batch_idxes], target_q_values_offline[batch_idxes], reduction="none")              # KEY CHANGE
            loss = th.mean(loss)
            losses.append(loss.item())

            # Optimize the policy
            self.policy.optimizer.zero_grad()
            loss.backward()

            # Clip gradient norm
            th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.policy.optimizer.step()

        # Increase update counter
        self._n_updates += gradient_steps

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/loss", np.mean(losses))

    def learn(
        self: SelfDQN,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "CustomDQN",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> SelfDQN:
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )