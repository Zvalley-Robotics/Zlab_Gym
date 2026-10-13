# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from itertools import chain
from tensordict import TensorDict

from rsl_rl.env import VecEnv
from rsl_rl.extensions import RandomNetworkDistillation, resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer


def _interp_obs_dict(obs: TensorDict, next_obs: TensorDict, weights: torch.Tensor) -> TensorDict:
    """Linearly interpolate two observation TensorDicts along the leading axis.

    Used by L2C2 smoothness regularization to construct a perturbation
    ``obs + w * (next_obs - obs)`` of each observation in the (s_t -> s_{t+1})
    direction. ``weights`` has shape [B] and is broadcast against every feature
    of every observation group (including multi-frame history groups).
    """
    interp = {}
    for k in obs.keys():
        w = weights.view([obs[k].shape[0]] + [1] * (obs[k].dim() - 1))
        interp[k] = obs[k] + w * (next_obs[k] - obs[k])
    return TensorDict(interp, batch_size=obs.batch_size, device=obs.device)


class PPO:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: MLPModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
        # Multi-Critic (MuC) parameters: number of value heads and per-group weights.
        num_critics: int = 1,
        reward_group_weights: tuple[float, ...] | list[float] | None = None,
        # L2C2 (Locally Lipschitz Continuous Constraint) smoothness regularization
        smoothness_lower_bound: float | None = None,
        smoothness_upper_bound: float | None = None,
        value_smoothness_coef: float = 0.01,
    ) -> None:
        """Initialize the algorithm with models, storage, and optimization settings."""
        # Device-related parameters
        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None

        # Multi-GPU parameters
        if multi_gpu_cfg is not None:
            self.gpu_global_rank = multi_gpu_cfg["global_rank"]
            self.gpu_world_size = multi_gpu_cfg["world_size"]
        else:
            self.gpu_global_rank = 0
            self.gpu_world_size = 1

        # RND components
        if rnd_cfg:
            # Extract parameters used in ppo
            rnd_lr = rnd_cfg.pop("learning_rate", 1e-3)
            # Create RND module
            self.rnd = RandomNetworkDistillation(device=self.device, **rnd_cfg)
            # Create RND optimizer
            params = self.rnd.predictor.parameters()
            self.rnd_optimizer = optim.Adam(params, lr=rnd_lr)
        else:
            self.rnd = None
            self.rnd_optimizer = None

        # Symmetry components
        if symmetry_cfg is not None:
            # Check if symmetry is enabled
            use_symmetry = symmetry_cfg["use_data_augmentation"] or symmetry_cfg["use_mirror_loss"]
            # Print that we are not using symmetry
            if not use_symmetry:
                print("Symmetry not used for learning. We will use it for logging instead.")
            # Resolve the data augmentation function (supports string names or direct callables)
            symmetry_cfg["data_augmentation_func"] = resolve_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(symmetry_cfg["data_augmentation_func"]):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{symmetry_cfg['data_augmentation_func']}"
                )
            # Check if the policy is compatible with symmetry
            if actor.is_recurrent or critic.is_recurrent:
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Create the optimizer
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor.parameters(), self.critic.parameters()), lr=learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorage.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        # [2026-08-17] Entropy-coef linear-annealing config (anti-divergence fix).
        #   Intent: in the previous training run `entropy_coef` was a constant 0.005.
        #   That constant exploration bonus kept pushing the Gaussian action std from
        #   1.0 up to ~4.3, which eventually produced extreme actions and blew up the
        #   simulation (robot launched ~58m, reward overflowed to 1e25). We now anneal
        #   entropy_coef linearly from its configured start value to `entropy_coef_end`
        #   over the first `entropy_anneal_iters` learning iterations, then hold at the
        #   floor, so the policy converges to a low-noise (small-std) solution.
        #   Tuning: lower `entropy_anneal_iters` to remove exploration faster; raise
        #   `entropy_coef_end` if you still want a little residual exploration.
        self.entropy_coef = entropy_coef
        self.entropy_coef_start = entropy_coef
        self.entropy_coef_end = 0.0
        self.entropy_anneal_iters = 3000
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch

        # L2C2 (Locally Lipschitz Continuous Constraint) smoothness regularization.
        #   HoST (paper Sec. IV-D) applies this to BOTH actor and critic: penalize
        #   how much the policy/value change when the observation is perturbed along
        #   the (s_t -> s_{t+1}) direction. This stabilizes the sparse, non-convex
        #   get-up optimization. Enabled only when both bounds are provided
        #   (default None -> disabled, preserving the standard single-critic path).
        self.use_l2c2 = smoothness_lower_bound is not None and smoothness_upper_bound is not None
        if self.use_l2c2:
            self.smoothness_lower_bound = smoothness_lower_bound
            self.smoothness_upper_bound = smoothness_upper_bound
            self.value_smoothness_coef = value_smoothness_coef
            # Reproduce HoST's coefficient schedule (ppo.py:182-185 upstream).
            epsilon = self.smoothness_lower_bound / (self.smoothness_upper_bound - self.smoothness_lower_bound)
            self.policy_smooth_coef = self.smoothness_upper_bound * epsilon
            self.value_smooth_coef = self.value_smoothness_coef * self.policy_smooth_coef
        else:
            self.policy_smooth_coef = 0.0
            self.value_smooth_coef = 0.0

        # Multi-Critic (MuC): HoST uses one value head per reward group
        #   (task/style/regu/post). Each head gets its own GAE stream and
        #   independently normalized advantage; the policy gradient then uses the
        #   group-weighted sum (see compute_returns). num_critics=1 with
        #   reward_group_weights=None preserves the standard single-critic path.
        self.num_critics = int(num_critics)
        if self.num_critics > 1:
            if reward_group_weights is None or len(reward_group_weights) != self.num_critics:
                raise ValueError(
                    "MuC requires `reward_group_weights` with one weight per critic "
                    f"(got num_critics={self.num_critics}, weights={reward_group_weights})."
                )
            self.reward_group_weights = list(reward_group_weights)
        else:
            self.reward_group_weights = None

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample actions and store transition data."""
        # Record the hidden states for recurrent policies
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        # Compute the actions and values
        self.transition.actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(self.transition.actions).detach()  # type: ignore
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        # Record observations before env.step()
        self.transition.observations = obs
        return self.transition.actions  # type: ignore

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        # Update the normalizers
        self.actor.update_normalization(obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones

        # Record next observations for L2C2 smoothness regularization. The ``obs``
        # argument here is already s_{t+1} (the runner steps the env before calling
        # process_env_step), while ``transition.observations`` holds s_t from ``act``.
        if self.storage.store_next_observations:
            self.transition.next_observations = obs

        # Compute the intrinsic rewards and add to extrinsic rewards
        if self.rnd:
            # Compute the intrinsic rewards
            self.intrinsic_rewards = self.rnd.get_intrinsic_reward(obs)
            # Add intrinsic rewards to extrinsic rewards
            self.transition.rewards += self.intrinsic_rewards

        # Bootstrapping on time outs
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].unsqueeze(1).to(self.device),  # type: ignore
                1,
            )

        # Record the transition
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs: TensorDict) -> None:
        """Compute return and advantage targets from stored transitions."""
        st = self.storage
        # Compute value for the last step
        last_values = self.critic(obs).detach()
        # [2026-08-18] NaN/inf safety net: a sim blow-up (e.g. contact_sensor_maxmatch
        #   overflow) can make critic outputs non-finite. If they enter the TD backup,
        #   returns/advantages become NaN and the value network learns NaN weights.
        #   Replace any non-finite bootstrap value with 0 so the backup degrades
        #   gracefully to finite rewards.
        last_values = torch.nan_to_num(last_values, nan=0.0, posinf=0.0, neginf=0.0)
        # Compute returns and advantages
        advantage = torch.zeros_like(st.rewards[0])
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            # [2026-08-18] Sanitize any NaN/inf in stored rewards/values before backup.
            reward = torch.nan_to_num(st.rewards[step], nan=0.0, posinf=0.0, neginf=0.0)
            value = torch.nan_to_num(st.values[step], nan=0.0, posinf=0.0, neginf=0.0)
            next_values = torch.nan_to_num(next_values, nan=0.0, posinf=0.0, neginf=0.0)
            delta = reward + next_is_not_terminal * self.gamma * next_values - value
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + value
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Sanitize normalized advantages so a single bad transition cannot corrupt
        # an entire mini-batch.
        st.returns = torch.nan_to_num(st.returns, nan=0.0, posinf=0.0, neginf=0.0)
        st.advantages = torch.nan_to_num(st.advantages, nan=0.0, posinf=0.0, neginf=0.0)
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            if self.num_critics > 1 and self.reward_group_weights is not None:
                # MuC (HoST): independently normalize each critic's advantage
                # stream, then combine with the per-group weights. This prevents
                # one group (e.g. a sparse, high-variance task reward) from
                # dominating the normalized advantage of the others.
                mean = st.advantages.mean(dim=(0, 1), keepdim=True)
                std = st.advantages.std(dim=(0, 1), keepdim=True)
                norm_advantages = (st.advantages - mean) / (std + 1e-8)
                weights = torch.tensor(self.reward_group_weights, device=self.device).view(1, 1, -1)
                st.advantages = (norm_advantages * weights).sum(dim=-1, keepdim=True)
            else:
                st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self, iteration: int = 0) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses.

        Args:
            iteration: Current learning iteration index (0-based). Used to linearly
                anneal ``entropy_coef`` over training so exploration decays over time.
        """
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None

        # [2026-08-17] Apply entropy-coef linear annealing for this iteration.
        #   Intent: shrink the entropy bonus as training progresses so the policy
        #   stops being rewarded for random (high-std) actions and settles into a
        #   low-noise behavior. Fraction is clamped to [0, 1]; once `iteration`
        #   exceeds `entropy_anneal_iters`, entropy_coef stays at `entropy_coef_end`.
        anneal_frac = min(1.0, iteration / max(1, self.entropy_anneal_iters))
        self.entropy_coef = max(
            self.entropy_coef_end,
            self.entropy_coef_start * (1.0 - anneal_frac),
        )

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        # Iterate over batches
        for batch in generator:
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Augmentation using symmetry
                data_augmentation_func = self.symmetry["data_augmentation_func"]
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = data_augmentation_func(
                    env=self.symmetry["_env"],
                    obs=batch.observations,
                    actions=batch.actions,
                )
                # Compute number of augmentations per sample
                num_aug = int(batch.observations.batch_size[0] / original_batch_size)
                # Repeat the rest of the batch
                batch.old_actions_log_prob = batch.old_actions_log_prob.repeat(num_aug, 1)
                batch.values = batch.values.repeat(num_aug, 1)
                batch.advantages = batch.advantages.repeat(num_aug, 1)
                batch.returns = batch.returns.repeat(num_aug, 1)

            # Recompute actions log prob and entropy for current batch of transitions
            # Note: We need to do this because we updated the policy with the new parameters
            self.actor(
                batch.observations,
                masks=batch.masks,
                hidden_state=batch.hidden_states[0],
                stochastic_output=True,
            )
            actions_log_prob = self.actor.get_output_log_prob(batch.actions)  # type: ignore
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            # Note: We only keep the distribution parameters and entropy of the first augmentation (the original one)
            distribution_params = tuple(p[:original_batch_size] for p in self.actor.output_distribution_params)
            entropy = self.actor.output_entropy[:original_batch_size]

            # Compute KL divergence and adapt the learning rate
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = self.actor.get_kl_divergence(batch.old_distribution_params, distribution_params)  # type: ignore
                    kl_mean = torch.mean(kl)

                    # Reduce the KL divergence across all GPUs
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    # Update the learning rate only on the main process
                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    # Update the learning rate for all GPUs
                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    # Update the learning rate for all parameter groups
                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            # Surrogate loss
            ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))  # type: ignore
            surrogate = -torch.squeeze(batch.advantages) * ratio  # type: ignore
            surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(  # type: ignore
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            # Value function loss
            if self.use_clipped_value_loss:
                value_clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_losses = (values - batch.returns).pow(2)
                value_losses_clipped = (value_clipped - batch.returns).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (batch.returns - values).pow(2).mean()

            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy.mean()

            # L2C2 (Locally Lipschitz Continuous Constraint) smoothness regularization
            #   (HoST Sec. IV-D): penalize how much the actor/critic output changes
            #   when the observation is perturbed along the (s_t -> s_{t+1}) direction.
            #   This is a training-loss regularizer (NOT a reward term), applied to
            #   both the policy and the value function.
            if self.use_l2c2 and batch.next_observations is not None and batch.dones is not None:
                # Exclude episode-boundary transitions (next_obs is a fresh reset).
                cont_batch = (1.0 - batch.dones.float()).squeeze(-1)
                mix_weights = cont_batch * (torch.rand_like(cont_batch) - 0.5) * 2.0
                mix_obs_batch = _interp_obs_dict(batch.observations, batch.next_observations, mix_weights)
                # Policy smoothness: actor mean output continuity.
                mu_batch = self.actor.output_mean
                mu_mix = self.actor(mix_obs_batch)
                policy_smooth_loss = (mu_batch - mu_mix).pow(2).sum(dim=-1).mean()
                # Value smoothness: critic output continuity.
                value_mix = self.critic(mix_obs_batch)
                value_smooth_loss = (values - value_mix).pow(2).sum(dim=-1).mean()
                smooth_loss = self.policy_smooth_coef * policy_smooth_loss + self.value_smooth_coef * value_smooth_loss
                loss = loss + smooth_loss

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    data_augmentation_func = self.symmetry["data_augmentation_func"]
                    batch.observations, _ = data_augmentation_func(
                        obs=batch.observations, actions=None, env=self.symmetry["_env"]
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = data_augmentation_func(
                    obs=None, actions=action_mean_orig, env=self.symmetry["_env"]
                )

                # Compute the loss
                mse_loss = torch.nn.MSELoss()
                symmetry_loss = mse_loss(
                    mean_actions[original_batch_size:], actions_mean_symm.detach()[original_batch_size:]
                )
                # Add the loss to the total loss
                if self.symmetry["use_mirror_loss"]:
                    loss += self.symmetry["mirror_loss_coeff"] * symmetry_loss
                else:
                    symmetry_loss = symmetry_loss.detach()

            # RND loss
            rnd_loss = None
            if self.rnd:
                # Extract the rnd_state
                with torch.no_grad():
                    rnd_state = self.rnd.get_rnd_state(batch.observations[:original_batch_size])  # type: ignore
                    rnd_state = self.rnd.state_normalizer(rnd_state)
                # Predict the embedding and the target
                predicted_embedding = self.rnd.predictor(rnd_state)
                target_embedding = self.rnd.target(rnd_state).detach()
                # Compute the loss as the mean squared error
                mseloss = torch.nn.MSELoss()
                rnd_loss = mseloss(predicted_embedding, target_embedding)

            # [2026-08-18] NaN/inf safety net: if the current mini-batch is contaminated
            #   (e.g. from a sim blow-up that produced NaN rewards/obs), skip the gradient
            #   step entirely. Otherwise the optimizer applies NaN gradients and the next
            #   forward pass produces NaN std, which crashes `torch.normal`. We also guard
            #   RND independently because it is optimized separately.
            skip = not torch.isfinite(loss)
            if self.rnd and not torch.isfinite(rnd_loss):
                skip = True
            if skip:
                mean_value_loss += 0.0
                mean_surrogate_loss += 0.0
                mean_entropy += 0.0
                if mean_rnd_loss is not None:
                    mean_rnd_loss += 0.0
                if mean_symmetry_loss is not None:
                    mean_symmetry_loss += 0.0
                continue

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()

            # Store the losses
            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy.mean().item()
            # RND loss
            if mean_rnd_loss is not None:
                mean_rnd_loss += rnd_loss.item()
            # Symmetry loss
            if mean_symmetry_loss is not None:
                mean_symmetry_loss += symmetry_loss.item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            # [2026-08-17] Log the (annealed) entropy coefficient so we can confirm
            # the annealing schedule is actually taking effect during training.
            "entropy_coef": self.entropy_coef,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }
        if self.rnd:
            saved_dict["rnd_state_dict"] = self.rnd.state_dict()
            saved_dict["rnd_optimizer_state_dict"] = self.rnd_optimizer.state_dict()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        """Load specified models from a saved dict."""
        # If no load_cfg is provided, load all models and states
        if load_cfg is None:
            load_cfg = {
                "actor": True,
                "critic": True,
                "optimizer": True,
                "iteration": True,
                "rnd": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> MLPModel:
        """Get the policy model."""
        return self.actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Construct the PPO algorithm."""
        # Resolve class callables
        alg_class: type[PPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[MLPModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve symmetry config if used
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Initialize the policy
        actor: MLPModel = actor_class(obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        # Multi-Critic (MuC): the critic outputs one value per reward group.
        num_critics = int(cfg["algorithm"].get("num_critics", 1))
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", num_critics, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage. Next-observation storage is only needed for the
        # L2C2 smoothness regularizer (kept off otherwise to avoid doubling obs memory).
        use_l2c2 = (
            cfg["algorithm"].get("smoothness_lower_bound") is not None
            and cfg["algorithm"].get("smoothness_upper_bound") is not None
        )
        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
            store_next_observations=use_l2c2,
            num_critics=num_critics,
        )

        # Initialize the algorithm
        alg: PPO = alg_class(actor, critic, storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.actor.state_dict(), self.critic.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[2])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters())
        if self.rnd:
            all_params = chain(all_params, self.rnd.parameters())
        all_params = list(all_params)
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        all_grads = torch.cat(grads)
        # Average the gradients across all GPUs
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        # Update the gradients for all parameters with the reduced gradients
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                # Copy data back from shared buffer
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                # Update the offset for the next parameter
                offset += numel
