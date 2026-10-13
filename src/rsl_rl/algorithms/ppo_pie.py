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
from rsl_rl.utils import resolve_callable, resolve_obs_groups, resolve_optimizer
from src.rsl_rl.models import PIEModel, AMPModel, resolve_amp_config
from src.rsl_rl.storage import RolloutStorageVAE, CircularBuffer
from src.rsl_rl.models.amp_model import LossType


class PPOPIE:
    """Proximal Policy Optimization algorithm.

    Reference:
        - Schulman et al. "Proximal policy optimization algorithms." arXiv preprint arXiv:1707.06347 (2017).
    """

    actor: PIEModel
    """The actor model."""

    critic: MLPModel
    """The critic model."""

    def __init__(
        self,
        actor: PIEModel,
        critic: MLPModel,
        storage: RolloutStorageVAE,
        disc_obs_buffer: CircularBuffer, 
        disc_demo_obs_buffer: CircularBuffer,
        obs_groups: dict,
        vae_kld_weight: float = 1.0,
        explicit_estimation_loss_weight: float = 1.0,
        reconstruction_loss_weight: float = 1.0,
        map_loss_weight: float = 1.0,
        num_encoder_epochs: int = 1,
        num_learning_epochs: int = 5,
        num_mini_batches: int = 4,
        clip_param: float = 0.2,
        gamma: float = 0.99,
        lam: float = 0.95,
        value_loss_coef: float = 1.0,
        entropy_coef: float = 0.01,
        learning_rate: float = 0.001,
        vae_learning_rate: float = 0.001,
        max_grad_norm: float = 1.0,
        optimizer: str = "adam",
        amp_optimizer: str = "adam",
        use_clipped_value_loss: bool = True,
        schedule: str = "adaptive",
        desired_kl: float = 0.01,
        normalize_advantage_per_mini_batch: bool = False,
        device: str = "cpu",
        # AMP parameters
        amp_cfg: dict | None = None,
        # RND parameters
        rnd_cfg: dict | None = None,
        # Symmetry parameters
        symmetry_cfg: dict | None = None,
        # Distributed training parameters
        multi_gpu_cfg: dict | None = None,
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
            self.data_augmentation_func = resolve_callable(symmetry_cfg["data_augmentation_func"])
            # Check valid configuration
            if not callable(self.data_augmentation_func):
                raise ValueError(
                    f"Symmetry configuration exists but the function is not callable: "
                    f"{self.data_augmentation_func}"
                )
            # Check if the policy is compatible with symmetry
            if actor.is_recurrent or critic.is_recurrent:
                raise ValueError("Symmetry augmentation is not supported for recurrent policies.")
            # Store symmetry configuration
            self.symmetry = symmetry_cfg
        else:
            self.symmetry = None

        # AMP components
        self.amp_cfg = amp_cfg
        if self.amp_cfg is None:
            raise ValueError("AMP configuration must be provided for PPOAMP algorithm.")

        if self.amp_cfg["loss_type"] == "GAN":
            self.loss_type = LossType.GAN
        elif self.amp_cfg["loss_type"] == "LSGAN":
            self.loss_type = LossType.LSGAN
        elif self.amp_cfg["loss_type"] == "WGAN":
            self.loss_type = LossType.WGAN
        else:
            raise ValueError(f"Unknown AMP loss type: {self.amp_cfg['loss_type']}. Should be 'GAN', 'LSGAN', or 'WGAN'")

        self.amp_discriminator: AMPModel = AMPModel(
            disc_obs_dim=self.amp_cfg["disc_obs_dim"],
            disc_obs_steps=self.amp_cfg["disc_obs_steps"],
            obs_groups=obs_groups,
            loss_type=self.loss_type,
            **self.amp_cfg.get("amp_discriminator", {}),
        ).to(self.device)
        print(f"Amp Model: {self.amp_discriminator}")

        # optimizer for policy and discriminator
        self.amp_parameters = [
            {
                "name": "disc_trunk", 
                "params": self.amp_discriminator.disc_trunk.parameters(),
                "weight_decay": self.amp_cfg["disc_trunk_weight_decay"],  # L2 regularization for the discriminator trunk
            },
            {
                "name": "disc_linear",
                "params": self.amp_discriminator.disc_linear.parameters(),
                "weight_decay": self.amp_cfg["disc_linear_weight_decay"],  # L2 regularization for the discriminator linear layer
            },
        ]
        # use a separate optimizer for the AMP discriminator
        self.disc_optimizer = resolve_optimizer(amp_optimizer)(
            self.amp_parameters,
            lr=self.amp_cfg["disc_learning_rate"],
        )
        self.disc_max_grad_norm = self.amp_cfg.get("disc_max_grad_norm", 0.5)

        # Storage for AMP discriminator observations
        self.disc_obs_buffer: CircularBuffer = disc_obs_buffer
        self.disc_demo_obs_buffer: CircularBuffer = disc_demo_obs_buffer

        # PPO components
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)

        # Create the optimizer
        self.actor_parameters = [
            param for name, param in self.actor.named_parameters()
            if not name.startswith("vae.")
        ]
        self.optimizer = resolve_optimizer(optimizer)(
            chain(self.actor_parameters, self.critic.parameters()), lr=learning_rate
        )  # type: ignore
        self.vae_optimizer = resolve_optimizer(optimizer)(
            self.actor.vae.parameters(), lr=vae_learning_rate
        )  # type: ignore

        # Add storage
        self.storage = storage
        self.transition = RolloutStorageVAE.Transition()

        # PPO parameters
        self.clip_param = clip_param
        self.num_encoder_epochs = num_encoder_epochs
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.learning_rate = learning_rate
        self.normalize_advantage_per_mini_batch = normalize_advantage_per_mini_batch
        self.vae_kld_weight = vae_kld_weight
        self.explicit_estimation_loss_weight = explicit_estimation_loss_weight
        self.reconstruction_loss_weight = reconstruction_loss_weight
        self.map_loss_weight = map_loss_weight
        self.num_encoder_epochs = num_encoder_epochs
        self.vae_learning_rate = vae_learning_rate

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
        self, obs: TensorDict, next_obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        """Record one environment step and update the normalizers."""
        disc_obs = self.amp_discriminator.get_disc_obs(obs, flatten_history_dim=False)
        disc_demo_obs = self.amp_discriminator.get_disc_demo_obs(obs, flatten_history_dim=False)
        # Compute the Style Reward
        self.style_rewards, self.disc_score = self.amp_discriminator.predict_style_reward(disc_obs, dt=self.amp_cfg["step_dt"])
        # Linearly interpolate between task reward and style reward
        self.rewards_lerp = self.amp_discriminator.lerp_reward(task_reward=rewards, style_reward=self.style_rewards)
        # Store the un-normalized disc obs and disc demo obs into buffers
        self.disc_obs_buffer.append(disc_obs)
        self.disc_demo_obs_buffer.append(disc_demo_obs)

        # Update the normalizers
        self.actor.update_normalization(obs, next_obs)
        self.critic.update_normalization(obs)
        if self.rnd:
            self.rnd.update_normalization(obs)

        # Record the rewards and dones
        # Note: We clone here because later on we bootstrap the rewards based on timeouts
        self.transition.next_observations = next_obs.clone()
        self.transition.rewards = self.rewards_lerp.clone()
        self.transition.dones = dones

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
        # Compute returns and advantages
        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            # If we are at the last step, bootstrap the return value
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            # 1 if we are not in a terminal state, 0 otherwise
            next_is_not_terminal = 1.0 - st.dones[step].float()
            # TD error: r_t + gamma * V(s_{t+1}) - V(s_t)
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            # Advantage: A(s_t, a_t) = delta_t + gamma * lambda * A(s_{t+1}, a_{t+1})
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            # Return: R_t = A(s_t, a_t) + V(s_t)
            st.returns[step] = advantage + st.values[step]
        # Compute the advantages
        st.advantages = st.returns - st.values
        # Normalize the advantages if per minibatch normalization is not used
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def update(self) -> dict[str, float]:
        """Run optimization epochs over stored batches and return mean losses."""
        mean_value_loss = 0
        mean_surrogate_loss = 0
        mean_entropy = 0
        mean_explicit_estimation_loss = 0
        mean_reconstruction_loss = 0
        mean_map_loss = 0
        mean_kld_loss = 0
        # RND loss
        mean_rnd_loss = 0 if self.rnd else None
        # Symmetry loss
        mean_symmetry_loss = 0 if self.symmetry else None
        # AMP discriminator loss and other info
        mean_disc_loss = 0
        mean_disc_grad_penalty = 0
        mean_disc_score = 0
        mean_disc_demo_score = 0

        # Get mini batch generator
        if self.actor.is_recurrent or self.critic.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        disc_obs_generator = self.disc_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,  # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )
        disc_demo_obs_generator = self.disc_demo_obs_buffer.mini_batch_generator(
            fetch_length=self.storage.num_transitions_per_env,  # type: ignore
            num_mini_batches=self.num_mini_batches,
            num_epochs=self.num_learning_epochs,
        )

        # Iterate over batches
        for batch, disc_obs_batch, disc_demo_obs_batch in zip(generator, disc_obs_generator, disc_demo_obs_generator):
            original_batch_size = batch.observations.batch_size[0]

            # Check if we should normalize advantages per mini batch
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    batch.advantages = (batch.advantages - batch.advantages.mean()) / (batch.advantages.std() + 1e-8)  # type: ignore

            # Perform symmetric augmentation
            if self.symmetry and self.symmetry["use_data_augmentation"]:
                # Returned shape: [batch_size * num_aug, ...]
                batch.observations, batch.actions = self.data_augmentation_func(
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

            # Symmetry loss
            if self.symmetry:
                # Obtain the symmetric actions
                # Note: If we did augmentation before then we don't need to augment again
                if not self.symmetry["use_data_augmentation"]:
                    batch.observations, _ = self.data_augmentation_func(
                        obs=batch.observations, actions=None
                    )

                # Actions predicted by the actor for symmetrically-augmented observations
                mean_actions = self.actor(batch.observations.detach().clone())

                # Compute the symmetrically augmented actions
                # Note: We are assuming the first augmentation is the original one. We do not use the batch.actions from
                # earlier since that action was sampled from the distribution. However, the symmetry loss is computed
                # using the mean of the distribution.
                action_mean_orig = mean_actions[:original_batch_size]
                _, actions_mean_symm = self.data_augmentation_func(
                    obs=None, actions=action_mean_orig
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

            # AMP loss
            with torch.no_grad():
                disc_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_obs_batch)  # [mini_batch_size, disc_obs_steps, disc_obs_dim]
                disc_demo_obs_batch_normed = self.amp_discriminator.normalize_disc_obs(disc_demo_obs_batch)

            mini_batch_size = disc_obs_batch_normed.shape[0]
            disc_score = self.amp_discriminator(disc_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]
            disc_demo_score = self.amp_discriminator(disc_demo_obs_batch_normed.reshape(mini_batch_size, -1))  # [mini_batch_size, 1]

            if self.loss_type == LossType.GAN:
                bce = torch.nn.BCEWithLogitsLoss()
                policy_loss = bce(
                    disc_score, torch.zeros_like(disc_score, device=self.device)
                )
                demo_loss = bce(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.LSGAN:
                policy_loss = torch.nn.MSELoss()(
                    disc_score, -1 * torch.ones_like(disc_score, device=self.device)
                )
                demo_loss = torch.nn.MSELoss()(
                    disc_demo_score, torch.ones_like(disc_demo_score, device=self.device)
                )
                disc_loss = 0.5 * (policy_loss + demo_loss)
            elif self.loss_type == LossType.WGAN:
                disc_loss = - torch.mean(disc_demo_score) + torch.mean(disc_score)
            else: 
                raise ValueError(f"Unknown AMP loss type: {self.loss_type}. Should be 'GAN', 'LSGAN', or 'WGAN'")

            disc_grad_penalty = self.amp_discriminator.compute_grad_penalty(
                demo_data=disc_demo_obs_batch_normed.reshape(mini_batch_size, -1),
                scale=self.amp_cfg["grad_penalty_scale"]
            )
            disc_total_loss = disc_loss + disc_grad_penalty

            # Compute the gradients for PPO
            self.optimizer.zero_grad()
            loss.backward()
            # Compute the gradients for RND
            if self.rnd:
                self.rnd_optimizer.zero_grad()
                rnd_loss.backward()
            # Compute the gradients for AMP discriminator
            self.disc_optimizer.zero_grad()
            disc_total_loss.backward()

            # Compute the gradients for VAE
            for _ in range(self.num_encoder_epochs):
                # VAE loss
                explicit_estimation_loss, reconstruction_loss, map_loss, kld_loss = self.actor.compute_vae_loss(
                    batch.observations[:original_batch_size], batch.next_observations
                )
                vae_loss = (
                    self.explicit_estimation_loss_weight * explicit_estimation_loss
                    + self.reconstruction_loss_weight * reconstruction_loss
                    + self.map_loss_weight * map_loss
                    + self.vae_kld_weight * kld_loss
                )

                self.vae_optimizer.zero_grad()
                vae_loss.backward()

                mean_explicit_estimation_loss += explicit_estimation_loss.item()
                mean_reconstruction_loss += reconstruction_loss.item()
                mean_map_loss += map_loss.item()
                mean_kld_loss += kld_loss.item()

            # Collect gradients from all GPUs
            if self.is_multi_gpu:
                self.reduce_parameters()

            # Apply the gradients for PPO
            nn.utils.clip_grad_norm_(self.actor_parameters, self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            nn.utils.clip_grad_norm_(self.actor.vae.parameters(), self.max_grad_norm)
            self.vae_optimizer.step()
            # Apply the gradients for RND
            if self.rnd_optimizer:
                self.rnd_optimizer.step()
            # Apply the gradients for AMP
            self.disc_optimizer.step()
            # Update the AMP normalizer
            self.amp_discriminator.update_normalization(disc_obs_batch)

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
            # AMP discriminator loss and other info
            mean_disc_loss += disc_loss.item()
            mean_disc_grad_penalty += disc_grad_penalty.item()
            mean_disc_score += disc_score.mean().item()
            mean_disc_demo_score += disc_demo_score.mean().item()

        # Divide the losses by the number of updates
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_explicit_estimation_loss /= (num_updates * self.num_encoder_epochs)
        mean_reconstruction_loss /= (num_updates * self.num_encoder_epochs)
        mean_map_loss /= (num_updates * self.num_encoder_epochs)
        mean_kld_loss /= (num_updates * self.num_encoder_epochs)
        if mean_rnd_loss is not None:
            mean_rnd_loss /= num_updates
        if mean_symmetry_loss is not None:
            mean_symmetry_loss /= num_updates
        mean_disc_loss /= num_updates
        mean_disc_grad_penalty /= num_updates
        mean_disc_score /= num_updates
        mean_disc_demo_score /= num_updates

        # Clear the storage
        self.storage.clear()

        # Construct the loss dictionary
        loss_dict = {
            "value": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "explicit_estimation": mean_explicit_estimation_loss,
            "reconstruction": mean_reconstruction_loss,
            "map": mean_map_loss,
            "kld": mean_kld_loss,
        }
        if self.rnd:
            loss_dict["rnd"] = mean_rnd_loss
        if self.symmetry:
            loss_dict["symmetry"] = mean_symmetry_loss
        loss_dict["disc_loss"] = mean_disc_loss
        loss_dict["disc_grad_penalty"] = mean_disc_grad_penalty
        loss_dict["disc_score"] = mean_disc_score
        loss_dict["disc_demo_score"] = mean_disc_demo_score

        return loss_dict

    def train_mode(self) -> None:
        """Set train mode for learnable models."""
        self.actor.train()
        self.critic.train()
        self.amp_discriminator.train()
        if self.rnd:
            self.rnd.train()

    def eval_mode(self) -> None:
        """Set evaluation mode for learnable models."""
        self.actor.eval()
        self.critic.eval()
        self.amp_discriminator.eval()
        if self.rnd:
            self.rnd.eval()

    def save(self) -> dict:
        """Return a dict of all models for saving."""
        saved_dict = {
            "actor_state_dict": self.actor.state_dict(),
            "critic_state_dict": self.critic.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "vae_optimizer_state_dict": self.vae_optimizer.state_dict(),
            "amp_state_dict": self.amp_discriminator.state_dict(),
            "amp_optimizer_state_dict": self.disc_optimizer.state_dict(),
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
                "amp": True,
            }

        # Load the specified models
        if load_cfg.get("actor"):
            self.actor.load_state_dict(loaded_dict["actor_state_dict"], strict=strict)
        if load_cfg.get("critic"):
            self.critic.load_state_dict(loaded_dict["critic_state_dict"], strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
            self.vae_optimizer.load_state_dict(loaded_dict["vae_optimizer_state_dict"])
        if load_cfg.get("rnd") and self.rnd:
            self.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=strict)
            self.rnd_optimizer.load_state_dict(loaded_dict["rnd_optimizer_state_dict"])
        if load_cfg.get("amp"):
            self.amp_discriminator.load_state_dict(loaded_dict["amp_state_dict"])
            self.disc_optimizer.load_state_dict(loaded_dict["amp_optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def get_policy(self) -> PIEModel:
        """Get the policy model."""
        return self.actor

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPOPIE:
        """Construct the PPOAMP algorithm."""
        # Resolve class callables
        alg_class: type[PPOAMP] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore
        actor_class: type[PIEModel] = resolve_callable(cfg["actor"].pop("class_name"))  # type: ignore
        critic_class: type[MLPModel] = resolve_callable(cfg["critic"].pop("class_name"))  # type: ignore

        # Resolve observation groups
        default_sets = ["actor", "critic"]
        if "rnd_cfg" in cfg["algorithm"] and cfg["algorithm"]["rnd_cfg"] is not None:
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        # Resolve RND config if used
        cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Resolve AMP config
        cfg["algorithm"] = resolve_amp_config(cfg["algorithm"], obs, cfg["obs_groups"], env)

        # Initialize the policy
        actor: PIEModel = actor_class(obs, cfg["obs_groups"], "actor", "hist", "explicit", "next", "map", "depth", env.num_actions, **cfg["actor"]).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):  # Share CNN encoders between actor and critic
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore
        critic: MLPModel = critic_class(obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]).to(device)
        print(f"Critic Model: {critic}")

        # Initialize the storage
        storage = RolloutStorageVAE("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        # Initialize AMP discriminator observation buffers
        disc_obs_buffer = CircularBuffer(
            max_len=cfg["algorithm"]["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=env.num_envs,
            device=device,
        )
        disc_demo_obs_buffer = CircularBuffer(
            max_len=cfg["algorithm"]["amp_cfg"]["disc_obs_buffer_size"],
            batch_size=env.num_envs,
            device=device,
        )

        # Initialize the algorithm
        alg: PPOPIE = alg_class(
            actor,
            critic,
            storage,
            disc_obs_buffer,
            disc_demo_obs_buffer,
            cfg["obs_groups"],
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

        return alg

    def broadcast_parameters(self) -> None:
        """Broadcast model parameters to all GPUs."""
        # Obtain the model parameters on current GPU
        model_params = [self.actor.state_dict(), self.critic.state_dict(), self.amp_discriminator.state_dict()]
        if self.rnd:
            model_params.append(self.rnd.predictor.state_dict())
        # Broadcast the model parameters
        torch.distributed.broadcast_object_list(model_params, src=0)
        # Load the model parameters on all GPUs from source GPU
        self.actor.load_state_dict(model_params[0])
        self.critic.load_state_dict(model_params[1])
        self.amp_discriminator.load_state_dict(model_params[2])
        if self.rnd:
            self.rnd.predictor.load_state_dict(model_params[3])

    def reduce_parameters(self) -> None:
        """Collect gradients from all GPUs and average them.

        This function is called after the backward pass to synchronize the gradients across all GPUs.
        """
        # Create a tensor to store the gradients
        all_params = chain(self.actor.parameters(), self.critic.parameters(), self.amp_discriminator.parameters())
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
