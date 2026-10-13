# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.modules import MLP, EmpiricalNormalization, HiddenState
from rsl_rl.modules.distribution import Distribution
from rsl_rl.utils import resolve_callable, unpad_trajectories
from src.rsl_rl.modules import MLPPie
from src.rsl_rl.modules import Conv2dHead


class PIEModel(nn.Module):
    """PIE-based neural model."""

    is_recurrent: bool = False
    """Whether the model contains a recurrent module."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        hist_set: str,
        explicit_set: str,
        next_set: str,
        map_set: str,
        depth_set: str,
        output_dim: int,
        encoder_output_dim: int = 64,
        vae_latent_dim: int = 16,
        map_latent_dim: int = 16,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        map_decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        depth_cfg: dict | None = None,
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Initialize the MLP-based model.

        Args:
            obs: Observation Dictionary.
            obs_groups: Dictionary mapping observation sets to lists of observation groups.
            obs_set: Observation set to use for this model (e.g., "actor" or "critic").
            output_dim: Dimension of the output.
            hidden_dims: Hidden dimensions of the MLP.
            activation: Activation function of the MLP.
            obs_normalization: Whether to normalize the observations before feeding them to the MLP.
            distribution_cfg: Configuration dictionary for the output distribution. If provided, the model outputs
                stochastic values sampled from the distribution.
        """
        super().__init__()

        # Resolve observation groups and dimensions
        self.obs_groups, self.obs_dim = self._get_obs_dim(obs, obs_groups, obs_set)
        self.hist_groups, self.hist_dim = self._get_obs_dim(obs, obs_groups, hist_set)
        self.explicit_groups, self.explicit_dim = self._get_obs_dim(obs, obs_groups, explicit_set)
        self.next_groups, self.next_dim = self._get_obs_dim(obs, obs_groups, next_set)
        self.map_groups, self.map_dim = self._get_obs_dim(obs, obs_groups, map_set)
        self.depth_group, self.depth_shape = self._get_image_shape(obs, obs_groups, depth_set)

        # Observation normalization
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(self.obs_dim)
            self.hist_normalizer = EmpiricalNormalization(self.hist_dim)
            self.explicit_normalizer = EmpiricalNormalization(self.explicit_dim)
            self.next_normalizer = EmpiricalNormalization(self.next_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()
            self.hist_normalizer = torch.nn.Identity()
            self.explicit_normalizer = torch.nn.Identity()
            self.next_normalizer = torch.nn.Identity()

        # Distribution
        if distribution_cfg is not None:
            dist_class: type[Distribution] = resolve_callable(distribution_cfg.pop("class_name"))  # type: ignore
            self.distribution: Distribution | None = dist_class(output_dim, **distribution_cfg)
            mlp_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            mlp_output_dim = output_dim

        # HEAD
        if depth_cfg is None:
            raise ValueError("DEPTH configuration must be provided.")
        output_dim = depth_cfg.pop("output_dim")
        hidden_dims = depth_cfg.pop("hidden_dims") + (output_dim,)
        self.depth_head = Conv2dHead(
            self.depth_shape,
            hidden_dims=hidden_dims,
            output_dim=None,
            **depth_cfg,
        )

        # VAE
        self.vae = MLPPie(
            self.hist_dim,
            encoder_output_dim,
            self.explicit_dim,
            vae_latent_dim,
            map_latent_dim,
            self.depth_head.output_dim,
            self.next_dim,
            self.map_dim,
            encoder_hidden_dims,
            decoder_hidden_dims,
            map_decoder_hidden_dims,
            activation,
        )

        # MLP
        mlp_input_dim = self._get_latent_dim() + self.explicit_dim + vae_latent_dim + map_latent_dim
        self.mlp = MLP(mlp_input_dim, mlp_output_dim, hidden_dims, activation)

        # Initialize distribution-specific VAE weights
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state: HiddenState = None,
        stochastic_output: bool = False,
    ) -> torch.Tensor:
        """Forward pass of the PIE model.

        ..note::
            The `stochastic_output` flag only has an effect if the model has a distribution (i.e., ``distribution_cfg``
            was provided) and defaults to ``False``, meaning that even stochastic models will return deterministic
            outputs by default.
        """
        # If observations are padded for recurrent training but the model is non-recurrent, unpad the observations
        obs = unpad_trajectories(obs, masks) if masks is not None and not self.is_recurrent else obs
        # Get MLP input latent
        latent = self.get_latent(obs, masks, hidden_state)
        # MLP forward pass
        mlp_output = self.mlp(latent)
        # If stochastic output is requested, update the distribution and sample from it, otherwise return MLP output
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(mlp_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(mlp_output)
        return mlp_output

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Build the model latent by concatenating and normalizing selected observation groups."""
        # Select and concatenate observations
        mlp = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        hist = torch.cat([obs[hist_group] for hist_group in self.hist_groups], dim=-1)
        depth = obs[self.depth_group]
        # Normalize observations
        mlp = self.obs_normalizer(mlp)
        hist = self.hist_normalizer(hist)
        depth = self.depth_head(depth)
        explicit, latent, map, _, _, _, _, _, _, _, _, = self.vae(hist, depth)
        return torch.cat((mlp, explicit, latent, map), dim=1)

    def reset(self, dones: torch.Tensor | None = None, hidden_state: HiddenState = None) -> None:
        """Reset the internal state for recurrent models (no-op)."""
        pass

    def get_hidden_state(self) -> HiddenState:
        """Return the recurrent hidden state (``None`` for MLP)."""
        return None

    def detach_hidden_state(self, dones: torch.Tensor | None = None) -> None:
        """Detach therecurrent hidden state for truncated backpropagation (no-op)."""
        pass

    @property
    def output_mean(self) -> torch.Tensor:
        """Return the mean of the current output distribution."""
        return self.distribution.mean

    @property
    def output_std(self) -> torch.Tensor:
        """Return the standard deviation of the current output distribution."""
        return self.distribution.std

    @property
    def output_entropy(self) -> torch.Tensor:
        """Return the entropy of the current output distribution."""
        return self.distribution.entropy

    @property
    def output_distribution_params(self) -> tuple[torch.Tensor, ...]:
        """Return raw parameters of the current output distribution."""
        return self.distribution.params

    def get_output_log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        """Compute log-probabilities of outputs under the current distribution."""
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(
        self, old_params: tuple[torch.Tensor, ...], new_params: tuple[torch.Tensor, ...]
    ) -> torch.Tensor:
        """Compute KL divergence between two parameterizations of the distribution."""
        return self.distribution.kl_divergence(old_params, new_params)

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchPIEModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxPIEModel(self, verbose)

    def update_normalization(self, obs: TensorDict, next_obs: TensorDict) -> None:
        """Update observation-normalization statistics from a batch of observations."""
        if self.obs_normalization:
            # Select and concatenate observations
            mlp_obs = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
            hist_obs = torch.cat([obs[hist_group] for hist_group in self.hist_groups], dim=-1)
            explicit = torch.cat([obs[explicit_group] for explicit_group in self.explicit_groups], dim=-1)
            next_state = torch.cat([next_obs[next_group] for next_group in self.next_groups], dim=-1)
            # Update the normalizer parameters
            self.obs_normalizer.update(mlp_obs)  # type: ignore
            self.hist_normalizer.update(hist_obs)  # type: ignore
            self.explicit_normalizer.update(explicit)  # type: ignore
            self.next_normalizer.update(next_state)  # type: ignore

    def compute_vae_loss(self, obs: TensorDict, next_obs: TensorDict) -> tuple[float, float, float]:
        hist = self.hist_normalizer(torch.cat([obs[hist_group] for hist_group in self.hist_groups], dim=-1))
        explicit = self.explicit_normalizer(torch.cat([obs[explicit_group] for explicit_group in self.explicit_groups], dim=-1))
        next = self.next_normalizer(torch.cat([next_obs[next_group] for next_group in self.next_groups], dim=-1))
        map = torch.cat([obs[map_group] for map_group in self.map_groups], dim=-1)
        depth = self.depth_head(obs[self.depth_group])
        code_explicit, _, _, decode, map_decode, _, _, latent_mu, latent_var, _, _ = self.vae(hist, depth)
        explicit_estimation_loss = nn.functional.mse_loss(code_explicit, explicit)
        reconstruction_loss = nn.functional.mse_loss(decode, next)
        map_loss = nn.functional.mse_loss(map_decode, map)
        kld_loss = -0.5 * torch.mean(torch.sum(1 + latent_var - latent_mu ** 2 - latent_var.exp(), dim=1))
        return explicit_estimation_loss, reconstruction_loss, map_loss, kld_loss

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"The MLP model only supports 1D observations, got shape {obs[obs_group].shape} for '{obs_group}'."
                )
            obs_dim += obs[obs_group].shape[-1]
        return active_obs_groups, obs_dim

    def _get_image_shape(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[str, list]:
        """Select active observation groups and compute observation dimension."""
        active_obs_groups = obs_groups[obs_set]
        assert len(active_obs_groups) == 1, "The image only supports one observation."
        active_obs_group = active_obs_groups[0]
        if len(obs[active_obs_group].shape) != 4:
            raise ValueError(
                f"The image only supports 2D observations, got shape {obs[active_obs_group].shape} for '{active_obs_group}'."
            )
        return active_obs_group, obs[active_obs_group].shape[1:4]

    def _get_latent_dim(self) -> int:
        """Return the latent dimensionality consumed by the MLP head."""
        return self.obs_dim


class _TorchPIEModel(nn.Module):
    """Exportable PIE model for JIT."""

    def __init__(self, model: PIEModel) -> None:
        """Create a TorchScript-friendly copy of an PIEModel."""
        super().__init__()
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.hist_normalizer = copy.deepcopy(model.hist_normalizer)
        self.vae = copy.deepcopy(model.vae)
        self.mlp = copy.deepcopy(model.mlp)
        self.depth_head = copy.deepcopy(model.depth_head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor, hist: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference on pre-concatenated observations."""
        x = self.obs_normalizer(x)
        hist = self.hist_normalizer(hist)
        depth = self.depth_head(depth)
        explicit, latent, map, _, _, _, _, _, _, _, _ = self.vae(hist, depth)
        out = self.mlp(torch.cat((x, explicit, latent, map), dim=1))
        return self.deterministic_output(out)

    @torch.jit.export
    def reset(self) -> None:
        """Reset recurrent export state (no-op for PIE exports)."""
        pass


class _OnnxPIEModel(nn.Module):
    """Exportable PIE model for ONNX."""

    is_recurrent: bool = False

    def __init__(self, model: PIEModel, verbose: bool) -> None:
        """Create an ONNX-export wrapper around an PIEModel."""
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.hist_normalizer = copy.deepcopy(model.hist_normalizer)
        self.vae = copy.deepcopy(model.vae)
        self.mlp = copy.deepcopy(model.mlp)
        self.depth_head = copy.deepcopy(model.depth_head)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()
        self.input_size = model.obs_dim
        self.hist_size = model.hist_dim
        self.depth_shape = model.depth_shape

    def forward(self, x: torch.Tensor, hist: torch.Tensor, depth: torch.Tensor) -> torch.Tensor:
        """Run deterministic inference for ONNX export."""
        x = self.obs_normalizer(x)
        hist = self.hist_normalizer(hist)
        depth = self.depth_head(depth)
        explicit, latent, map, _, _, _, _, _, _, _, _ = self.vae(hist, depth)
        out = self.mlp(torch.cat((x, explicit, latent, map), dim=1))
        return self.deterministic_output(out)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        """Return representative dummy inputs for ONNX tracing."""
        return (torch.zeros(1, self.input_size), torch.zeros(1, self.hist_size), torch.zeros(1, *self.depth_shape))

    @property
    def input_names(self) -> list[str]:
        """Return ONNX input tensor names."""
        return ["obs", "hist", "depth"]

    @property
    def output_names(self) -> list[str]:
        """Return ONNX output tensor names."""
        return ["actions"]
