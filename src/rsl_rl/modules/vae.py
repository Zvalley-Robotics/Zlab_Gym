import torch
import torch.nn as nn

from .mlp import MLP


class MLPVae(nn.Module):
    """A simple MLP-based VAE."""

    def __init__(
        self,
        encoder_input_dim: int,
        encoder_output_dim: int,
        explicit_dim: int,
        latent_dim: int,
        decoder_output_dim: int,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
    ):
        super().__init__()
        self.encoder = MLP(encoder_input_dim, encoder_output_dim, encoder_hidden_dims, activation, last_activation=activation)
        self.explicit_mu = nn.Linear(encoder_output_dim, explicit_dim)
        self.explicit_var = nn.Linear(encoder_output_dim, explicit_dim)
        self.latent_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.latent_var = nn.Linear(encoder_output_dim, latent_dim)
        self.decoder = MLP(explicit_dim + latent_dim, decoder_output_dim, decoder_hidden_dims, activation)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoded = self.encoder(x)
        explicit_mu = self.explicit_mu(encoded)
        explicit_var = self.explicit_var(encoded)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        code_explicit = self.reparameterize(explicit_mu, explicit_var)
        code_latent = self.reparameterize(latent_mu, latent_var)
        decode = self.decoder(torch.cat((code_explicit, code_latent), dim=-1))
        return code_explicit, code_latent, decode, explicit_mu, explicit_var, latent_mu, latent_var


class MLPPie(nn.Module):
    """A simple MLP-based PIE."""

    def __init__(
        self,
        encoder_input_dim: int,
        encoder_output_dim: int,
        explicit_dim: int,
        latent_dim: int,
        map_latent_dim: int,
        depth_dim: int,
        decoder_output_dim: int,
        map_decoder_output_dim: int,
        encoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        map_decoder_hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
    ):
        super().__init__()
        self.encoder = MLP(encoder_input_dim, encoder_output_dim, encoder_hidden_dims, activation, last_activation=activation)
        self.mlp = MLP(encoder_output_dim + depth_dim, encoder_output_dim, encoder_hidden_dims, activation, last_activation=activation)
        self.explicit_mu = nn.Linear(encoder_output_dim, explicit_dim)
        self.explicit_var = nn.Linear(encoder_output_dim, explicit_dim)
        self.latent_mu = nn.Linear(encoder_output_dim, latent_dim)
        self.latent_var = nn.Linear(encoder_output_dim, latent_dim)
        self.map_mu = nn.Linear(encoder_output_dim, map_latent_dim)
        self.map_var = nn.Linear(encoder_output_dim, map_latent_dim)
        self.decoder = MLP(explicit_dim + latent_dim + map_latent_dim, decoder_output_dim, decoder_hidden_dims, activation)
        self.map_decoder = MLP(map_latent_dim, map_decoder_output_dim, map_decoder_hidden_dims, activation)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        :param mu: (Tensor) Mean of the latent Gaussian
        :param logvar: (Tensor) Standard deviation of the latent Gaussian
        :return:
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

    def forward(self, x: torch.Tensor, depth: torch.Tensor) -> tuple[torch.Tensor, ...]:
        encoded = self.mlp(torch.cat((self.encoder(x), depth), dim=-1))
        explicit_mu = self.explicit_mu(encoded)
        explicit_var = self.explicit_var(encoded)
        latent_mu = self.latent_mu(encoded)
        latent_var = self.latent_var(encoded)
        map_mu = self.map_mu(encoded)
        map_var = self.map_var(encoded)
        code_explicit = self.reparameterize(explicit_mu, explicit_var)
        code_latent = self.reparameterize(latent_mu, latent_var)
        code_map = self.reparameterize(map_mu, map_var)
        decode = self.decoder(torch.cat((code_explicit, code_latent, code_map), dim=-1))
        map_decode = self.map_decoder(code_map)
        return code_explicit, code_latent, code_map, decode, map_decode, explicit_mu, explicit_var, latent_mu, latent_var, map_mu, map_var
