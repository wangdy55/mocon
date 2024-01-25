import torch
import torch.nn as nn
import torch.nn.functional as F

# primitive motion VAE
class MotionVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        h1 = 256
        # Encoder
        # Takes pose | condition (n * poses) as input
        self.fc1 = nn.Linear(
            frame_size * (num_future_predictions + num_condition_frames), h1
        )
        self.fc2 = nn.Linear(frame_size + h1, h1)
        # self.fc3 = nn.Linear(h1, h1)
        self.mu = nn.Linear(frame_size + h1, latent_size)
        self.logvar = nn.Linear(frame_size + h1, latent_size)

        # Decoder
        # Takes latent | condition as input
        self.fc4 = nn.Linear(latent_size + frame_size * num_condition_frames, h1)
        self.fc5 = nn.Linear(latent_size + h1, h1)
        # self.fc6 = nn.Linear(latent_size + h1, h1)
        self.out = nn.Linear(latent_size + h1, num_future_predictions * frame_size)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        # h3 = F.elu(self.fc3(h2))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        # h6 = F.elu(self.fc6(torch.cat((z, h5), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def sample(self, z, c, deterministic=False):
        return self.decode(z, c)

class Encoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Encoder
        # Takes pose | condition (n * poses) as input
        input_size = frame_size * (num_future_predictions + num_condition_frames)
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(frame_size + hidden_size, hidden_size)
        self.mu = nn.Linear(frame_size + hidden_size, latent_size)
        self.logvar = nn.Linear(frame_size + hidden_size, latent_size)

    def encode(self, x, c):
        h1 = F.elu(self.fc1(torch.cat((x, c), dim=1)))
        h2 = F.elu(self.fc2(torch.cat((x, h1), dim=1)))
        s = torch.cat((x, h2), dim=1)
        return self.mu(s), self.logvar(s)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x, c):
        mu, logvar = self.encode(x, c)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

class Decoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
    ):
        super().__init__()
        # Decoder
        # Takes latent | condition as input
        input_size = latent_size + frame_size * num_condition_frames
        output_size = num_future_predictions * frame_size
        self.fc4 = nn.Linear(input_size, hidden_size)
        self.fc5 = nn.Linear(latent_size + hidden_size, hidden_size)
        self.out = nn.Linear(latent_size + hidden_size, output_size)

    def decode(self, z, c):
        h4 = F.elu(self.fc4(torch.cat((z, c), dim=1)))
        h5 = F.elu(self.fc5(torch.cat((z, h4), dim=1)))
        return self.out(torch.cat((z, h5), dim=1))

    def forward(self, z, c):
        return self.decode(z, c)

class MixedDecoder(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        hidden_size,
        num_condition_frames,
        num_future_predictions,
        num_experts,
    ):
        super().__init__()

        input_size = latent_size + frame_size * num_condition_frames
        inter_size = latent_size + hidden_size
        output_size = num_future_predictions * frame_size
        self.decoder_layers = [
            (
                nn.Parameter(torch.empty(num_experts, input_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, hidden_size)),
                nn.Parameter(torch.empty(num_experts, hidden_size)),
                F.elu,
            ),
            (
                nn.Parameter(torch.empty(num_experts, inter_size, output_size)),
                nn.Parameter(torch.empty(num_experts, output_size)),
                None,
            ),
        ]

        for index, (weight, bias, _) in enumerate(self.decoder_layers):
            index = str(index)
            torch.nn.init.kaiming_uniform_(weight)
            bias.data.fill_(0.01)
            self.register_parameter("w" + index, weight)
            self.register_parameter("b" + index, bias)

        # Gating network
        gate_hsize = 64
        self.gate = nn.Sequential(
            nn.Linear(input_size, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, gate_hsize),
            nn.ELU(),
            nn.Linear(gate_hsize, num_experts),
        )

    def forward(self, z, c):
        coefficients = F.softmax(self.gate(torch.cat((z, c), dim=1)), dim=1)
        layer_out = c

        for (weight, bias, activation) in self.decoder_layers:
            flat_weight = weight.flatten(start_dim=1, end_dim=2)
            mixed_weight = torch.matmul(coefficients, flat_weight).view(
                coefficients.shape[0], *weight.shape[1:3]
            )

            input = torch.cat((z, layer_out), dim=1).unsqueeze(1)
            mixed_bias = torch.matmul(coefficients, bias).unsqueeze(1)
            out = torch.baddbmm(mixed_bias, input, mixed_weight).squeeze(1)
            layer_out = activation(out) if activation is not None else out

        return layer_out

class MotionMixtureVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 256
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)
        self.decoder = MixedDecoder(*args, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def encode(self, x, c):
        _, mu, logvar = self.encoder(x, c)
        return mu, logvar

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        return self.decoder(z, c), mu, logvar

    def sample(self, z, c, deterministic=False):
        return self.decoder(z, c)

class MotionMixtureSpecialistVAE(nn.Module):
    def __init__(
        self,
        frame_size,
        latent_size,
        num_condition_frames,
        num_future_predictions,
        normalization,
        num_experts,
    ):
        super().__init__()
        self.frame_size = frame_size
        self.latent_size = latent_size
        self.num_condition_frames = num_condition_frames
        self.num_future_predictions = num_future_predictions

        self.mode = normalization.get("mode")
        self.data_max = normalization.get("max")
        self.data_min = normalization.get("min")
        self.data_avg = normalization.get("avg")
        self.data_std = normalization.get("std")

        hidden_size = 128
        args = (
            frame_size,
            latent_size,
            hidden_size,
            num_condition_frames,
            num_future_predictions,
        )

        self.encoder = Encoder(*args)

        self.decoders = []
        for i in range(num_experts):
            decoder = Decoder(*args)
            self.decoders.append(decoder)
            self.add_module("d" + str(i), decoder)

        # Gating network
        gate_hsize = 128
        input_size = latent_size + frame_size * num_condition_frames
        self.g_fc1 = nn.Linear(input_size, gate_hsize)
        self.g_fc2 = nn.Linear(latent_size + gate_hsize, gate_hsize)
        self.g_fc3 = nn.Linear(latent_size + gate_hsize, num_experts)

    def normalize(self, t):
        if self.mode == "minmax":
            return 2 * (t - self.data_min) / (self.data_max - self.data_min) - 1
        elif self.mode == "zscore":
            return (t - self.data_avg) / self.data_std
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def denormalize(self, t):
        if self.mode == "minmax":
            return (t + 1) * (self.data_max - self.data_min) / 2 + self.data_min
        elif self.mode == "zscore":
            return t * self.data_std + self.data_avg
        elif self.mode == "none":
            return t
        else:
            raise ValueError("Unknown normalization mode")

    def gate(self, z, c):
        h1 = F.elu(self.g_fc1(torch.cat((z, c), dim=1)))
        h2 = F.elu(self.g_fc2(torch.cat((z, h1), dim=1)))
        return self.g_fc3(torch.cat((z, h2), dim=1))

    def forward(self, x, c):
        z, mu, logvar = self.encoder(x, c)
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)
        return predictions, mu, logvar, coefficients

    def sample(self, z, c, deterministic=False):
        coefficients = F.softmax(self.gate(z, c), dim=1)
        predictions = torch.stack([decoder(z, c) for decoder in self.decoders], dim=1)

        if not deterministic:
            dist = torch.distributions.Categorical(coefficients)
            indices = dist.sample()
        else:
            indices = coefficients.argmax(dim=1)

        return predictions[torch.arange(predictions.size(0)), indices]
