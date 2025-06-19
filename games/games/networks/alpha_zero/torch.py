import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from pathlib import Path
from dataclasses import dataclass
from games.utils import load_json
from pyspiel import load_game


DEVICE = "cpu"  # "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class InputBlockConfig:
    input_channels: int
    filters: int
    kernel_size: int
    padding: int


class ResInputBlock(nn.Module):
    def __init__(self, config: InputBlockConfig):
        super(ResInputBlock, self).__init__()

        self.input_conv = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.filters,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.padding,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.input_batch_norm = nn.BatchNorm2d(
            num_features=config.filters,
            eps=0.001,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, x):
        x = self.input_conv(x)
        x = self.input_batch_norm(x)
        x = torch.relu(x)
        return x


class ResTorsoBlock(nn.Module):
    def __init__(self, config: InputBlockConfig):
        super().__init__()

        self.conv_1 = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.filters,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.padding,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.conv_2 = nn.Conv2d(
            in_channels=config.filters,
            out_channels=config.filters,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.padding,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.batch_norm_1 = nn.BatchNorm2d(
            num_features=config.filters,
            eps=0.001,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )

        self.batch_norm_2 = nn.BatchNorm2d(
            num_features=config.filters,
            eps=0.001,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )

    def forward(self, x):
        residual = x
        output = F.relu(self.batch_norm_1(self.conv_1(x)))
        output = self.batch_norm_2(self.conv_2(output))
        output += residual
        output = F.relu(output)

        return output


@dataclass
class ResOutputBlockConfig:
    input_channels: int
    value_filters: int
    kernel_size: int
    padding: int
    value_linear_in_features: int
    value_linear_out_features: int
    value_observation_size: int
    policy_filters: int
    policy_linear_in_features: int
    policy_linear_out_features: int
    policy_observation_size: int


class ResOutputBlock(nn.Module):
    def __init__(self, config: ResOutputBlockConfig):
        super().__init__()

        self.value_conv_ = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.value_filters,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.padding,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.value_batch_norm_ = nn.BatchNorm2d(
            num_features=config.value_filters,
            eps=0.001,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )

        self.value_linear1_ = nn.Linear(
            in_features=config.value_linear_in_features, out_features=config.value_linear_out_features, bias=True
        )

        self.value_linear2_ = nn.Linear(in_features=config.value_linear_out_features, out_features=1, bias=True)

        self.value_observation_size_ = config.value_observation_size

        self.policy_conv_ = nn.Conv2d(
            in_channels=config.input_channels,
            out_channels=config.policy_filters,
            kernel_size=config.kernel_size,
            stride=1,
            padding=config.padding,
            dilation=1,
            groups=1,
            bias=True,
            padding_mode="zeros",
        )

        self.policy_batch_norm_ = nn.BatchNorm2d(
            num_features=config.policy_filters,
            eps=0.001,
            momentum=0.01,
            affine=True,
            track_running_stats=True,
        )

        self.policy_linear_ = nn.Linear(
            in_features=config.policy_linear_in_features, out_features=config.policy_linear_out_features, bias=True
        )

        self.policy_observation_size_ = config.policy_observation_size

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        value_output = F.relu(self.value_batch_norm_(self.value_conv_(x)))
        value_output = value_output.view(-1, self.value_observation_size_)
        value_output = F.relu(self.value_linear1_(value_output))
        value_output = torch.tanh(self.value_linear2_(value_output))

        policy_logits = F.relu(self.policy_batch_norm_(self.policy_conv_(x)))
        policy_logits = policy_logits.view(-1, self.policy_observation_size_)
        policy_logits = self.policy_linear_(policy_logits)
        if mask is not None:
            mask = torch.Tensor(mask).bool()
            policy_logits = torch.where(mask, policy_logits, -(1 << 16) * torch.ones_like(policy_logits))

        return value_output, policy_logits


@dataclass
class ModelConfig:
    nn_depth: int
    weight_decay: float
    observation_tensor_shape: tuple[int]
    nn_width: int
    number_of_actions: int


class AZModel(nn.Module):
    def __init__(self, config: ModelConfig, device: str):
        super().__init__()
        self.device = device
        self.num_torso_blocks = config.nn_depth
        self.weight_decay = config.weight_decay
        self.observation_tensor_shape = (-1,) + config.observation_tensor_shape

        input_config = InputBlockConfig(
            input_channels=config.observation_tensor_shape[0],
            filters=config.nn_width,
            kernel_size=3,
            padding=1,
        )

        torso_config = InputBlockConfig(
            input_channels=config.nn_width, filters=config.nn_width, kernel_size=3, padding=1
        )

        output_config = ResOutputBlockConfig(
            input_channels=config.nn_width,
            value_filters=1,
            policy_filters=2,
            kernel_size=1,
            padding=0,
            value_linear_in_features=config.observation_tensor_shape[1] * config.observation_tensor_shape[2],
            value_linear_out_features=config.nn_width,
            policy_linear_in_features=2 * config.observation_tensor_shape[1] * config.observation_tensor_shape[2],
            policy_linear_out_features=config.number_of_actions,
            value_observation_size=config.observation_tensor_shape[1] * config.observation_tensor_shape[2],
            policy_observation_size=2 * config.observation_tensor_shape[1] * config.observation_tensor_shape[2],
        )

        self.layers = nn.ModuleList()
        self.layers.append(ResInputBlock(input_config))
        for _ in range(self.num_torso_blocks):
            self.layers.append(ResTorsoBlock(torso_config))

        self.layers.append(ResOutputBlock(output_config))

    def forward(self, x, legals_mask=None):
        x = torch.Tensor(x).reshape(self.observation_tensor_shape)

        for layer in self.layers[:-1]:
            x = layer(x)

        value_output, policy_logits = self.layers[-1](x, legals_mask)
        priors = torch.softmax(policy_logits, 1)
        return value_output, priors

    def get_value(self, x, legals_mask):
        value, _ = self(x, legals_mask)
        return value

    def get_prior(self, x, legals_mask):
        _, prior = self(x, legals_mask)
        return prior


def load_az_model(path: Path, checkpoint_nr: Optional[int] = None) -> AZModel:
    config = _load_config(path)
    model = AZModel(config, DEVICE)
    if checkpoint_nr is None:
        model_weights = _load_last_checkpoint(path)
    else:
        model_weights = _load_checkpoint(path, checkpoint_nr)

    # The names are defined in a way that is hard to replicate in Python. Especially for the residual layers, which are
    # defined as f"layers.{n}.res_{n-1}_{component_info}". As I don't want to based the name of the attribute of a layer
    # dynamically, based on the layer number, I am renaming the keys of the model weights to initialize the model.
    state_dict_right_keys = {k: v for k, v in zip(model.state_dict().keys(), model_weights.state_dict().values())}
    model.load_state_dict(state_dict_right_keys)
    return model


def _load_config(path: Path) -> ModelConfig:
    config_dict = load_config(path=path)
    game = load_game(config_dict["game"])

    model_config = ModelConfig(
        nn_depth=config_dict["nn_depth"],
        weight_decay=config_dict["weight_decay"],
        observation_tensor_shape=tuple(game.observation_tensor_shape()),
        nn_width=config_dict["nn_width"],
        number_of_actions=game.num_distinct_actions(),
    )

    return model_config


def load_config(path: Path) -> dict:
    return load_json(path / "config.json")


def _load_checkpoint(path: Path, checkpoint_nr: int):
    last_checkpoint_path = path / f"checkpoint-{checkpoint_nr}.pt"
    model_weights = torch.load(last_checkpoint_path)
    return model_weights


def _load_last_checkpoint(path: Path):
    last_checkpoint_path = path / "checkpoint--1.pt"
    model_weights = torch.load(last_checkpoint_path)
    return model_weights
