from typing import NamedTuple

import math
import torch
from torch.nn import Linear, ReLU, Sequential, LayerNorm, Conv2d, BatchNorm2d, Softmax, Module, Flatten, init
import torch.nn.functional as F


class ActorMLP(Module):
    def __init__(self, input_dim, output_dims, hidden_dim, min_val=torch.finfo(torch.float).min, activation=ReLU):
        super(ActorMLP, self).__init__()
        self.min_val = min_val
        self.output_dims = output_dims

        self.actor = Sequential(
            Linear(input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            activation(),
            Linear(hidden_dim, hidden_dim),
            activation(),
            Linear(hidden_dim, hidden_dim),
            activation(),
        )

        self.out_layers = []
        for i, output_dim in enumerate(output_dims):
            self.out_layers.append(Linear(hidden_dim, output_dim))

        self.softmax = Softmax(dim=1)

    # Dict input
    def forward(self, observations, state, info={}):

        x = observations["obs"]
        x = self.actor(x)

        # if we are in inference mode, mask is optional:
        # if observations.get('mask') is not None:
        #     action_masks = observations['mask']
        #     x[~action_masks] = self.min_val
        #     x = self.softmax(x)

        outs = []
        for out_layer in self.out_layers:
            outs.append(self.softmax(out_layer(x)).unsqueeze(dim=1))

        x = torch.cat(outs, dim=1)

        return x, state


class CriticMLP(Module):
    def __init__(self, input_dim, hidden_dim, min_val=torch.finfo(torch.float).min, activation=ReLU):
        super(CriticMLP, self).__init__()
        self.min_val = min_val
        self.critic = Sequential(
            Linear(input_dim, hidden_dim),
            LayerNorm(hidden_dim),
            activation(),
            Linear(hidden_dim, hidden_dim),
            activation(),
            Linear(hidden_dim, hidden_dim),
            activation(),
            Linear(hidden_dim, 1),
        )

    def forward(self, observations, state=None, info={}):

        x = observations["obs"]
        x = self.critic(x)
        return x


def calc_conv2d_output(h_w, kernel_size=1, stride=1, pad=0, dilation=1):
    """takes a tuple of (h,w) and returns a tuple of (h,w)"""

    if not isinstance(kernel_size, tuple):
        kernel_size = (kernel_size, kernel_size)
    h = math.floor(((h_w[0] + (2 * pad) - (dilation * (kernel_size[0] - 1)) - 1) / stride) + 1)
    w = math.floor(((h_w[1] + (2 * pad) - (dilation * (kernel_size[1] - 1)) - 1) / stride) + 1)
    return h, w


def initialize_weights(net: Module) -> None:
    """Initialize weights for Conv2d and Linear layers using kaming initializer."""
    assert isinstance(net, Module)

    for module in net.modules():
        if isinstance(module, (Conv2d, Linear)):
            init.kaiming_uniform_(module.weight, nonlinearity="relu")

            if module.bias is not None:
                init.zeros_(module.bias)


class ResNetBlock(Module):
    """Basic redisual block."""

    def __init__(
        self,
        num_filters: int,
    ) -> None:
        super().__init__()

        self.conv_block1 = Sequential(
            Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(num_features=num_filters),
            ReLU(),
        )

        self.conv_block2 = Sequential(
            Conv2d(
                in_channels=num_filters,
                out_channels=num_filters,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
            BatchNorm2d(num_features=num_filters),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        out = self.conv_block1(x)
        out = self.conv_block2(out)
        out += residual
        out = F.relu(out)
        return out


class ResNet(Module):
    def __init__(
        self,
        input_shape: tuple,
        num_output_features: int,
        num_res_block: int,
        num_filters: int,
        num_fc_units: int,
        output: Module,
        forward_returns_state: bool,
    ) -> None:
        super().__init__()
        c, h, w = input_shape
        self.forward_returns_state = forward_returns_state

        kernel_size = 3
        stride = 1
        num_padding = 1

        self.first_conv_block = Sequential(
            Conv2d(
                in_channels=c,
                out_channels=num_filters,
                kernel_size=kernel_size,
                stride=stride,
                padding=num_padding,
                bias=False,
            ),
            BatchNorm2d(num_features=num_filters),
            ReLU(),
        )

        self.res_blocks = Sequential(*[ResNetBlock(num_filters) for _ in range(num_res_block)])

        conv_out_hw = calc_conv2d_output(h_w=(h, w), kernel_size=kernel_size, stride=stride, pad=num_padding)
        conv_out = conv_out_hw[0] * conv_out_hw[1]

        self.output_head = Sequential(
            Conv2d(
                in_channels=num_filters,
                out_channels=1,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
            BatchNorm2d(num_features=1),
            ReLU(),
            Flatten(),
            Linear(conv_out, num_fc_units),
            ReLU(),
            Linear(num_fc_units, num_output_features),
            output(),
        )

        initialize_weights(self)

    def forward(self, observations, state=None, info={}):
        x = observations["obs"]
        conv_block_out = self.first_conv_block(x)
        features = self.res_blocks(conv_block_out)
        output = self.output_head(features)

        if self.forward_returns_state:
            return output, state

        return output
