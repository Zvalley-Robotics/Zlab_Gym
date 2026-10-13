import torch

from rsl_rl.utils import resolve_nn_activation


def conv2d_output_shape(h, w, kernel_size=1, stride=1, padding=0, dilation=1):
    """
    Returns output H, W after convolution/pooling on input H, W.
    """
    kh, kw = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
    sh, sw = stride if isinstance(stride, tuple) else (stride,) * 2
    ph, pw = padding if isinstance(padding, tuple) else (padding,) * 2
    d = dilation
    h = (h + (2 * ph) - (d * (kh - 1)) - 1) // sh + 1
    w = (w + (2 * pw) - (d * (kw - 1)) - 1) // sw + 1
    return h, w


class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.

    Args:
        input_dim (int): number of inputs
        hidden_dims (list): can be empty list for none (linear model).
        output_dim: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
        self,
        input_dim,
        hidden_dims,  # Can be empty list or None for none.
        output_dim=None,  # if None, last layer has nonlinearity applied.
        nonlinearity="relu",  # Module, not Functional.
    ):
        super().__init__()
        if isinstance(hidden_dims, int):
            hidden_dims = (hidden_dims,)
        elif hidden_dims is None:
            hidden_dims = ()
        if isinstance(nonlinearity, str):
            nonlinearity = resolve_nn_activation(nonlinearity)
        hidden_layers = [
            torch.nn.Linear(n_in, n_out) for n_in, n_out in zip((input_dim,) + hidden_dims[:-1], hidden_dims)
        ]
        sequence = list()
        for layer in hidden_layers:
            sequence.extend([layer, nonlinearity])
        if output_dim is not None:
            last_size = hidden_dims[-1] if hidden_dims else input_dim
            sequence.append(torch.nn.Linear(last_size, output_dim))
        self.model = torch.nn.Sequential(*sequence)
        self._output_dim = hidden_dims[-1] if output_dim is None else output_dim

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_dim]."""
        return self.model(input)

    @property
    def output_dim(self):
        """Returns the output size of the model."""
        return self._output_dim


class Conv2d(torch.nn.Module):
    """2-D Convolutional model component, with option for max-pooling vs
    downsampling for strides > 1.  Requires number of input channels, but
    not input shape.  Uses ``torch.nn.Conv2d``.
    """

    def __init__(
        self,
        in_channels,
        channels,
        kernel_sizes,
        strides,
        paddings=None,
        nonlinearity="relu",  # Module, not Functional.
        use_maxpool=False,  # if True: convs use stride 1, maxpool downsample.
        normlayer=None,  # If None, will not be used
    ):
        super().__init__()
        if isinstance(nonlinearity, str):
            nonlinearity = resolve_nn_activation(nonlinearity)
        if paddings is None:
            paddings = [0 for _ in range(len(channels))]
        if isinstance(normlayer, str):
            normlayer = getattr(torch.nn, normlayer)
        assert len(channels) == len(kernel_sizes) == len(strides) == len(paddings)
        in_channels = (in_channels,) + channels[:-1]
        ones = [1 for _ in range(len(strides))]
        if use_maxpool:
            maxp_strides = strides
            strides = ones
        else:
            maxp_strides = ones
        conv_layers = [
            torch.nn.Conv2d(in_channels=ic, out_channels=oc, kernel_size=k, stride=s, padding=p)
            for (ic, oc, k, s, p) in zip(in_channels, channels, kernel_sizes, strides, paddings)
        ]
        sequence = list()
        for conv_layer, oc, maxp_stride in zip(conv_layers, channels, maxp_strides):
            if normlayer is not None:
                sequence.extend([conv_layer, normlayer(oc), nonlinearity])
            else:
                sequence.extend([conv_layer, nonlinearity])
            if maxp_stride > 1:
                sequence.append(torch.nn.MaxPool2d(maxp_stride))  # No padding.
        self.conv = torch.nn.Sequential(*sequence)

    def forward(self, input):
        """Computes the convolution stack on the input; assumes correct shape
        already: [B,C,H,W]."""
        return self.conv(input)

    def conv_out_dim(self, h, w, c=None):
        """Helper function to return the output size for a given input shape,
        without actually performing a forward pass through the model."""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size, child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h * w * c

    def conv_out_resolution(self, h, w):
        """Helper function that return the resolution (H, W) for a giben input resolution"""
        for child in self.conv.children():
            try:
                h, w = conv2d_output_shape(h, w, child.kernel_size, child.stride, child.padding)
            except AttributeError:
                pass  # Not a conv or maxpool layer.
            try:
                c = child.out_channels
            except AttributeError:
                pass  # Not a conv layer.
        return h, w


class Conv2dHead(torch.nn.Module):
    """Model component composed of a ``Conv2d`` component followed by
    a fully-connected ``Mlp`` head.  Requires full input image shape to
    instantiate the MLP head.
    """

    def __init__(
        self,
        image_shape,
        channels,
        kernel_sizes,
        strides,
        hidden_dims,
        output_dim=None,  # if None: nonlinearity applied to output.
        paddings=None,
        nonlinearity="relu",
        use_maxpool=False,
        normlayer=None,  # if None, will not be used
    ):
        super().__init__()
        c, h, w = image_shape
        self.conv = Conv2d(
            in_channels=c,
            channels=channels,
            kernel_sizes=kernel_sizes,
            strides=strides,
            paddings=paddings,
            nonlinearity=nonlinearity,
            use_maxpool=use_maxpool,
            normlayer=None,  # if None, will not be used
        )
        conv_out_dim = self.conv.conv_out_dim(h, w)
        if hidden_dims or output_dim:
            self.head = MlpModel(conv_out_dim, hidden_dims, output_dim, nonlinearity)
            if output_dim is not None:
                self._output_dim = output_dim
            else:
                self._output_dim = hidden_dims if isinstance(hidden_dims, int) else hidden_dims[-1]
        else:
            self.head = lambda x: x
            self._output_dim = conv_out_dim

    def forward(self, input):
        """Compute the convolution and fully connected head on the input;
        assumes correct input shape: [B,C,H,W]."""
        return self.head(self.conv(input).view(input.shape[0], -1))

    @property
    def output_dim(self):
        """Returns the final output dim after MLP head."""
        return self._output_dim
