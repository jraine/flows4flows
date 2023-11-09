import warnings

import numpy as np
import torch
from nflows import transforms
from nflows.transforms.splines import rational_quadratic_spline, \
    unconstrained_rational_quadratic_spline
from nflows.utils import torchutils
from nflows.transforms import splines
from torch import nn


class MaskedPiecewiseRationalQuadraticAutoregressiveTransformIdentInit(
    transforms.MaskedPiecewiseRationalQuadraticAutoregressiveTransform):
    def __init__(self, *args, identity_init=True, **kwargs):
        self.identity_init = identity_init

        super().__init__(*args, **kwargs)
        if self.identity_init:
            # set the last layer weights to zero
            self.autoregressive_net.final_layer.bias = nn.Parameter(
                torch.zeros_like(self.autoregressive_net.final_layer.bias)
            )
            self.autoregressive_net.final_layer.weight = nn.Parameter(
                torch.zeros_like(self.autoregressive_net.final_layer.weight)
            )

    def _elementwise(self, inputs, autoregressive_params, inverse=False):
        batch_size, features = inputs.shape[0], inputs.shape[1]

        transform_params = autoregressive_params.view(
            batch_size, features, self._output_dim_multiplier()
        )

        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins: 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins:]

        if hasattr(self.autoregressive_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.autoregressive_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.autoregressive_net.hidden_features)

        if self.tails is None:
            spline_fn = rational_quadratic_spline
            spline_kwargs = {}
        elif self.tails == "linear":
            spline_fn = unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}
        else:
            raise ValueError

        outputs, logabsdet = spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            enable_identity_init=self.identity_init,
            **spline_kwargs
        )

        return outputs, torchutils.sum_except_batch(logabsdet)


class PiecewiseRationalQuadraticCouplingTransformIdentInit(transforms.PiecewiseRationalQuadraticCouplingTransform):
    def __init__(self, mask, transform_net_create_fn, identity_init=True, **kwargs):
        self.identity_init = identity_init

        super().__init__(mask, transform_net_create_fn, **kwargs)
        # Check that the maker function will return zeros at initialisation

        if self.identity_init:
            if not self.check_init(mask):
                try:
                    final_layer = self.transform_net[-1]
                    final_layer.weight = nn.Parameter(torch.zeros_like(final_layer.weight))
                    final_layer.bias = nn.Parameter(torch.zeros_like(final_layer.bias))
                except Exception as e:
                    print(e)
                if not self.check_init(mask):
                    raise Exception("Initialise transform_net_create_fn to return zeros at initialisation.")

    def check_init(self, mask):
        data = torch.randn((100, np.sum(1 - np.array(mask))))
        output = self.transform_net(data)
        return output.sum() <= 10 ** (-8)

    def _piecewise_cdf(self, inputs, transform_params, inverse=False):
        unnormalized_widths = transform_params[..., : self.num_bins]
        unnormalized_heights = transform_params[..., self.num_bins: 2 * self.num_bins]
        unnormalized_derivatives = transform_params[..., 2 * self.num_bins:]

        if hasattr(self.transform_net, "hidden_features"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_features)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_features)
        elif hasattr(self.transform_net, "hidden_channels"):
            unnormalized_widths /= np.sqrt(self.transform_net.hidden_channels)
            unnormalized_heights /= np.sqrt(self.transform_net.hidden_channels)
        else:
            warnings.warn(
                "Inputs to the softmax are not scaled down: initialization might be bad."
            )

        if self.tails is None:
            spline_fn = splines.rational_quadratic_spline
            spline_kwargs = {}
        else:
            spline_fn = splines.unconstrained_rational_quadratic_spline
            spline_kwargs = {"tails": self.tails, "tail_bound": self.tail_bound}

        return spline_fn(
            inputs=inputs,
            unnormalized_widths=unnormalized_widths,
            unnormalized_heights=unnormalized_heights,
            unnormalized_derivatives=unnormalized_derivatives,
            inverse=inverse,
            min_bin_width=self.min_bin_width,
            min_bin_height=self.min_bin_height,
            min_derivative=self.min_derivative,
            enable_identity_init=self.identity_init,
            **spline_kwargs
        )


def test_initialisations():
    n_dim = 5
    data = torch.rand((100, n_dim))
    layer = MaskedPiecewiseRationalQuadraticAutoregressiveTransformIdentInit(n_dim, 128)
    output, _ = layer(data)
    print(f'Max MAE {(data - output).abs().max()} on autoregressive')

    # Define a simple MLP for testing
    class MLP(nn.Sequential):
        def __init__(self, input_dim, output_dim):
            layers = (
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, output_dim)
            )
            super(MLP, self).__init__(*layers)
            # This is how you can set the final layer to zeros
            # final_layer = self[-1]
            # final_layer.weight = nn.Parameter(torch.zeros_like(final_layer.weight))
            # final_layer.bias = nn.Parameter(torch.zeros_like(final_layer.bias))

        def forward(self, input, context=None):
            '''
            A context keyword is expected here, but is ignored for testing.
            '''
            return super(MLP, self).forward(input)

    n_mask = n_dim // 2
    mask = [1] * n_mask + [0] * int(n_dim - n_mask)
    layer = PiecewiseRationalQuadraticCouplingTransformIdentInit(mask, MLP)
    output, _ = layer(data)
    print(f'Max MAE {(data - output).abs().max()} on coupling')


if __name__ == '__main__':
    test_initialisations()