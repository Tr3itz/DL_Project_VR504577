# *-----MPSModule
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            * * you may not use this file except in compliance with the License.           *
# * You may obtain a copy of the License at                                    *
# *                                                                            *
# * http://www.apache.org/licenses/LICENSE-2.0                                 *
# *                                                                            *
# * Unless required by applicable law or agreed to in writing, software        *
# * distributed under the License is distributed on an "AS IS" BASIS,          *
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.   *
# * See the License for the specific language governing permissions and        *
# * limitations under the License.                                             *
# *                                                                            *
# * Author:  Matteo Risso <matteo.risso@polito.it>                             *
# *----------------------------------------------------------------------------*

from abc import abstractmethod
from enum import Enum, auto
from typing import Dict, Tuple, Type, cast, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..quant.quantizers import Quantizer
from .ste_argmax import STEArgmax


class MPSType(Enum):
    PER_CHANNEL = auto()
    PER_LAYER = auto()


class MPSBaseQtz(nn.Module):
    """A nn.Module implementing a generic searchable mixed-precision quantization. Base class
    specialized below for per-channel and per-layer search.

    :param precision: different bitwitdth alternatives among which perform search
    :type precision: Tuple[int, ...]
    :param quantizer: input quantizer
    :type quantizer: Quantizer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    :param softmax_temperature: initial temperature for SoftMax computation
    :type softmax_temperature: float, optional (default = 1.0)
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool, optional (default = False)
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SoftMax
    :type gumbel_softmax: bool, optional (default = False)
    :param disable_sampling: whether to disable the sampling of the architectural coefficients
    during the forward pass
    :type disable_sampling: bool, optional (default = False)
    """
    def __init__(self,
                 precision: Tuple[int, ...],
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {},
                 softmax_temperature: float = 1.0,
                 hard_softmax: bool = False,
                 gumbel_softmax: bool = False,
                 disable_sampling: bool = False):
        super(MPSBaseQtz, self).__init__()
        if len(precision) != len(set(precision)):
            raise ValueError("precision cannot be repeated")
        self.register_buffer('precision', torch.tensor(precision, dtype=torch.float))
        self.quantizer = quantizer
        self.quantizer_kwargs = quantizer_kwargs
        # create individual quantizers
        self.qtz_funcs = nn.ModuleList()
        for p in precision:
            qtz = quantizer(p, **quantizer_kwargs)
            qtz = cast(nn.Module, qtz)
            self.qtz_funcs.append(qtz)
        # create buffer for temperature tensor
        self.register_buffer('temperature', torch.tensor(softmax_temperature, dtype=torch.float))
        # set the sampling options
        self.update_softmax_options(
                temperature=softmax_temperature,
                hard=hard_softmax,
                gumbel=gumbel_softmax,
                disable_sampling=disable_sampling)

    @abstractmethod
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, it computes the different quantized representations of `mix_qtz`
        and combines them weighting the different terms channel-wise or layer-wise by means of
        softmax-ed `alpha` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        :raises: NotImplementedError on the base class
        """
        raise NotImplementedError("Trying to call forward on the base MPS quantizer class")

    def sample_alpha_sm(self):
        """
        Samples the alpha coefficients using a standard SoftMax (with temperature).
        The corresponding normalized parameters (summing to 1) are stored in the theta_alpha buffer.
        """
        self.theta_alpha = F.softmax(
                cast(torch.Tensor, self.alpha) / self.temperature.item(), dim=0)
        if (self.hard_softmax) or (not self.training):
            self.theta_alpha = cast(torch.Tensor, STEArgmax.apply(self.theta_alpha))

    def sample_alpha_gs(self):
        """
        Samples the alpha architectural coefficients using a Gumbel SoftMax (with temperature).
        The corresponding normalized parameters (summing to 1) are stored in the theta_alpha buffer.
        """
        if self.training:
            self.theta_alpha = F.gumbel_softmax(
                logits=cast(torch.Tensor, self.alpha),
                tau=self.temperature.item(),
                hard=self.hard_softmax,
                dim=0)
        else:
            self.sample_alpha_sm()

    def sample_alpha_none(self):
        """Sample the previous alpha architectural coefficients. Used to change the alpha
        coefficients at each iteration"""
        return

    def update_softmax_options(
            self,
            temperature: Optional[float] = None,
            hard: Optional[bool] = None,
            gumbel: Optional[bool] = None,
            disable_sampling: Optional[bool] = None):
        """Set the flags to choose between the softmax, the hard and soft Gumbel-softmax
        and the sampling disabling of the architectural coefficients in the quantizers

        :param temperature: SoftMax temperature
        :type temperature: Optional[float]
        :param hard: Hard vs Soft sampling
        :type hard: Optional[bool]
        :param gumbel: Gumbel-softmax vs standard softmax
        :type gumbel: Optional[bool]
        :param disable_sampling: disable the sampling of the architectural coefficients in the
        forward pass
        :type disable_sampling: Optional[bool]
        """
        if temperature is not None:
            self.temperature = torch.tensor(temperature, dtype=torch.float32)
        if hard is not None:
            self.hard_softmax = hard
        if disable_sampling is not None and disable_sampling:
            self.sample_alpha = self.sample_alpha_none
        elif gumbel is not None and gumbel:
            self.sample_alpha = self.sample_alpha_gs
        else:
            self.sample_alpha = self.sample_alpha_sm

    @property
    def effective_scale(self) -> torch.Tensor:
        """Return the quantizer's effective scale factor as the average scale factor weighted
        by softmax-ed `alpha` parameters

        :return: the effective scale factor, either a scalar tensor (per-layer scale) or a
        rank-1 tensor (per-channel scale)
        :rtype: torch.Tensor
        :raises: NotImplementedError on the base class
        """
        scale = torch.tensor(0, dtype=torch.float32)
        for i, qtz in enumerate(self.qtz_funcs):
            scale = scale + (self.theta_alpha[i] * qtz.scale)
        return scale

    @property
    @abstractmethod
    def effective_precision(self) -> torch.Tensor:
        """Return the layer's effective precision as the average precision weighted by softmax-ed
        `alpha` parameters

        :return: the effective precision
        :rtype: torch.Tensor
        :raises: NotImplementedError on the base class
        """
        raise NotImplementedError(
                "Trying to compute effective precision on the base MPS quantizer class")


class MPSPerChannelQtz(MPSBaseQtz):
    """A nn.Module implementing a generic searchable mixed-precision quantization applied to each
    channel of the tensor provided as input. This module includes trainable NAS parameters.

    :param precision: different bitwitdth alternatives among which perform search
    :type precision: Tuple[int, ...]
    :param quantizer: input quantizer
    :type quantizer: Quantizer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    :param softmax_temperature: initial temperature for SoftMax computation
    :type softmax_temperature: float, optional (default = 1.0)
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool, optional (default = False)
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SoftMax
    :type gumbel_softmax: bool, optional (default = False)
    :param disable_sampling: whether to disable the sampling of the architectural coefficients
    during the forward pass
    :type disable_sampling: bool, optional (default = False)
    """
    def __init__(self,
                 precision: Tuple[int, ...],
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {},
                 softmax_temperature: float = 1.0,
                 hard_softmax: bool = False,
                 gumbel_softmax: bool = False,
                 disable_sampling: bool = False):
        super(MPSPerChannelQtz, self).__init__(
                precision,
                quantizer,
                quantizer_kwargs,
                softmax_temperature,
                hard_softmax,
                gumbel_softmax,
                disable_sampling)
        # find the 0-bit precision (if present)
        self.zero_index = None if 0 not in precision else precision.index(0)
        # create and initialize the NAS parameters
        self.alpha = nn.Parameter(torch.empty((len(precision), quantizer_kwargs['cout']),
                                              dtype=torch.float32), requires_grad=True)
        max_precision = max(precision)
        for i, p in enumerate(precision):
            self.alpha.data[i, :].fill_(float(p) / max_precision)
        self.register_buffer('theta_alpha', torch.ones((len(precision), quantizer_kwargs['cout']),
                                                       dtype=torch.float32))
        # initial sampling
        with torch.no_grad():
            self.sample_alpha()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, it computes the different quantized representations of `mix_qtz`
        and combines them weighting the different terms channel-wise by means of
        softmax-ed `alpha` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        self.sample_alpha()
        y = []
        for i, quantizer in enumerate(self.qtz_funcs):
            theta_alpha_i = self.theta_alpha[i].view((self.theta_alpha.size(dim=1),) +
                                                     (1,) * len(input.shape[1:]))
            y.append(theta_alpha_i * quantizer(input))
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    @property
    def features_mask(self) -> torch.Tensor:
        """Return the binarized mask that specifies which output features (channels) are kept by
        the NAS

        :return: the binarized mask over the features axis
        :rtype: torch.Tensor
        """
        with torch.no_grad():
            if self.zero_index is not None:
                # extract the row corresponding to 0-bit precision from the one-hot argmax matrix
                # of size (n_prec, cout), and do a 1s complement
                return 1 - cast(torch.Tensor, STEArgmax.apply(self.theta_alpha))[self.zero_index, :]
            else:
                return torch.ones((self.theta_alpha.size(dim=1),))

    @property
    def out_features_eff(self) -> torch.Tensor:
        """Effective number of not-pruned channels"""
        if self.zero_index is not None:
            return self.theta_alpha.size(dim=1) - torch.sum(self.theta_alpha[self.zero_index, :])
        else:
            return torch.tensor(self.theta_alpha.size(dim=1), dtype=torch.float32)

    @property
    def effective_precision(self) -> torch.Tensor:
        """Return each channel effective precision as the average precision weighted by
        softmax-ed `alpha` parameters

        :return: the effective precision
        :rtype: torch.Tensor
        """
        # stabilization constant added to the denominator to avoid NaNs
        eff_prec = ((self.theta_alpha.sum(dim=1) * self.precision).sum() /
                    (self.out_features_eff + 1e-10))
        return eff_prec


class MPSPerLayerQtz(MPSBaseQtz):
    """A nn.Module implementing a generic mixed-precision quantization searchable
    operation of the tensor provided as input.
    This module includes trainable NAS parameters.

    :param precision: different bitwitdth alternatives among which perform search
    :type precision: Tuple[int, ...]
    :param quantizer: input quantizer
    :type quantizer: Quantizer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    :param gumbel_softmax: use Gumbel SoftMax for sampling, instead of a normal SoftMax
    :type gumbel_softmax: bool, optional
    :param hard_softmax: use hard Gumbel SoftMax sampling (only applies when gumbel_softmax = True)
    :type hard_softmax: bool, optional
    :param disable_sampling: whether to disable the sampling of the architectural coefficients
    during the forward pass
    :type disable_sampling: bool, optional
    """
    def __init__(self,
                 precision: Tuple[int, ...],
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {},
                 softmax_temperature: float = 1.0,
                 hard_softmax: bool = False,
                 gumbel_softmax: bool = False,
                 disable_sampling: bool = False):
        super(MPSPerLayerQtz, self).__init__(
                precision,
                quantizer,
                quantizer_kwargs,
                softmax_temperature,
                hard_softmax,
                gumbel_softmax,
                disable_sampling)
        self.alpha = nn.Parameter(torch.empty((len(precision),), dtype=torch.float32),
                                  requires_grad=True)
        max_precision = max(precision)
        for i, p in enumerate(precision):
            self.alpha.data[i].fill_(float(p) / max_precision)
        self.register_buffer('theta_alpha', torch.ones((len(precision),), dtype=torch.float32))
        # initial sampling
        with torch.no_grad():
            self.sample_alpha()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the searchable mixed-precision layer.

        In a nutshell, it computes the different quantized representations of `mix_qtz`
        and combines them weighting the different terms by means of softmax-ed
        `alpha` trainable parameters.

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        self.sample_alpha()
        y = []
        for i, quantizer in enumerate(self.qtz_funcs):
            y.append(self.theta_alpha[i] * quantizer(input))
        y = torch.stack(y, dim=0).sum(dim=0)
        return y

    @property
    def effective_precision(self) -> torch.Tensor:
        """Return the effective precision as the average precision weighted by
        softmax-ed `alpha` parameters

        :return: the effective precision
        :rtype: torch.Tensor
        """
        eff_prec = (self.theta_alpha * self.precision).sum()
        return eff_prec


class MPSBiasQtz(nn.Module):
    """A nn.Module implementing mixed-precision quantization searchable
    operation of the bias vector provided as input.

    Just a wrapper around a bias quantizer

    :param quantizer: bias quantizer
    :type quantizer: Quantizer
    :param quantizer_kwargs: quantizer kwargs, if no kwargs are passed default is used
    :type quantizer_kwargs: Dict, optional
    """
    def __init__(self,
                 quantizer: Type[Quantizer],
                 quantizer_kwargs: Dict = {},
                 ):
        super(MPSBiasQtz, self).__init__()
        self.quantizer = quantizer
        self.quantizer_kwargs = quantizer_kwargs
        self.qtz_func = quantizer(**quantizer_kwargs)

    def forward(self, input: torch.Tensor, scale_a: torch.Tensor,
                scale_w: torch.Tensor) -> torch.Tensor:
        """The forward function

        :param input: the input float tensor
        :type input: torch.Tensor
        :return: the output fake-quantized with searchable precision tensor
        :rtype: torch.Tensor
        """
        y = self.qtz_func(input, scale_a, scale_w)
        return y
