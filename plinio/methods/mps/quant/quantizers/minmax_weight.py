# *----------------------------------------------------------------------------*
# * Copyright (C) 2022 Politecnico di Torino, Italy                            *
# * SPDX-License-Identifier: Apache-2.0                                        *
# *                                                                            *
# * Licensed under the Apache License, Version 2.0 (the "License");            *
# * you may not use this file except in compliance with the License.           *
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

from typing import Dict, Any, Optional, Iterator, Tuple
import torch
import torch.fx as fx
import torch.nn as nn
from .quantizer import Quantizer


class MinMaxWeight(Quantizer):
    """A nn.Module implementing a min-max quantization strategy for weights.

    :param precision: quantization precision
    :type precision: int
    :param cout: number of output channels
    :type cout: int
    :param symmetric: wether the weight upper and lower bound should be the same
    :type init_clip_val: bool
    :param dequantize: whether the output should be fake-quantized or not
    :type dequantize: bool
    """
    def __init__(self,
                 precision: int,
                 cout: int,
                 symmetric: bool = True,
                 dequantize: bool = True):
        super(MinMaxWeight, self).__init__(precision, dequantize)
        if symmetric:
            self.qtz_func = MinMaxSymSTE if symmetric else MinMaxAsymSTE
            self.compute_min_max = self._compute_min_max_sym
        else:
            self.qtz_func = MinMaxAsymSTE
            self.compute_min_max = self._compute_min_max_asym
        self.ch_max = torch.Tensor(cout)
        self.ch_min = torch.Tensor(cout)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        """The forward function of the MinMax weight quantizer.

        Compute quantization using the whole weights' value span and implements STE
        for the backward pass

        :param input: the input float weights tensor
        :type input: torch.Tensor
        :return: the output fake-quantized weights tensor
        :rtype: torch.Tensor
        """
        # N.B., here detach is necessary to avoid the nasty error message
        # "backpropapating two times within the computational graph"
        self.ch_min, self.ch_max = self.compute_min_max(input.detach())
        input_q = self.qtz_func.apply(input,
                                      self.ch_min,
                                      self.ch_max,
                                      self.precision,
                                      self.dequantize)
        return input_q

    @staticmethod
    def export(n: fx.Node, mod: fx.GraphModule, backend: Optional[str]):
        """Replaces a fx.Node corresponding to a Quantizer, with a "backend-aware" layer
        within a fx.GraphModule

        :param n: the node to be rewritten
        :type n: fx.Node
        :param mod: the parent module, where the new node has to be inserted
        :type mod: fx.GraphModule
        :param backend: an optional string specifying the target backend
        :type backend: Optional[str]
        """
        raise NotImplementedError("TODO")

    @property
    def scale(self) -> torch.Tensor:
        """Return the computed scale factor which depends upon self.precision and
        weights magnitude

        :return: the scale factor
        :rtype: torch.Tensor
        """
        ch_range = self.ch_max - self.ch_min
        if self.precision != 0:
            ch_range.masked_fill_(ch_range.eq(0), 1)
            n_steps = 2 ** self.precision - 1
            scale_factor = ch_range / n_steps
        else:
            scale_factor = torch.zeros(ch_range.shape, device=ch_range.device)
        return scale_factor

    def summary(self) -> Dict[str, Any]:
        """Export a dictionary with the optimized layer quantization hyperparameters

        :return: a dictionary containing the optimized layer quantization hyperparameter values
        :rtype: Dict[str, Any]
        """
        return {
            'scale_factor': self.scale.detach().item(),
        }

    def named_quant_parameters(
            self, prefix: str = '', recurse: bool = False) -> Iterator[Tuple[str, nn.Parameter]]:
        """Returns an iterator over the quantization parameters of this layer, yielding
        both the name of the parameter as well as the parameter itself

        :param prefix: prefix to prepend to all parameter names.
        :type prefix: str
        :param recurse: recurse to sub-modules
        :type recurse: bool
        :return: an iterator over the quantization parameters of this layer
        :rtype: Iterator[nn.Parameter]
        """
        prfx = prefix
        prfx += "." if len(prefix) > 0 else ""
        for name, param in self.named_parameters(
                prfx + "weight_quantizer", recurse):
            yield name, param

    def _compute_min_max_sym(self, input: torch.Tensor
                             ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes symmetric ch_min and ch_max of the given input tensor.

        :param input: the input tensor to be analyzed
        :type input: torch.Tensor
        :return: a tuple containing respectively the min and the max.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        ch_max, _ = input.view(input.size(0), -1).abs().max(1)
        ch_min = -1 * ch_max
        return ch_min, ch_max

    def _compute_min_max_asym(self, input: torch.Tensor
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Computes asymmetric ch_min and ch_max of the given input tensor.

        :param input: the input tensor to be analyzed
        :type input: torch.Tensor
        :return: a tuple containing respectively the min and the max.
        :rtype: Tuple[torch.Tensor, torch.Tensor]
        """
        ch_max, _ = input.view(input.size(0), -1).max(1)
        ch_min, _ = input.view(input.size(0), -1).min(1)
        return ch_min, ch_max

    def __repr__(self):
        msg = (
            f'{self.__class__.__name__}'
            f'(precision={self.precision}, '
            f'scale_factor={self.scale})'
        )
        return msg


class MinMaxAsymSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ch_min, ch_max, precision, dequantize):
        return _min_max_quantize(x, ch_min, ch_max, precision, dequantize)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


class MinMaxSymSTE(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, ch_min, ch_max, precision, dequantize):
        return _min_max_quantize(x, ch_min, ch_max, precision, dequantize)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None


def _min_max_quantize(x, ch_min, ch_max, precision, dequantize):

    # if the precision is equal to 0 bit, return a zeros-filled tensor
    if precision != 0:
        # Compute scale factor
        ch_range = ch_max - ch_min
        ch_range.masked_fill_(ch_range.eq(0), 1)
        n_steps = 2 ** precision - 1
        scale_factor = ch_range / n_steps

        # Reshape
        shape = (x.shape[0],) + (1,) * len(x.shape[1:])
        scale_factor = scale_factor.view(shape)

        # Quantize
        # N.B., round gives problems during integerazation cause may happen that
        # a positive signed number is rounded-up exceeding what is representable
        # on n bits (e.g., n=8bits round(127.5)-->128).
        # Conversely, floor is a biased rounding operator which introduces
        # asymmetric quantization erros that severely hinder computation.
        # We solve the problem using the unbiased rounding to nearest even strategy
        # (aka round) and we clip to avoid exceeding the dynamic.
        y = torch.round(x / scale_factor)
        y = torch.clip(y, max=2 ** (precision - 1) - 1)
        # y = torch.floor(x / scale_factor)

        if dequantize:
            y = y * scale_factor

    else:  # 0-bit precision
        y = torch.zeros(x.shape, device=x.device)

    return y
