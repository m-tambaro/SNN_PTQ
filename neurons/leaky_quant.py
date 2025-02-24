from snntorch._neurons import Leaky
from .quantization_utils import round_to_minifloat

import torch


class Leaky_fixedpoint(Leaky):

    def __init__(
        self,
        fractional_bits,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
            reset_delay,
        )

        self.fractional_bits = fractional_bits


    def _base_state_function(self, input_):
        return torch.round(self.beta.clamp(0, 2**self.fractional_bits) * self.mem / (2**self.fractional_bits)) + input_



class Leaky_minifloat(Leaky):

    def __init__(
        self,
        exponent_bits,
        mantissa_bits,
        beta,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_beta=False,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
        reset_delay=True,
    ):
        super().__init__(
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
            reset_delay,
        )

        self.mantissa_bits = mantissa_bits
        self.exponent_bits = exponent_bits


    def _base_state_function(self, input_):
        return round_to_minifloat(round_to_minifloat(tensor=self.beta.clamp(0, 1) * self.mem, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
                                   + input_, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
 

