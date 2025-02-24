import torch
import torch.nn as nn
from snntorch._neurons import Leaky
from .quantization_utils import round_to_minifloat

class LinearLeaky(Leaky):

    def __init__(
        self,
        beta=0.0,
        threshold=1.0,
        resting=0.0,
        learn_resting=False,
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

        self._resting_buffer(resting, learn_resting)


    def _resting_buffer(self, resting, learn_resting):
        if not isinstance(resting, torch.Tensor):
            resting = torch.as_tensor(resting)
        if learn_resting:
            self.resting = nn.Parameter(resting)
        else:
            self.register_buffer('resting', resting)


    def _base_state_function(self, input_):
        base_fn = self.mem - self.beta + input_
        return base_fn.clamp(min=self.resting)



class LinearLeaky_minifloat(LinearLeaky):

    def __init__(
        self,
        exponent_bits,
        mantissa_bits,
        beta,
        threshold=1.0,
        resting=0.0,
        learn_resting=False,
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
            resting,
            learn_resting,
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

        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits


    def _base_state_function(self, input_):
        base_fn = round_to_minifloat(self.mem - self.beta + input_, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        return base_fn.clamp(min=self.resting)


