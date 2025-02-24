# https://arxiv.org/html/2408.07517v1

import torch
import torch.nn as nn
from snntorch._neurons import Synaptic
from .quantization_utils import round_to_minifloat

class Adlif(Synaptic):

    def __init__(
        self,
        alpha,
        beta,
        a=0.5,
        b=1.0,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=True,
        learn_beta=False,
        learn_ab=True,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        reset_delay=True,
    ):
        super().__init__(
            alpha,
            beta,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_alpha,
            learn_beta,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            reset_delay,
        )

        self._ab_register(a, b, learn_ab)
    

    def init_adlif(self):
        """Deprecated, use :class:`AdLIF.reset_mem` instead"""
        return self.reset_mem()


    def _ab_register(self, a, b, learn_ab):
        if not isinstance(a, torch.Tensor):
            a = torch.as_tensor(a, dtype=torch.float32)
        if not isinstance(b, torch.Tensor):
            b = torch.as_tensor(b, dtype=torch.float32)
        if learn_ab:
            self.a = nn.Parameter(a)
            self.b = nn.Parameter(b)
        else:
            self.register_buffer('a', a)
            self.register_buffer('b', b)


    def _base_state_function(self, input_):
        alpha_clamp = self.alpha.clamp(0, 1)
        base_fn_mem = alpha_clamp * self.mem + (1.0-alpha_clamp) * (input_ - self.syn)

        beta_clamp = self.beta.clamp(0, 1)
        a_clamp = self.a.clamp(0, 1)
        b_clamp = self.b.clamp(0, 2) # nel paper b è tra 0 e 2
        base_fn_syn = beta_clamp*self.syn + (1.0-beta_clamp) * (a_clamp*base_fn_mem + b_clamp*self.reset) # nel paper b è tra 0 e 2

        return base_fn_syn, base_fn_mem


    def _base_sub(self, input_):
        syn, mem = self._base_state_function(input_)
        mem = mem - self.reset * self.threshold
        return syn, mem


    def _base_zero(self, input_):
        syn, mem = self._base_state_function(input_)
        syn2, mem2 = 0.0, mem.clone()
        syn -= syn2 * self.reset
        mem -= mem2 * self.reset
        return syn, mem


    def _base_int(self, input_):
        return self._base_state_function(input_)



class Adlif_fixedpoint(Adlif):
    def __init__(
        self,
        fractional_bits,
        alpha,
        beta,
        a=0.5,
        b=1.0,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=True,
        learn_beta=False,
        learn_ab=True,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        reset_delay=True,
    ):
        super().__init__(
            alpha,
            beta,
            a,
            b,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_alpha,
            learn_beta,
            learn_ab,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            reset_delay,
        )

        self.frac_bits = fractional_bits


    def _base_state_function(self, input_):
        alpha_clamp = self.alpha.clamp(0, 2**self.frac_bits)
        part1 = torch.round(alpha_clamp * self.mem / 2**self.frac_bits)
        part2 = torch.round((2**self.frac_bits - alpha_clamp) * (input_ - self.syn) / 2**self.frac_bits)
        base_fn_mem = part1 + part2

        beta_clamp = self.beta.clamp(0, 2**self.frac_bits)
        a_clamp = self.a.clamp(0, 2**self.frac_bits)
        b_clamp = self.b.clamp(0, 2*2**self.frac_bits) # nel paper b è tra 0 e 2
        part1 = torch.round(beta_clamp*self.syn / 2**self.frac_bits)
        part2 = torch.round(a_clamp*base_fn_mem / 2**self.frac_bits)
        part3 = torch.round(b_clamp*(self.reset * 2**self.frac_bits) / 2**self.frac_bits)
        base_fn_syn = torch.round(part1 + (2**self.frac_bits-beta_clamp) * (part2 + part3) / 2**self.frac_bits)

        return base_fn_syn, base_fn_mem



class Adlif_minifloat(Adlif):
    def __init__(
        self,
        exponent_bits,
        mantissa_bits,
        alpha,
        beta,
        a=0.5,
        b=1.0,
        threshold=1.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_alpha=True,
        learn_beta=False,
        learn_ab=True,
        learn_threshold=False,
        reset_mechanism="subtract",
        state_quant=False,
        output=False,
        reset_delay=True,
    ):
        super().__init__(
            alpha,
            beta,
            a,
            b,
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_alpha,
            learn_beta,
            learn_ab,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            reset_delay,
        )

        self.exponent_bits = exponent_bits
        self.mantissa_bits = mantissa_bits


    def _base_state_function(self, input_):
        alpha_clamp = self.alpha.clamp(0, 1)
        one_alpha = round_to_minifloat(1.0-alpha_clamp, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        alpha_mem = round_to_minifloat(alpha_clamp * self.mem , exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        input_syn = round_to_minifloat(input_ - self.syn, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        one_alpha_syn = round_to_minifloat(one_alpha * input_syn, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        base_fn_mem = round_to_minifloat(alpha_mem + one_alpha_syn, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)

        beta_clamp = self.beta.clamp(0, 1)
        a_clamp = self.a.clamp(0, 1)
        b_clamp = self.b.clamp(0, 2) # nel paper b è tra 0 e 2

        one_beta = round_to_minifloat(1.0-beta_clamp, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        a_mem = round_to_minifloat(a_clamp*base_fn_mem, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        beta_syn = round_to_minifloat(beta_clamp*self.syn, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        b_reset = round_to_minifloat(b_clamp*self.reset, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        b_mem = round_to_minifloat(a_mem + b_reset, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        beta_mem = round_to_minifloat(one_beta * b_mem, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)
        base_fn_syn = round_to_minifloat(beta_syn + beta_mem, exponent_bits=self.exponent_bits, mantissa_bits=self.mantissa_bits)

        return base_fn_syn, base_fn_mem

