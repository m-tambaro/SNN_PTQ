import torch


## Fixed-point

def param_to_fullscale(net, total_bits, frac_bits): # Find the scale factor to use full dynamic range of fixed-point representation
    max_value = -float("Inf")
    min_value = +float("Inf")
    for param in net.parameters():
        max_value = max(max_value, torch.max(param))
        min_value = min(min_value, torch.min(param))

    scale_for_max = (2**(total_bits - 1) - 1) / max_value
    scale_for_min = 2**(total_bits - 1) / -min_value

    return min(scale_for_max, scale_for_min) / 2**frac_bits


def stochastic_round(tensor):
    floor_values = torch.floor(tensor)
    probs = tensor - floor_values
    random_values = torch.rand_like(tensor)
    return torch.where(random_values < probs, floor_values + 1, floor_values)


def quantize_to_fixed_point(tensor, total_bits, frac_bits, scaling_factor=1.0):
    quantized_tensor = torch.round(tensor * scaling_factor * 2**frac_bits)
    max_val = (2 ** (total_bits - 1) - 1)
    min_val = -(2 ** (total_bits - 1))
    greather_than_max = (quantized_tensor > max_val).sum().item()
    less_than_min = (quantized_tensor < min_val).sum().item()
    if greather_than_max > 0 or less_than_min > 0:
        print("Values exceeding max:", greather_than_max + less_than_min, "of ", quantized_tensor.shape[:])
    return quantized_tensor.clamp_(min_val, max_val)



def copy_and_quantize_to_fixed_point(source_net, target_net, total_bits, frac_bits, scale=False):
    """Copy and quantize parameters from one network to another using fixed-point representation.

    Args:
        source_net (torch.nn.Module): The original snnTorch network.
        target_net (torch.nn.Module): The new network where quantized parameters will be stored.
        total_bits (int): Total number of bits (including sign bit).
        frac_bits (int): Number of fractional bits.
    """
    if scale:
        scaling_factor = param_to_fullscale(source_net, total_bits, frac_bits)
    else:
        scaling_factor = 1.0
    
    with torch.no_grad():  # Disable gradient tracking
        for source_param, target_param in zip(source_net.named_parameters(), target_net.named_parameters()):
            #print("Copy", source_param[0], "to", target_param[0])
            target_param[1].copy_(quantize_to_fixed_point(source_param[1], total_bits, frac_bits, scaling_factor))

    return scaling_factor



### Minifloat


def round_to_minifloat(tensor, exponent_bits, mantissa_bits):
    """
    Convert a tensor of double precision numbers to minifloat representation.
    
    Args:
        x (torch.Tensor): Input tensor of float values
        mantissa_bits (int): Number of bits for mantissa (excluding sign bit)
        exponent_bits (int): Number of bits for exponent
        
    Returns:
        torch.Tensor: Tensor with values quantized to minifloat representation
    """
    if not isinstance(tensor, torch.Tensor):
        tensor = torch.tensor(tensor, dtype=torch.float64)
    
    # Handle special cases
    zero_mask = tensor == 0
    inf_mask = torch.isinf(tensor)
    nan_mask = torch.isnan(tensor)
    
    # Extract sign
    sign = torch.sign(tensor)
    x_abs = torch.abs(tensor)
    
    # Calculate exponent bias
    exp_bias = (1 << (exponent_bits - 1)) - 1
    
    # Calculate max and min exponents
    max_exp = (1 << exponent_bits) - 1 - exp_bias
    min_exp = 1 - exp_bias
    
    # Calculate log2 of absolute values
    with torch.no_grad():
        exp = torch.floor(torch.log2(x_abs + 1e-30))
        exp = exp.clamp_(min=min_exp, max=max_exp)
    
    # Calculate mantissa
    mantissa = x_abs / (2.0 ** exp)
    
    # Normalize mantissa to [1, 2)
    mantissa = mantissa.clamp_(1.0, 2.0 - 2.0 ** (-mantissa_bits))
    
    # Quantize mantissa
    mantissa = torch.floor((mantissa - 1.0) * (1 << mantissa_bits)) / (1 << mantissa_bits) + 1.0
    
    # Reconstruct float
    result = sign * mantissa * (2.0 ** exp)
    
    # Handle special cases
    result = torch.where(zero_mask, torch.zeros_like(result), result)
    result = torch.where(inf_mask, torch.sign(tensor) * float('inf'), result)
    result = torch.where(nan_mask, float('nan'), result)
    
    return result


def copy_and_quantize_to_minifloat(source_net, target_net, exp_bits, mant_bits, scale=1.0):
    """Copy and quantize parameters from one network to another using fixed-point representation.

    Args:
        source_net (torch.nn.Module): The original snnTorch network.
        target_net (torch.nn.Module): The new network where quantized parameters will be stored.
        total_bits (int): Total number of bits (including sign bit).
        frac_bits (int): Number of fractional bits.
    """
    with torch.no_grad():  # Disable gradient tracking
        for source_param, target_param in zip(source_net.named_parameters(), target_net.named_parameters()):
            #print("Copy", source_param[0])
            rounded_param = round_to_minifloat(tensor=source_param[1]*scale, exponent_bits=exp_bits, mantissa_bits=mant_bits)
            target_param[1].copy_(rounded_param)

