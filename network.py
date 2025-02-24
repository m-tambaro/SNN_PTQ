import os
import re
import time
import statistics
import matplotlib.pyplot as plt
from IPython.display import display
from ipywidgets import IntProgress

import torch
import torch.nn as nn
import snntorch as snn
from snntorch import utils
from snntorch import functional as SF

from importlib import reload

# Convert elapsed time to minutes, seconds, and milliseconds
def print_elapsed_time(elapsed_time):
    minutes, seconds = divmod(elapsed_time, 60)
    seconds, milliseconds = divmod(seconds, 1)
    milliseconds = round(milliseconds * 1000)

    # Print the elapsed time
    print(f"Elapsed time: {int(minutes)} minutes, {int(seconds)} seconds, {milliseconds} milliseconds")


def device_information():
    os.system('nvidia-smi')
    print("Using torch", torch.__version__)

    if torch.cuda.is_available():
        device = torch.device('cuda')   # CUDA GPU
    elif torch.backends.mps.is_available():
        device = torch.device('mps')    # Apple GPU
    else:
        device = torch.device("cpu")

    print('Using device:', device)
    #Additional Info when using cuda
    if device.type == 'cuda':
        print("Device name: ", torch.cuda.get_device_name(0))
        print("Device properties:", torch.cuda.get_device_properties(0))
        print('Memory Usage:')
        print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
        print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')

    return device


def forward_pass(net, data):
    spk_rec = []
    utils.reset(net)  # resets hidden states for all LIF neurons in net

    for step in range(data.size(0)):  # data.size(0) = number of time steps
        spk_out = net(data[step])
        spk_rec.append(spk_out)

    return torch.stack(spk_rec)


def train_network(device, net, trainloader, num_epochs=20, testloader=None):

    optimizer = torch.optim.Adam(net.parameters(), lr=2e-2, betas=(0.9, 0.999))
    loss_fn = SF.mse_count_loss(correct_rate=0.8, incorrect_rate=0.2)

    start_time = time.time()

    train_bar = IntProgress(min=0, max=len(trainloader))
    epoch_bar = IntProgress(min=0, max=len(trainloader) * num_epochs)
    print("Training progress:")
    display(epoch_bar)
    print("Epoch progress:")
    display(train_bar)

    loss_hist = []
    acc_hist = []
    loss_hist_epoch = [0]
    acc_hist_epoch = [0]
    epoch_end = [0]

    # training loop
    for epoch in range(num_epochs):
        for i, (data, targets) in enumerate(iter(trainloader)):
            # Move data and targets to the device (GPU or CPU)
            data = data.to(device)
            targets = targets.to(device)

            net.train()
            spk_rec = forward_pass(net, data)
            loss_val = loss_fn(spk_rec, targets)

            # Gradient calculation + weight update
            optimizer.zero_grad()
            loss_val.backward(retain_graph=True)
            optimizer.step()

            # Store loss history for future plotting
            loss_hist.append(loss_val.item())

            # Calculate accuracy rate and then append it to accuracy history
            acc = SF.accuracy_rate(spk_rec, targets) * 100
            acc_hist.append(acc)
            
            train_bar.value += 1
            epoch_bar.value += 1
        train_bar.value = 0

        epoch_end.append(epoch_end[-1]+i)
        loss_hist_epoch.append(statistics.mean(loss_hist[-len(targets):]))
        acc_hist_epoch.append(statistics.mean(acc_hist[-len(targets):]))

        # Print loss & accuracy every epoch
        #if (i+1)%10 == 0:
        print(f"Epoch {epoch+1}", f"\tBatch Avg Train Loss: {loss_hist_epoch[-1]:.2f}", f"\tBatch Avg Accuracy: {acc_hist_epoch[-1]:.2f}%")

        if testloader is not None:
            acc = eval_accuracy(device, net, testloader)
            print(f"\tAccuracy on testset: {acc/len(testloader)*100:.2f}")

    loss_hist_epoch[0] = loss_hist[0]
    acc_hist_epoch[0] = acc_hist[0]

    print_elapsed_time(time.time() - start_time)

    return acc_hist, loss_hist, acc_hist_epoch, loss_hist_epoch, epoch_end


def plot_accuracy_and_loss(acc_hist, loss_hist, acc_hist_epoch=None, loss_hist_epoch=None, epoch_step=None):
    _, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(acc_hist)
    if epoch_step is not None:
        ax1.plot(epoch_step, acc_hist_epoch)
    ax1.set_ylabel("Accuracy")
    ax1.set_xlabel("Iteration")
    ax2.plot(loss_hist)
    if epoch_step is not None:
        ax2.plot(epoch_step, loss_hist_epoch)
    ax2.set_ylabel("Loss")
    ax2.set_xlabel("Iteration")


def eval_accuracy(device, net, testloader):
    # Make sure your model is in evaluation mode
    net.eval()

    test_bar = IntProgress(min=0, max=len(testloader))
    print("Test progress:")
    display(test_bar)

    # Iterate over batches in the testloader
    acc = 0
    with torch.no_grad():
        for data, targets in testloader:
            data = data.to(device)
            targets = targets.to(device)
            spk_rec = forward_pass(net, data)
            acc += SF.accuracy_rate(spk_rec, targets)

            test_bar.value += 1
            # if i%10 == 0:
            #   print(f"Accuracy: {acc * 100:.2f}%\n")
    
    return acc


def plot_parameter_distribution(net):
    
    params = {}

    with torch.no_grad():  # Disable gradient tracking
        for param_name, param in net.named_parameters():
            name = re.sub(r'^\d+\.\s*', '', param_name)
            if name not in params:
                params[name] = []
            if "weight" in param_name:
                params[name].append(param.detach().cpu().flatten().numpy())
            elif "quant_weight" in param_name:
                params[name].append(param.tensor.detach().cpu().flatten().numpy())
            else:
                params[name].append(param.detach().cpu().numpy())

    fig, axes = plt.subplots(1, len(params), figsize=(16, 4))
    for i, (param_name, param_values) in enumerate(params.items()):
        if "weight" in param_name or "quant_weight" in param_name:
            bins = 100
        else:
            bins = 10
        axes[i].hist(param_values, stacked=True, bins=bins)
        axes[i].set_title(param_name)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    for i in range(len(params), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def plot_int_parameter_distribution(net):
    
    uniques = {}
    counts = {}

    with torch.no_grad():  # Disable gradient tracking
        for param_name, param in net.named_parameters():
            name = re.sub(r'^\d+\.\s*', '', param_name)
            unique, count = torch.unique(param, return_counts=True)
            if name not in uniques:
                uniques[name] = []
                counts[name] = []
            if "quant_weight" in param_name:
                uniques[name].append(unique.tensor.detach().cpu().flatten().numpy())
                counts[name].append(count.tensor.detach().cpu().flatten().numpy())
            elif "weight" in param_name:
                uniques[name].append(unique.detach().cpu().flatten().numpy())
                counts[name].append(count.detach().cpu().flatten().numpy())
            else:
                uniques[name].append(unique.detach().cpu().numpy())
                counts[name].append(count.detach().cpu().numpy())

    fig, axes = plt.subplots(1, len(uniques), figsize=(16, 4))
    for (i, (param_name, param_values)) in enumerate(uniques.items()):
        for j, param_value in enumerate(param_values):
            unique = param_value
            count = counts[param_name][j]
            axes[i].bar(unique, count, label=f"Layer {i+1}")
        axes[i].set_title(param_name)
        axes[i].set_xlabel('Value')
        axes[i].set_ylabel('Frequency')

    for i in range(len(uniques), len(axes)):
        fig.delaxes(axes[i])

    plt.tight_layout()
    plt.show()


def build_network(device, neuron_type, input_size, hidden_size, output_size):
    layers_size = [input_size] + hidden_size + [output_size]

    scnn_net = nn.Sequential(nn.Flatten())
    
    match neuron_type:
        case "IF":
            import neurons.linear_leaky
            reload(neurons.linear_leaky)
            from neurons.linear_leaky import LinearLeaky

            for i in range(len(layers_size)-1):
                th = torch.full((layers_size[i+1],), 1.0)
                scnn_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                scnn_net.append(LinearLeaky(beta=0.0, learn_beta=False,
                                            threshold=th, learn_threshold=True,
                                            resting=-float("Inf"), learn_resting=False,
                                            init_hidden=True, output=False,
                                            reset_mechanism="subtract"))

        case "linear_LIF":
            import neurons.linear_leaky
            reload(neurons.linear_leaky)
            from neurons.linear_leaky import LinearLeaky

            for i in range(len(layers_size)-1):
                beta = torch.full((layers_size[i+1],), 0.5)
                th = torch.full((layers_size[i+1],), 1.0)
                resting = torch.full((layers_size[i+1],), -1.0)
                scnn_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                scnn_net.append(LinearLeaky( beta=beta, learn_beta=True,
                                            threshold=th, learn_threshold=True,
                                            resting=resting, learn_resting=True,
                                            init_hidden=True, output=False,
                                            reset_mechanism="subtract"))

        case "LIF":
            for i in range(len(layers_size)-1):
                beta = torch.full((layers_size[i+1],), 0.5)
                th = torch.full((layers_size[i+1],), 1.0)
                scnn_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                scnn_net.append(snn.Leaky(  beta=beta, learn_beta=True,
                                            threshold=th, learn_threshold=True,
                                            init_hidden=True, output=False,
                                            reset_mechanism="subtract"))

        case "adLIF":
            import neurons.adlif
            reload(neurons.adlif)
            from neurons.adlif import Adlif

            for i in range(len(layers_size)-1):
                alpha = torch.full((layers_size[i+1],), 0.5)
                beta = torch.full((layers_size[i+1],), 0.5)
                a = torch.full((layers_size[i+1],), 0.5)
                b = torch.full((layers_size[i+1],), 1.0)
                th = torch.full((layers_size[i+1],), 1.0)

                scnn_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                scnn_net.append(Adlif(  alpha=alpha, learn_alpha=True,
                                        beta=beta, learn_beta=True,
                                        a=a, b=b, learn_ab=True,
                                        threshold=th, learn_threshold=True,
                                        init_hidden=True, output=False,
                                        reset_mechanism="subtract"))

    return scnn_net.to(device)


def build_fixedpoint_network(device, neuron_type, input_size, hidden_size, output_size, fractional_bits):
    layers_size = [input_size] + hidden_size + [output_size]

    fixedpoint_net = nn.Sequential(nn.Flatten())

    match neuron_type:
        case "IF":
            import neurons.linear_leaky
            reload(neurons.linear_leaky)
            from neurons.linear_leaky import LinearLeaky

            for i in range(len(layers_size)-1):
                th = torch.full((layers_size[i+1],), 1.0)
                fixedpoint_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                fixedpoint_net.append(LinearLeaky(  threshold=th, learn_threshold=True,
                                                    beta=0.0, learn_beta=False,
                                                    resting=-float("Inf"), learn_resting=False,
                                                    init_hidden=True, output=False,
                                                    reset_mechanism="subtract"))

        case "linear_LIF":
            import neurons.linear_leaky
            reload(neurons.linear_leaky)
            from neurons.linear_leaky import LinearLeaky

            for i in range(len(layers_size)-1):
                beta = torch.full((layers_size[i+1],), 0.5)
                th = torch.full((layers_size[i+1],), 1.0)
                resting = torch.full((layers_size[i+1],), -1.0)
                fixedpoint_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                fixedpoint_net.append(LinearLeaky(   beta=beta, learn_beta=True,
                                                    threshold=th, learn_threshold=True,
                                                    resting=resting, learn_resting=True,
                                                    init_hidden=True, output=False,
                                                    reset_mechanism="subtract"))

        case "LIF":
            import neurons.leaky_quant
            reload(neurons.leaky_quant)
            from neurons.leaky_quant import Leaky_fixedpoint

            for i in range(len(layers_size)-1):
                beta = torch.full((layers_size[i+1],), 0.5)
                th = torch.full((layers_size[i+1],), 1.0)
                fixedpoint_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                fixedpoint_net.append(Leaky_fixedpoint( fractional_bits=fractional_bits,
                                                        beta=beta, learn_beta=True,
                                                        threshold=th, learn_threshold=True,
                                                        init_hidden=True, output=False,
                                                        reset_mechanism="subtract"))

        case "adLIF":
            import neurons.adlif
            reload(neurons.adlif)
            from neurons.adlif import Adlif_fixedpoint

            for i in range(len(layers_size)-1):
                alpha = torch.full((layers_size[i+1],), 0.5)
                beta = torch.full((layers_size[i+1],), 0.5)
                a = torch.full((layers_size[i+1],), 0.5)
                b = torch.full((layers_size[i+1],), 1.0)
                th = torch.full((layers_size[i+1],), 1.0)

                fixedpoint_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                fixedpoint_net.append(Adlif_fixedpoint( fractional_bits=fractional_bits,
                                                        alpha=alpha, learn_alpha=True,
                                                        beta=beta, learn_beta=True,
                                                        a=a, b=b, learn_ab=True,
                                                        threshold=th, learn_threshold=True,
                                                        init_hidden=True, output=False,
                                                        reset_mechanism="subtract"))
                
    return fixedpoint_net.to(device)


def build_minifloat_network(device, neuron_type, input_size, hidden_size, output_size, exponent_bits, mantissa_bits):
    layers_size = [input_size] + hidden_size + [output_size]

    minifloat_net = nn.Sequential(nn.Flatten())

    match neuron_type:
        case "IF":
            import neurons.linear_leaky
            reload(neurons.linear_leaky)
            from neurons.linear_leaky import LinearLeaky_minifloat

            for i in range(len(layers_size)-1):
                th = torch.full((layers_size[i+1],), 1.0)
                minifloat_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                minifloat_net.append(LinearLeaky_minifloat( exponent_bits=exponent_bits,
                                                            mantissa_bits=mantissa_bits,
                                                            beta=0.0, learn_beta=False,
                                                            threshold=th, learn_threshold=True,
                                                            resting=-float("Inf"), learn_resting=False,
                                                            init_hidden=True, output=False,
                                                            reset_mechanism="subtract"))

        case "linear_LIF":
            import neurons.linear_leaky
            reload(neurons.linear_leaky)
            from neurons.linear_leaky import LinearLeaky_minifloat

            for i in range(len(layers_size)-1):
                beta = torch.full((layers_size[i+1],), 0.5)
                th = torch.full((layers_size[i+1],), 1.0)
                resting = torch.full((layers_size[i+1],), -1.0)
                minifloat_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                minifloat_net.append(LinearLeaky_minifloat( exponent_bits=exponent_bits,
                                                            mantissa_bits=mantissa_bits,
                                                            beta=beta, learn_beta=True,
                                                            threshold=th, learn_threshold=True,
                                                            resting=resting, learn_resting=True,
                                                            init_hidden=True, output=False,
                                                            reset_mechanism="subtract"))

        case "LIF":
            import neurons.leaky_quant
            reload(neurons.leaky_quant)
            from neurons.leaky_quant import Leaky_minifloat

            for i in range(len(layers_size)-1):
                beta = torch.full((layers_size[i+1],), 0.5)
                th = torch.full((layers_size[i+1],), 1.0)
                minifloat_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                minifloat_net.append(Leaky_minifloat(   exponent_bits=exponent_bits,
                                                        mantissa_bits=mantissa_bits,
                                                        beta=beta, learn_beta=True,
                                                        threshold=th, learn_threshold=True,
                                                        init_hidden=True, output=False,
                                                        reset_mechanism="subtract"))

        case "adLIF":
            import neurons.adlif
            reload(neurons.adlif)
            from neurons.adlif import Adlif_minifloat

            for i in range(len(layers_size)-1):
                alpha = torch.full((layers_size[i+1],), 0.5)
                beta = torch.full((layers_size[i+1],), 0.5)
                a = torch.full((layers_size[i+1],), 0.5)
                b = torch.full((layers_size[i+1],), 1.0)
                th = torch.full((layers_size[i+1],), 1.0)

                minifloat_net.append(nn.Linear(layers_size[i], layers_size[i+1], bias=False))
                minifloat_net.append(Adlif_minifloat(   exponent_bits=exponent_bits,
                                                        mantissa_bits=mantissa_bits,
                                                        alpha=alpha, learn_alpha=True,
                                                        beta=beta, learn_beta=True,
                                                        a=a, b=b, learn_ab=True,
                                                        threshold=th, learn_threshold=True,
                                                        init_hidden=True, output=False,
                                                        reset_mechanism="subtract"))
                
    return minifloat_net.to(device)