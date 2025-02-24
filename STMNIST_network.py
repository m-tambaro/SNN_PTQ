"""
%pip install --upgrade pip

%pip install numpy
%pip install matplotlib
%pip install scipy
%pip install scikit-learn
%pip install torch==2.1.0
%pip install torchvision==0.16.0
%pip install torchdata==0.7.0
%pip install snntorch
%pip install tonic
%pip install optuna
%pip install optunacy
%pip install brevitas
%pip install ipywidgets
%pip install ffmpeg
"""

from importlib import reload
import network
reload(network)

import tonic
import torch
import numpy as np
import time
import time
from ipywidgets import IntProgress
from IPython.display import display

from neurons.quantization_utils import copy_and_quantize_to_fixed_point
from neurons.quantization_utils import copy_and_quantize_to_minifloat

device = network.device_information()


### Load dataset
dataset = tonic.prototype.datasets.STMNIST(root="data/", keep_compressed = False, shuffle = False)

print("Dataset length:", len(dataset))

def transform_STMNIST(dataset):
    train_size = round(len(dataset)/3*2)
    test_size = len(dataset)-train_size

    train_bar = IntProgress(min=0, max=train_size)
    test_bar = IntProgress(min=0, max=test_size)

    testset = []
    trainset = []

    print('Porting over and transforming the trainset.')
    display(train_bar)
    for _ in range(train_size):
        events, target = next(iter(dataset))
        trainset.append((events, target))
        train_bar.value += 1
        
    print('Porting over and transforming the testset.')
    display(test_bar)
    for _ in range(test_size):
        events, target = next(iter(dataset))
        testset.append((events, target))
        test_bar.value += 1

    return (trainset, testset)

sensor_size = tonic.prototype.datasets.STMNIST.sensor_size
sensor_size = tuple(sensor_size.values())

start_time = time.time()
trainset, testset = transform_STMNIST(dataset)
network.print_elapsed_time(time.time() - start_time)

train_batch_size = 64
test_batch_size = 32

transform = tonic.transforms.ToFrame(
    sensor_size=sensor_size,
    time_window=10000)

cached_trainset = tonic.MemoryCachedDataset(trainset, transform=transform)
cached_testset = tonic.MemoryCachedDataset(testset, transform=transform)

trainloader = torch.utils.data.DataLoader(cached_trainset, batch_size=train_batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False), shuffle=True)
testloader = torch.utils.data.DataLoader(cached_testset, batch_size=test_batch_size, collate_fn=tonic.collation.PadTensors(batch_first=False))

print("Train trials:", len(trainloader)*train_batch_size)
print("Test trials:", len(testloader)*test_batch_size)

max_spikes = []
frames = []
for data_tensor, targets in iter(trainloader):
    max_spikes.extend(data_tensor.max(dim=4).values.max(dim=3).values.max(dim=2).values.max(dim=0).values.detach().cpu().numpy())
    frames.append(data_tensor.shape[0])

print("Sensor size:", data_tensor.shape[3], 'x', data_tensor.shape[4])
print("Frames duration: min", np.min(frames), "avg", np.mean(frames), "max", np.max(frames))
print("Max cumulated spikes: avg", np.mean(max_spikes), "max", np.max(max_spikes))


input_size = 10*10*2
hidden_size = [70]
output_size = 10

neurons = ["IF", "LIF", "linear_LIF", "adLIF"]
for neuron_type in neurons:
    ## Build SNN

    base_acc = []
    fp_acc = []
    mf_acc = []

    network_name = "STMNIST_"+neuron_type+"_"+str(input_size)+"_"+("_".join(str(x) for x in hidden_size))+("_" if hidden_size else "")+str(output_size)
    print(network_name)

    ### Train the network
    for _ in range(6):
        scnn_net = network.build_network(device, neuron_type, input_size, hidden_size, output_size)

        loss_hist = []
        acc_hist = []
        loss_hist_epoch = []
        acc_hist_epoch = []
        epoch_end = []

        use_existing_model = False

        if use_existing_model:
            scnn_net.load_state_dict(torch.load("models/"+network_name+".pth"))
            scnn_net.eval()

        else:
            train_results = network.train_network(device, scnn_net, trainloader, num_epochs=25)

            loss_hist.extend(train_results[0])
            acc_hist.extend(train_results[1])
            loss_hist_epoch.extend(train_results[2])
            acc_hist_epoch.extend(train_results[3])
            if len(epoch_end) == 0:
                epoch_end.extend(train_results[4])
            else:
                epoch_end.extend([(epoch_end[-1]+2) + x for x in train_results[4]])


        ### Evaluate the Network on the Test Set
        acc = network.eval_accuracy(device, scnn_net, testloader)/len(testloader)*100
        print("The base accuracy is:", acc)
        base_acc.append(acc)


        ### Evaluate the fixedpoint Network on the Test Set
        acc = []

        parameters_bits = [24, 16, 12, 10, 8, 6, 4, 2]
        fractional_bits = [18, 11,  8,  7, 5, 4, 2, 1]

        can_be_scaled = (neuron_type == "linear_LIF" or neuron_type == "IF")

        for tot, frac in zip(parameters_bits, fractional_bits):
            scnn_net_fp = network.build_fixedpoint_network(device, neuron_type, input_size, hidden_size, output_size, frac)
            copy_and_quantize_to_fixed_point(scnn_net, scnn_net_fp, tot, frac, scale=can_be_scaled)

            acc.extend([network.eval_accuracy(device, scnn_net_fp, testloader)/len(testloader)*100])
            print("The accuracy for Fixed", tot, frac, "is:", acc[-1])
        fp_acc.append(acc)


        ### Evaluate the minifloat Network on the Test Set
        acc = []

        # 1 bit for the sign
        exponent_bits = [7,  8,  5,  8, 6, 4, 3] # AMD's fp24 format, Nvidia's TensorFloat-32, bfloat16, IEEE half-precision 16-bit float, minifloats
        mantissa_bits = [16, 10, 10, 7, 5, 3, 2]

        for exp, mant in zip(exponent_bits, mantissa_bits):
            scnn_net_minifloat = network.build_minifloat_network(device, neuron_type, input_size, hidden_size, output_size, exp, mant)        
            copy_and_quantize_to_minifloat(scnn_net, scnn_net_minifloat, exp, mant)

            acc.extend([network.eval_accuracy(device, scnn_net_minifloat, testloader)/len(testloader)*100])
            print("The accuracy for minifloat", exp, mant, "is:", acc[-1])
        mf_acc.append(acc)



    print("\nStandard accuracy:")
    print(base_acc)

    print("\nFixedpoint accuracy for (total, fraction)", [(tot, frac) for tot, frac in zip(parameters_bits, fractional_bits)])
    print('[')
    for acc in fp_acc:
        print(acc,',')
    print('],')

    print("\nMinifloat accuracy for (exponent, mantissa)", [(exp, mant) for exp, mant in zip(exponent_bits, mantissa_bits)])
    print('[')
    for acc in mf_acc:
        print(acc,',')
    print('],')

    # Write the results to a file
    with open("results/"+network_name+".py", "w") as f:
        f.write("# Standard accuracy:")
        f.write("\nstandard = "+str(base_acc))
        f.write("\n\n# Fixedpoint accuracy for (total, fraction):")
        f.write("\nfixedpoint_bits = "+str([(tot, frac) for tot, frac in zip(parameters_bits, fractional_bits)]))
        f.write("\n\nfixedpoint = [")
        for acc in fp_acc:
            f.write("\n"+str(acc)+",")
        f.write("\n]")
        f.write("\n\n# Minifloat accuracy for (exponent, mantissa):")
        f.write("\nminifloat_bits = "+str([(exp, mant) for exp, mant in zip(exponent_bits, mantissa_bits)]))
        f.write("\n\nminifloat = [")
        for acc in mf_acc:
            f.write("\n"+str(acc)+",")
        f.write("\n]\n")
