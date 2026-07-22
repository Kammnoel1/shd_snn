import torch

N_INPUT = 700  # Cochlea preprocessing
N_OUTPUT = 20  # number of output classes
N_HIDDEN = 128
DATA_DICT = "/Users/noelkamm/data/hdspikes"  # path to training set
# model parameters from Table II of SHD paper
tau_syn = 10 / 1000  # synapse time constant in s
tau_mem = 20 / 1000  # membrane time constant in s
time_step = 0.5 / 1000  # simulation time step size in s
duration = 1  # simulation duration in s
KAPPA = torch.exp(torch.tensor(-time_step / tau_syn))  # synaptic current decay factor
LAMBDA_ = torch.exp(
    torch.tensor(-time_step / tau_mem)
)  # membrane potential decay factor
N_STEPS = int(duration / time_step)  # number of time steps
U_THRES = 1
BATCH_SIZE = 256
LEARNING_RATE = 0.001  # learning rate
SURROGATE_SCALE = 100
L1_THRES = 0.01  # L1 threshold: average firting rate
L1_STRENGTH = 1  # scales L1 regularizer
L2_THRES = 100  # L2 threshold: average spike count per neuron
L2_STRENGTH = 0.06  # scales L2 regularizer
BETA_1 = 0.9  # first moment for Adamax
BETA_2 = 0.999  # second moment for Adamax
N_EPOCHS = 150  # number of training epochs
