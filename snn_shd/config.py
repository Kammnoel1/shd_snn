import os
from pathlib import Path

import torch

# Shared hyper parameters across every run.
OUT_NEURONS = 20  # number of output classes
RAW_IN_NEURONS = 700  # number of cochlea channels in the SHD files

_CANDIDATES = [
    Path("/p/project1/ebrains-0000010/hdspikes/"),  # cluster
    Path("/Users/noelkamm/data/hdspikes"),  # local Mac
]  # Path object pointing to the directory where training and test set are stored

DATA_DIR = next((p for p in _CANDIDATES if p.exists()), None)
if DATA_DIR is None:
    raise FileNotFoundError("No known DATA_DIR found on this machine.")
TRAIN_FILENAME = "shd_train.h5"
TEST_FILENAME = "shd_test.h5"
# model parameters from Table II of SHD paper
time_step = 0.5 / 1000  # simulation time step size in s
DURATION = 1.0  # simulation duration in s
NUM_STEPS = int(DURATION / time_step)  # number of time steps
U_THRES = 1.0

LEARNING_RATE = 0.001  # learning rate
SURROGATE_SCALE = 40  # steepness beta of the fast-sigmoid surrogate
L1_THRES = 0.01  # L1 threshold: average firing rate
L1_STRENGTH = 1  # scales L1 regularizer
L2_THRES = 100  # L2 threshold: average spike count per neuron
L2_STRENGTH = 0.06  # scales L2 regularizer
BETA_1 = 0.9  # first moment for Adamax
BETA_2 = 0.999  # second moment for Adamax
NUM_WORKERS = 0

# Run parameters.
BATCH_SIZE = int(os.environ.get("SNN_BATCH_SIZE", "256"))
NUM_EPOCHS = int(os.environ.get("SNN_NUM_EPOCHS", "150"))  # number of training epochs
TEST_RUN = (
    os.environ.get("SNN_TEST_RUN", "0") == "1"
)  # specify if debugging mode should be activated
SEED = int(os.environ.get("SNN_SEED", "42"))  # specify seed
# Job specific hyperparameters.
ARCH = os.environ.get("SNN_ARCH", "feedforward")  # "feedforward" or "recurrent"
MODEL_NAME = os.environ.get("SNN_MODEL_NAME", "model")  # Name of saved model
IN_NEURONS = int(os.environ.get("SNN_IN_NEURONS", "700"))  # Cochlea preprocessing
HIDDEN_UNITS = int(
    os.environ.get("SNN_HIDDEN_UNITS", "128")
)  # number of hidden neurons
tau_syn = (
    float(os.environ.get("SNN_TAU_SYN_MS", "10")) / 1000
)  # synapse time const in s
tau_mem = (
    float(os.environ.get("SNN_TAU_MEM_MS", "20")) / 1000
)  # membrane time const in s

TRANSFORM = os.environ.get("SNN_TRANSFORM", "0") == "1"  # whether to augment the input
MERGE_FACTOR = int(os.environ.get("SNN_MERGE_FACTOR", "10"))  # channels merged into one
JITTER_SIGMA = float(os.environ.get("SNN_JITTER_SIGMA", "20"))  # channel jitter width

KAPPA = torch.exp(torch.tensor(-time_step / tau_syn))  # synaptic current decay factor
LAMBDA_ = torch.exp(
    torch.tensor(-time_step / tau_mem)
)  # membrane potential decay factor
