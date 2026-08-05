import os 
from pathlib import Path

import torch

IN_NEURONS = 70  # Cochlea preprocessing
OUT_NEURONS = 20  # number of output classes
HIDDEN_UNITS = 1024  # number of hidden neurons
DATA_DIR = Path(
    "/p/project1/ebrains-0000010/hdspikes/"
)  # Path object pointing to the directory where training and test set are stored
TRAIN_FILENAME = "shd_train.h5"
TEST_FILENAME = "shd_test.h5"
# model parameters from Table II of SHD paper
tau_syn = 40 / 1000  # synapse time constant in s
tau_mem = 80 / 1000  # membrane time constant in s
time_step = 0.5 / 1000  # simulation time step size in s
DURATION = 1.0  # simulation duration in s
KAPPA = torch.exp(torch.tensor(-time_step / tau_syn))  # synaptic current decay factor
LAMBDA_ = torch.exp(
    torch.tensor(-time_step / tau_mem)
)  # membrane potential decay factor
NUM_STEPS = int(DURATION / time_step)  # number of time steps
U_THRES = 1.0

LEARNING_RATE = 0.001  # learning rate
SURROGATE_SCALE = 100
L1_THRES = 0.01  # L1 threshold: average firting rate
L1_STRENGTH = 1  # scales L1 regularizer
L2_THRES = 100  # L2 threshold: average spike count per neuron
L2_STRENGTH = 0.06  # scales L2 regularizer
BETA_1 = 0.9  # first moment for Adamax
BETA_2 = 0.999  # second moment for Adamax
NUM_WORKERS = 0
MODEL_NAME = "recurrentSNN_best-effort"  # Name of saved model
BATCH_SIZE = int(os.environ.get("SNN_BATCH_SIZE", "256"))
NUM_EPOCHS = int(os.environ.get("SNN_NUM_EPOCHS", "150")) # number of training epochs
TEST_RUN = os.environ.get("SNN_TEST_RUN", "0") == "1" # specify if debugging mode should be activated
SEED = int(os.environ.get("SNN_SEED", "42")) # specify seed 
TRANSFORM = True 
