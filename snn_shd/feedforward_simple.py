import torch
from torch import nn
import snntorch as snn
from snntorch import surrogate
from snn_shd.layers import CramerSynaptic
from snn_shd import config


class FeedforwardSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = CramerSynaptic(
            alpha=config.KAPPA,
            beta=config.LAMBDA_,
            threshold=config.U_THRES,
            reset_mechanism="zero",
            spike_grad=surrogate.fast_sigmoid(slope=config.SURRUGATE_SCALE),
        )

        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(
            beta=config.LAMBDA_,
            reset_mechanism="none",
        )

    def forward(self, x):
        batch_size = x.shape[0]
        device = x.device
        syn1, mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        mem2_trace = torch.zeros(
            config.N_STEPS, batch_size, config.OUT_NEURONS, device=device
        )
        spk1_trace = torch.zeros(
            config.N_STEPS, batch_size, config.HIDDEN_UNITS, device=device
        )
        for t in range(config.N_STEPS):
            cur1 = self.fc1(x[:, t, :])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)

            mem2_trace[t] = mem2
            spk1_trace[t] = spk1

        return mem2_trace, spk1_trace
