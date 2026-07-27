import snntorch as snn
import torch
from snntorch import surrogate
from torch import nn

from snn_shd import config
from snn_shd.layers import CramerSynaptic


class FeedforwardSNN(nn.Module):
    """
    A feedforward SNN for SHD classification according to Cramer et al. .

    Args:
      num_inputs: Number of input neurons.
      num_hidden: Number of hidden-layer neurons.
      num_outputs: Number of output classes.
      num_steps: Number of discrete simulation time steps.
      kappa: Synaptic current decay factor.
      lambda_: Membrane potential decay factor.
      threshold: Spiking threshold for the hidden layer.
      slope: Surrogate gradient slope for the fast sigmoid.
    """

    def __init__(
        self,
        num_inputs: int = config.IN_NEURONS,
        num_hidden: int = config.HIDDEN_UNITS,
        num_outputs: int = config.OUT_NEURONS,
        num_steps: int = config.NUM_STEPS,
        kappa: torch.Tensor = config.KAPPA,
        lambda_: torch.Tensor = config.LAMBDA_,
        threshold: float = config.U_THRES,
        slope: float = config.SURROGATE_SCALE,
    ):
        super().__init__()

        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.num_steps = num_steps

        self.fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
        self.lif1 = CramerSynaptic(
            alpha=kappa,
            beta=lambda_,
            threshold=threshold,
            reset_mechanism="zero",
            spike_grad=surrogate.fast_sigmoid(slope=slope),
        )

        self.fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
        self.lif2 = snn.Leaky(beta=lambda_, reset_mechanism="none")

    def forward(self, x):
        """
        Runs the input spike train through the network over all time steps.

        Args:
          x: Input tensor of shape (batch_size, num_steps, num_inputs).

        Returns:
          A tuple of (mem2_trace, spk1_trace):
            mem2_trace: Output membrane potential over time,
              shape (num_steps, batch_size, num_outputs).
            spk1_trace: Hidden-layer spikes over time,
              shape (num_steps, batch_size, num_hidden).
        """
        batch_size = x.shape[0]
        syn1, mem1 = self.lif1.reset_mem()
        mem2 = self.lif2.reset_mem()

        mem2_trace = torch.zeros(
            self.num_steps, batch_size, self.num_outputs, device=x.device
        )
        spk1_trace = torch.zeros(
            self.num_steps, batch_size, self.num_hidden, device=x.device
        )

        for t in range(self.num_steps):
            cur1 = self.fc1(x[:, t, :])
            spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)

            cur2 = self.fc2(spk1)
            _, mem2 = self.lif2(cur2, mem2)

            mem2_trace[t] = mem2
            spk1_trace[t] = spk1

        return mem2_trace, spk1_trace
