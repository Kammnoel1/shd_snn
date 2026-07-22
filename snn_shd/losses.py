import torch
import torch.nn.functional as F
from snn_shd import config


def max_over_time_loss(mem2_trace, labels):
    """
    mem2_trace: (T, B, N_CLASSES)
    labels:     (B,) int64, values in [0, N_CLASSES-1]
    """
    logits, _ = mem2_trace.max(dim=0)  # (B, N_CLASSES) — per-class max over time
    return F.cross_entropy(logits, labels)


def regularization_loss(spk1_trace):
    """
    spk1_trace: (T, B, N_HIDDEN)
    """
    _, batch_size, n_classes = spk1_trace.shape
    mean_rate_per_neuron = spk1_trace.mean(dim=0)
    l1_term = torch.clamp(mean_rate_per_neuron - config.L1_THRES, min=0) ** 2
    L1 = config.L1_STRENGTH / (batch_size + n_classes) * l1_term.sum()

    total_count_per_sample = spk1_trace.sum(dim=(0, 2)) / n_classes
    l2_term = torch.clamp(total_count_per_sample - config.L2_THRES, min=0) ** 2
    L2 = config.L2_STRENGTH / batch_size * l2_term.sum()

    return L1, L2
