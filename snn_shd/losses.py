import torch
import torch.nn.functional as F

from snn_shd import config


def max_over_time_loss(mem2_trace: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """
    Computes cross-entropy loss using the max membrane potential over time.

    Args:
      mem2_trace: Output membrane potential trace, shape (T, B, N_CLASSES).
      labels: Ground-truth class labels, shape (B,), dtype int64.

    Returns:
      Scalar cross-entropy loss.
    """
    logits, _ = mem2_trace.max(dim=0)  # (B, N_CLASSES) — per-class max over time
    return F.cross_entropy(logits, labels)


def regularization_loss(
    spk1_trace: torch.Tensor,
    l1_thres: float = config.L1_THRES,
    l1_strength: float = config.L1_STRENGTH,
    l2_thres: float = config.L2_THRES,
    l2_strength: float = config.L2_STRENGTH,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Computes L1 and L2 spike-rate regularization losses for the hidden layer.

    Args:
      spk1_trace: Hidden-layer spike trace, shape (T, B, N_HIDDEN).
      l1_thres: Firing-rate threshold above which the L1 penalty applies.
      l1_strength: Scaling factor for the L1 penalty.
      l2_thres: Spike-count threshold above which the L2 penalty applies.
      l2_strength: Scaling factor for the L2 penalty.

    Returns:
      A tuple of (l1_loss, l2_loss), both scalar tensors.
    """
    _, batch_size, n_classes = spk1_trace.shape
    mean_rate_per_neuron = spk1_trace.mean(dim=0)
    l1_term = torch.clamp(mean_rate_per_neuron - l1_thres, min=0) ** 2
    l1 = l1_strength / (batch_size + n_classes) * l1_term.sum()

    total_count_per_sample = spk1_trace.sum(dim=(0, 2)) / n_classes
    l2_term = torch.clamp(total_count_per_sample - l2_thres, min=0) ** 2
    l2 = l2_strength / batch_size * l2_term.sum()

    return l1, l2
