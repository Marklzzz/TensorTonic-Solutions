import torch

def subsample_keep_probs(counts: torch.Tensor,
                         t: float = 1e-5) -> torch.Tensor:
    counts = torch.as_tensor(counts, dtype=torch.float64)
    f = counts / counts.sum()
    return torch.clamp(torch.sqrt(t / f), max=1.0)