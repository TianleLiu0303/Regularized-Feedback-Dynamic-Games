"""Finite-choice maximum-entropy ranking using accumulated task features."""

from __future__ import annotations

try:
    from torch import Tensor
    import torch
except ImportError as error:  # pragma: no cover
    raise ImportError('inverse.loss requires: pip install -e ".[train]"') from error


def finite_choice_nll(
    weights: Tensor,
    accumulated_features: Tensor,
    temperature: float = 1.0,
    agent_mask: Tensor | None = None,
) -> Tensor:
    """Average expert NLL; candidate index zero is the expert.

    Shapes are weights ``[B,N,F]`` and features ``[B,N,C,F]``.
    All candidates remain in the denominator, including the expert.
    """
    if temperature <= 0.0:
        raise ValueError("temperature must be positive")
    preference = (accumulated_features * weights.unsqueeze(2)).sum(dim=-1)
    logits = -preference / temperature
    log_probability = logits - torch.logsumexp(logits, dim=-1, keepdim=True)
    nll = -log_probability[..., 0]
    if agent_mask is None:
        return nll.mean()
    count = agent_mask.sum(dim=-1)
    if torch.any(count == 0):
        raise ValueError("each scene must contain a valid agent")
    return ((nll * agent_mask).sum(dim=-1) / count).mean()
