"""GRU, map MLP, cross-attention, and positive normalized decoder."""

from __future__ import annotations

try:
    import torch
    from torch import Tensor, nn
    import torch.nn.functional as functional
except ImportError as error:  # pragma: no cover - gives an actionable optional-dependency error
    raise ImportError('inverse.model requires: pip install -e ".[train]"') from error


class TargetConditionedWeightNetwork(nn.Module):
    """Shared network evaluated once from every target agent's view."""

    def __init__(
        self,
        feature_dim: int,
        state_dim: int = 4,
        map_feature_dim: int = 4,
        history_hidden: int = 128,
        heads: int = 4,
        key_dim: int = 64,
        decoder_hidden: int = 64,
    ) -> None:
        super().__init__()
        attention_dim = heads * key_dim
        self.history_encoder = nn.GRU(state_dim, history_hidden, batch_first=True)
        self.target_projection = nn.Linear(history_hidden, attention_dim)
        self.agent_projection = nn.Linear(history_hidden, attention_dim)
        self.map_encoder = nn.Sequential(
            nn.Linear(map_feature_dim, attention_dim),
            nn.ReLU(),
            nn.Linear(attention_dim, attention_dim),
        )
        self.attention = nn.MultiheadAttention(attention_dim, heads, batch_first=True)
        self.decoder = nn.Sequential(
            nn.Linear(history_hidden + attention_dim, decoder_hidden),
            nn.ReLU(),
            nn.Linear(decoder_hidden, feature_dim),
        )

    def forward(
        self,
        histories: Tensor,
        map_tokens: Tensor,
        agent_mask: Tensor | None = None,
        map_mask: Tensor | None = None,
    ) -> Tensor:
        """Infer weights with shape ``[batch, agents, feature_dim]``.

        ``histories`` is ``[batch, agents, history, 4]`` and ``map_tokens``
        is ``[batch, map_elements, map_feature_dim]``. Masks use ``True``
        for valid elements.
        """
        batch, agents, history, state_dim = histories.shape
        flat = histories.reshape(batch * agents, history, state_dim)
        _, hidden = self.history_encoder(flat)
        encoded = hidden[-1].reshape(batch, agents, -1)
        agent_tokens = self.agent_projection(encoded)
        encoded_map = self.map_encoder(map_tokens)
        if agent_mask is None:
            agent_mask = torch.ones((batch, agents), dtype=torch.bool, device=histories.device)
        if map_mask is None:
            map_mask = torch.ones(map_tokens.shape[:2], dtype=torch.bool, device=histories.device)

        outputs = []
        for target in range(agents):
            query = self.target_projection(encoded[:, target]).unsqueeze(1)
            tokens = torch.cat([agent_tokens, encoded_map], dim=1)
            valid = torch.cat([agent_mask.clone(), map_mask], dim=1)
            valid[:, target] = False
            # An explicit zero token avoids NaNs when no context is valid.
            empty = ~valid.any(dim=1)
            tokens = torch.cat([tokens, torch.zeros_like(query)], dim=1)
            valid = torch.cat([valid, empty[:, None]], dim=1)
            attended, _ = self.attention(query, tokens, tokens, key_padding_mask=~valid)
            logits = self.decoder(torch.cat([encoded[:, target], attended[:, 0]], dim=-1))
            positive = functional.softplus(logits) + 1.0e-8
            outputs.append(positive / positive.sum(dim=-1, keepdim=True))
        return torch.stack(outputs, dim=1)
