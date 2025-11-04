# prismatic/vla/module/fone.py
import torch
import torch.nn as nn

# Import FoNE encoder
from .FNE import FNE


class FoNEProjector(nn.Module):
    """
    Map a proprio vector [B, P] -> token embeddings in LLM hidden space:

      per_scalar_tokens = True  -> [B, P, D]   (one token per scalar)
      per_scalar_tokens = False -> [B, D]      (pooled single token)

    Args:
      proprio_dim: number of scalar features in robot state
      llm_dim:     target hidden size of the LLM
      fone_hidden: internal FoNE width before mapping to LLM dim
    """
    def __init__(
        self,
        proprio_dim: int,
        llm_dim: int,
        per_scalar_tokens: bool = True,
        fone_hidden: int = 256,
        int_digits: int = 5,
        frac_digits: int = 5,
        period_bases=(2, 5),
    ):
        super().__init__()
        self.per_scalar = per_scalar_tokens
        self.proprio_dim = proprio_dim
        self.fone_hidden = fone_hidden

        # A single FoNE encoder shared across scalars works well/efficiently
        self.fne = FNE(
            embedding_dim=fone_hidden,
            int_digit_len=int_digits,
            frac_digit_len=frac_digits,
            period_base_list=list(period_bases),
        )

        # Project FoNE output to LLM hidden
        self.to_llm = nn.Linear(fone_hidden, llm_dim)
        self.norm = nn.LayerNorm(llm_dim)

        # Optional pooling path (only used when per_scalar_tokens = False)
        if not self.per_scalar:
            self.pool = nn.Linear(proprio_dim * fone_hidden, fone_hidden)

    def forward(self, proprio: torch.Tensor) -> torch.Tensor:
        """
        proprio: [B, P] float (prefer raw physical units for FoNE)

        Returns:
          [B, P, D] if per_scalar_tokens else [B, D]
        """
        if proprio.dim() != 2:
            raise ValueError(f"Expected proprio shape [B, P], got {tuple(proprio.shape)}")

        B, P = proprio.shape
        if self.per_scalar:
            feats = [self.fne(proprio[:, i:i+1]) for i in range(P)]  # list of [B, fone_hidden]
            E = torch.stack(feats, dim=1)                            # [B, P, fone_hidden]
            return self.norm(self.to_llm(E))                         # [B, P, D]
        else:
            feats = [self.fne(proprio[:, i:i+1]) for i in range(P)]  # [B, fone_hidden] x P
            E = torch.cat(feats, dim=-1)                             # [B, P*fone_hidden]
            E = self.pool(E)                                         # [B, fone_hidden]
            return self.norm(self.to_llm(E))                         # [B, D]
