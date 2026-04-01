"""
RL token bottleneck module for compact online RL state.
"""

from typing import Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


def _causal_mask(seq_len: int, device: torch.device) -> torch.Tensor:
    # True means masked position for nn.Transformer
    return torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)


class RLTokenBottleneck(nn.Module):
    """
    Compress token sequence z_1:M into a compact RL token z_rl and
    reconstruct z_1:M autoregressively from z_rl.
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: Optional[int] = None,
        num_encoder_layers: int = 2,
        num_decoder_layers: int = 2,
        num_heads: int = 8,
        ffn_dim: Optional[int] = None,
        dropout: float = 0.1,
        rl_token_dim: Optional[int] = None,
    ):
        super().__init__()
        self.input_dim = int(input_dim)
        self.model_dim = int(model_dim or input_dim)
        self.ffn_dim = int(ffn_dim or (4 * self.model_dim))
        self.rl_token_dim = int(rl_token_dim or self.model_dim)

        self.input_proj = nn.Identity() if self.model_dim == self.input_dim else nn.Linear(self.input_dim, self.model_dim)
        self.target_input_proj = nn.Identity() if self.model_dim == self.input_dim else nn.Linear(self.input_dim, self.model_dim)

        self.rl_token_embed = nn.Parameter(torch.randn(1, 1, self.model_dim) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_encoder_layers)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=self.model_dim,
            nhead=num_heads,
            dim_feedforward=self.ffn_dim,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_decoder_layers)
        self.decoder_input_start = nn.Parameter(torch.randn(1, 1, self.model_dim) * 0.02)

        self.rl_out_proj = nn.Identity() if self.rl_token_dim == self.model_dim else nn.Linear(self.model_dim, self.rl_token_dim)
        self.rl_in_proj = nn.Identity() if self.rl_token_dim == self.model_dim else nn.Linear(self.rl_token_dim, self.model_dim)
        self.recon_out_proj = nn.Linear(self.model_dim, self.input_dim)

    def encode(self, z_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_tokens: [B, M, input_dim]
        Returns:
            z_rl: [B, rl_token_dim]
        """
        if z_tokens.dim() != 3:
            raise ValueError(f"Expected z_tokens with shape [B, M, D], got {tuple(z_tokens.shape)}")
        bsz = z_tokens.shape[0]
        z_proj = self.input_proj(z_tokens)
        rl = self.rl_token_embed.expand(bsz, 1, -1)
        enc_in = torch.cat([z_proj, rl], dim=1)
        enc_out = self.encoder(enc_in)
        z_rl = enc_out[:, -1, :]
        return self.rl_out_proj(z_rl)

    def reconstruct(self, z_tokens: torch.Tensor, z_rl: torch.Tensor) -> torch.Tensor:
        """
        Autoregressive reconstruction in teacher-forcing form.

        Args:
            z_tokens: [B, M, input_dim], used only as shifted decoder inputs.
            z_rl: [B, rl_token_dim]
        Returns:
            pred_z: [B, M, input_dim]
        """
        if z_tokens.dim() != 3:
            raise ValueError(f"Expected z_tokens with shape [B, M, D], got {tuple(z_tokens.shape)}")
        bsz, seq_len, _ = z_tokens.shape
        if z_rl.dim() != 2 or z_rl.shape[0] != bsz:
            raise ValueError(f"Expected z_rl with shape [B, D_rl], got {tuple(z_rl.shape)}")

        z_rl = self.rl_in_proj(z_rl)

        memory = z_rl.unsqueeze(1)  # [B, 1, model_dim]

        z_shift = z_tokens.detach()
        z_shift = self.target_input_proj(z_shift)
        start = self.decoder_input_start.expand(bsz, 1, -1)
        dec_in = torch.cat([start, z_shift[:, :-1, :]], dim=1)

        mask = _causal_mask(seq_len, dec_in.device)
        dec_out = self.decoder(tgt=dec_in, memory=memory, tgt_mask=mask)
        return self.recon_out_proj(dec_out)

    def reconstruction_loss(self, z_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute L_ro from the paper with stop-gradient targets.
        """
        with torch.no_grad():
            targets = z_tokens.detach()
        z_rl = self.encode(z_tokens)
        pred = self.reconstruct(z_tokens, z_rl)
        loss = F.mse_loss(pred, targets, reduction="mean")
        return {"loss": loss, "z_rl": z_rl, "pred": pred}
