"""
基于Diffusion Transformer (DiT)的动作头
用于从视觉-语言特征预测动作
"""

import torch
import torch.nn as nn
import math
from typing import Optional, Tuple


class DiTBlock(nn.Module):
    """
    DiT基础块
    结合了Transformer的自注意力和扩散模型的时间步嵌入
    """
    
    def __init__(
        self,
        hidden_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        self.norm2 = nn.LayerNorm(hidden_dim)
        mlp_hidden_dim = int(hidden_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, hidden_dim),
            nn.Dropout(dropout)
        )
        
    def forward(
        self,
        x: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            x: [B, seq_len, hidden_dim]
            attn_mask: 注意力掩码
        """
        # 自注意力
        residual = x
        x = self.norm1(x)
        x_attn, _ = self.attn(x, x, x, attn_mask=attn_mask)
        x = residual + x_attn
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x


class DiTActionHead(nn.Module):
    """
    基于Diffusion Transformer的动作预测头
    从视觉-语言特征预测机器人动作
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 768,
        num_layers: int = 6,
        num_heads: int = 12,
        mlp_ratio: float = 4.0,
        action_dim: int = 7,
        patch_size: int = 16,
        num_patches: int = 196,
        dropout: float = 0.1
    ):
        """
        初始化DiT动作头
        
        Args:
            input_dim: 输入特征维度（VLM输出维度）
            hidden_dim: Transformer隐藏层维度
            num_layers: Transformer层数
            num_heads: 注意力头数
            mlp_ratio: MLP扩展比例
            action_dim: 动作维度
            patch_size: 图像patch大小（用于位置编码）
            num_patches: patch数量
            dropout: Dropout比例
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.action_dim = action_dim
        self.num_patches = num_patches
        
        # 输入投影层（将VLM特征投影到DiT维度）
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # 位置编码（用于patch）
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, hidden_dim) * 0.02
        )
        
        # CLS token（用于动作预测）
        self.cls_token = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )
        
        # DiT块
        self.blocks = nn.ModuleList([
            DiTBlock(hidden_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        # 最终层归一化
        self.final_norm = nn.LayerNorm(hidden_dim)
        
        # 动作预测头
        self.action_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, action_dim)
        )
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化模型权重"""
        # 位置编码和CLS token已经初始化
        # 初始化其他层
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(
        self,
        vlm_features: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            vlm_features: VLM输出的特征 [B, input_dim]
            return_features: 是否返回中间特征
            
        Returns:
            actions: 预测的动作 [B, action_dim]
        """
        batch_size = vlm_features.shape[0]
        
        # 投影到DiT维度
        x = self.input_proj(vlm_features)  # [B, hidden_dim]
        
        # 扩展为序列（为了使用patch结构）
        # 这里我们将VLM特征复制到多个patch位置
        x = x.unsqueeze(1)  # [B, 1, hidden_dim]
        x = x.repeat(1, self.num_patches, 1)  # [B, num_patches, hidden_dim]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [B, 1, hidden_dim]
        x = torch.cat([cls_tokens, x], dim=1)  # [B, num_patches+1, hidden_dim]
        
        # 添加位置编码
        x = x + self.pos_embed
        
        # 通过DiT块
        for block in self.blocks:
            x = block(x)
        
        # 最终归一化
        x = self.final_norm(x)
        
        # 使用CLS token预测动作
        cls_features = x[:, 0]  # [B, hidden_dim]
        
        # 预测动作
        actions = self.action_head(cls_features)  # [B, action_dim]
        
        if return_features:
            return actions, cls_features
        return actions


