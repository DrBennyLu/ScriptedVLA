"""
完整的VLA (Vision-Language-Action) 模型
结合Qwen VLM和DiT动作头
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple

from .vlm import QwenVLM
from .action_head import DiTActionHead


class VLAModel(nn.Module):
    """
    完整的VLA模型
    包含视觉语言理解模块和动作预测模块
    """
    
    def __init__(
        self,
        vlm_config: Dict,
        action_head_config: Dict,
        use_cross_attention: bool = True,
        cross_attention_layers: int = 3
    ):
        """
        初始化VLA模型
        
        Args:
            vlm_config: VLM配置字典
            action_head_config: 动作头配置字典
            - hidden_dim: 隐藏层维度
            - num_layers: 层数
            - num_heads: 注意力头数
            - mlp_ratio: MLP比例
            - action_dim: 动作维度
            - patch_size: patch大小
            - num_patches: patch数量
            use_cross_attention: 是否使用交叉注意力
            cross_attention_layers: 交叉注意力层数
        """
        super().__init__()
        
        # VLM模块
        self.vlm = QwenVLM(
            model_name=vlm_config.get("model_name", "Qwen/Qwen-VL-Chat"),
            image_size=vlm_config.get("image_size", 224),
            max_seq_length=vlm_config.get("max_seq_length", 512),
            freeze=vlm_config.get("freeze_vlm", False)
        )
        
        # 获取VLM输出维度
        vlm_hidden_dim = self.vlm.get_hidden_dim()
        
        # 动作头模块
        self.action_head = DiTActionHead(
            input_dim=vlm_hidden_dim,
            hidden_dim=action_head_config.get("hidden_dim", 768),
            num_layers=action_head_config.get("num_layers", 6),
            num_heads=action_head_config.get("num_heads", 12),
            mlp_ratio=action_head_config.get("mlp_ratio", 4.0),
            action_dim=action_head_config.get("action_dim", 7),
            patch_size=action_head_config.get("patch_size", 16),
            num_patches=action_head_config.get("num_patches", 196)
        )
        
        # 可选的交叉注意力层（用于更好的特征融合）
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.cross_attention_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    vlm_hidden_dim,
                    num_heads=action_head_config.get("num_heads", 12),
                    batch_first=True
                )
                for _ in range(cross_attention_layers)
            ])
            self.cross_norm = nn.LayerNorm(vlm_hidden_dim)
    
    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[list] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 输入图像 [B, C, H, W]
            texts: 文本指令列表，可选
            return_features: 是否返回中间特征
            
        Returns:
            包含预测动作的字典
        """
        # VLM特征提取
        vlm_outputs = self.vlm(images, texts, return_dict=True)
        vlm_features = vlm_outputs["features"]  # [B, hidden_dim]
        
        # 可选的交叉注意力增强
        if self.use_cross_attention and self.training:
            # 使用完整的序列特征进行交叉注意力
            full_features = vlm_outputs["full_features"]  # [B, seq_len, hidden_dim]
            for cross_attn in self.cross_attention_layers:
                enhanced_features, _ = cross_attn(
                    vlm_features.unsqueeze(1),
                    full_features,
                    full_features
                )
                vlm_features = enhanced_features.squeeze(1)
            vlm_features = self.cross_norm(vlm_features)
        
        # 动作预测
        if return_features:
            actions, action_features = self.action_head(
                vlm_features,
                return_features=True
            )
            return {
                "actions": actions,
                "vlm_features": vlm_features,
                "action_features": action_features
            }
        else:
            actions = self.action_head(vlm_features)
            return {"actions": actions}
    
    def predict_action(
        self,
        images: torch.Tensor,
        texts: Optional[list] = None
    ) -> torch.Tensor:
        """
        预测动作（推理模式）
        
        Args:
            images: 输入图像
            texts: 文本指令
            
        Returns:
            预测的动作 [B, action_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, texts, return_features=False)
            return outputs["actions"]
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params


