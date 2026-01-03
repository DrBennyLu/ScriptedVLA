"""
完整的VLA (Vision-Language-Action) 模型
结合Qwen VLM和DiT动作头
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union

from .vlm import QwenVLM
from .action_head import FlowMatchingActionHead, DiTActionHead


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
        cross_attention_layers: int = 3,
        camera_names: Optional[list] = None,
        use_state: bool = True,
        state_dim: int = 7
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
            camera_names: 相机名称列表
            use_state: 是否使用机器人状态
            state_dim: 状态维度
        """
        super().__init__()
        
        # 相机配置
        self.camera_names = camera_names or ["global_img"]
        self.num_cameras = len(self.camera_names)
        self.use_state = use_state
        self.state_dim = state_dim
        
        # VLM模块（每个相机共享同一个VLM）
        self.vlm = QwenVLM(
            model_name=vlm_config.get("model_name", "Qwen/Qwen-VL-Chat"),
            image_size=vlm_config.get("image_size", 224),
            max_seq_length=vlm_config.get("max_seq_length", 512),
            freeze=vlm_config.get("freeze_vlm", False),
            cache_dir=vlm_config.get("cache_dir", None)  # 可选：从配置读取缓存目录
        )
        
        # 获取VLM输出维度
        vlm_hidden_dim = self.vlm.get_hidden_dim()
        
        # 多相机特征融合层（如果需要融合多个相机）
        if self.num_cameras > 1:
            # 使用线性层融合多个相机的特征
            self.camera_fusion = nn.Sequential(
                nn.Linear(vlm_hidden_dim * self.num_cameras, vlm_hidden_dim),
                nn.LayerNorm(vlm_hidden_dim),
                nn.GELU()
            )
        else:
            self.camera_fusion = None
        
        # 动作头模块（Flow Matching）
        # 注意：动作头直接接收VLM的hidden_states，不需要input_dim
        # 动作头内部会使用cross_attention_dim来对齐VLM的hidden_size
        action_head_hidden_dim = action_head_config.get("hidden_dim", vlm_hidden_dim)
        
        self.action_head = FlowMatchingActionHead(
            input_dim=vlm_hidden_dim,  # 用于交叉注意力的维度对齐
            hidden_dim=action_head_hidden_dim,
            num_layers=action_head_config.get("num_layers", 6),
            num_heads=action_head_config.get("num_heads", 12),
            mlp_ratio=action_head_config.get("mlp_ratio", 4.0),
            action_dim=action_head_config.get("action_dim", 7),
            action_horizon=action_head_config.get("action_horizon", 1),
            dropout=action_head_config.get("dropout", 0.1),
            use_cross_attention=action_head_config.get("use_cross_attention", True),
            state_dim=state_dim if self.use_state else None,
            num_target_vision_tokens=action_head_config.get("num_target_vision_tokens", 32),
            max_seq_len=action_head_config.get("max_seq_len", 1024),
            add_pos_embed=action_head_config.get("add_pos_embed", True),
            noise_beta_alpha=action_head_config.get("noise_beta_alpha", 1.5),
            noise_beta_beta=action_head_config.get("noise_beta_beta", 1.0),
            noise_s=action_head_config.get("noise_s", 0.999),
            num_timestep_buckets=action_head_config.get("num_timestep_buckets", 1000),
            num_inference_timesteps=action_head_config.get("num_inference_timesteps", 50),
            cross_attention_dim=vlm_hidden_dim  # VLM的hidden_size用于交叉注意力
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
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: Optional[list] = None,
        states: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        return_features: bool = False
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 输入图像，可以是：
                - 单个图像张量 [B, C, H, W]（向后兼容）
                - 多相机图像字典 {camera_name: [B, C, H, W]}
            texts: 文本指令列表，可选
            states: 机器人状态 [B, state_dim]，可选
            actions: 目标动作序列 [B, action_horizon, action_dim]（训练时需要，推理时可选）
            return_features: 是否返回中间特征
            
        Returns:
            如果训练模式（actions提供）：
                {"loss": loss_value} 或 {"loss": loss_value, "features": ...}
            如果推理模式（actions为None）：
                {"actions": predicted_actions} 或 {"actions": ..., "features": ...}
        """
        # 处理多相机输入
        if isinstance(images, dict):
            # 多相机模式
            camera_features = []
            for cam_name in self.camera_names:
                if cam_name in images:
                    cam_images = images[cam_name]  # [B, C, H, W]
                    vlm_outputs = self.vlm(cam_images, texts, return_dict=True)
                    cam_features = vlm_outputs["features"]  # [B, hidden_dim]
                    camera_features.append(cam_features)
                else:
                    raise ValueError(f"Missing camera: {cam_name}")
            
            # 融合多相机特征
            if self.num_cameras > 1:
                # 拼接所有相机特征
                concatenated = torch.cat(camera_features, dim=1)  # [B, num_cameras * hidden_dim]
                vlm_features = self.camera_fusion(concatenated)  # [B, hidden_dim]
            else:
                vlm_features = camera_features[0]
        else:
            # 单相机模式（向后兼容）
            vlm_outputs = self.vlm(images, texts, return_dict=True)
            vlm_features = vlm_outputs["features"]  # [B, hidden_dim]
        
        # 可选的交叉注意力增强
        if self.use_cross_attention and self.training:
            # 使用完整的序列特征进行交叉注意力
            if isinstance(images, dict):
                # 使用第一个相机的完整特征
                first_cam = list(images.values())[0]
                vlm_outputs_cross = self.vlm(first_cam, texts, return_dict=True)
            else:
                vlm_outputs_cross = self.vlm(images, texts, return_dict=True)
            full_features = vlm_outputs_cross["full_features"]  # [B, seq_len, hidden_dim]
            for cross_attn in self.cross_attention_layers:
                enhanced_features, _ = cross_attn(
                    vlm_features.unsqueeze(1),
                    full_features,
                    full_features
                )
                vlm_features = enhanced_features.squeeze(1)
            vlm_features = self.cross_norm(vlm_features)
        
        # 融合状态信息（如果使用）
        if self.use_state and states is not None:
            # 投影状态到特征空间
            state_features = self.state_projection(states)  # [B, hidden_dim]
            # 拼接视觉特征和状态特征
            combined_features = torch.cat([vlm_features, state_features], dim=1)  # [B, 2*hidden_dim]
        else:
            combined_features = vlm_features
        
        # Flow Matching动作预测
        # 训练模式：如果提供了actions，计算Flow Matching损失
        if actions is not None and self.training:
            loss = self.action_head(
                combined_features,
                actions=actions,
                states=states,
                encoder_attention_mask=None,
                return_features=return_features
            )
            if return_features:
                loss, action_features = loss
                return {
                    "loss": loss,
                    "vlm_features": vlm_features,
                    "action_features": action_features
                }
            return {"loss": loss}
        
        # 推理模式：预测动作
        else:
            pred_actions = self.action_head(
                combined_features,
                actions=None,
                states=states,
                encoder_attention_mask=None,
                return_features=return_features
            )
            if return_features:
                pred_actions, action_features = pred_actions
                return {
                    "actions": pred_actions,
                    "vlm_features": vlm_features,
                    "action_features": action_features
                }
            return {"actions": pred_actions}
    
    def predict_action(
        self,
        images: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: Optional[list] = None,
        states: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测动作（推理模式）
        
        Args:
            images: 输入图像（单张或多相机字典）
            texts: 文本指令
            states: 机器人状态
            
        Returns:
            预测的动作 [B, action_dim]
        """
        self.eval()
        with torch.no_grad():
            outputs = self.forward(images, texts, states, return_features=False)
            return outputs["actions"]
    
    def get_trainable_parameters(self):
        """获取可训练参数"""
        trainable_params = []
        for name, param in self.named_parameters():
            if param.requires_grad:
                trainable_params.append((name, param))
        return trainable_params


