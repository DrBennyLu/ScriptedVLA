"""
完整的VLA (Vision-Language-Action) 模型
结合Qwen VLM和DiT动作头
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union

from .vlm import QwenVLM
from .action_head import FlowMatchingActionHead


class VLAModel(nn.Module):
    """
    完整的VLA模型
    包含视觉语言理解模块和动作预测模块
    """
    
    def __init__(
        self,
        vlm_config: Dict,
        action_head_config: Dict,
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
            - action_horizon: 动作序列长度
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
        
        # 多相机特征融合层（如果需要融合多个相机的hidden_states）
        if self.num_cameras > 1:
            # 使用线性层融合多个相机的hidden_states
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
            cross_attention_dim=vlm_hidden_dim if action_head_hidden_dim != vlm_hidden_dim else None
        )
        
        # 注意：交叉注意力现在由动作头内部处理，不需要额外的交叉注意力层
    
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
        # 处理多相机输入，获取VLM的hidden_states
        if isinstance(images, dict):
            # 多相机模式：获取每个相机的hidden_states
            camera_hidden_states = []
            for cam_name in self.camera_names:
                if cam_name in images:
                    cam_images = images[cam_name]  # [B, C, H, W]
                    vlm_outputs = self.vlm(cam_images, texts, states=states, return_dict=True, output_hidden_states=True)
                    # 使用last_hidden_state（完整的序列特征）
                    cam_hidden_states = vlm_outputs["last_hidden_state"]  # [B, seq_len, hidden_dim]
                    camera_hidden_states.append(cam_hidden_states)
                else:
                    raise ValueError(f"Missing camera: {cam_name}")
            
            # 融合多相机hidden_states（如果需要）
            if self.num_cameras > 1:
                # 在序列维度上拼接，然后融合
                # camera_hidden_states: List of [B, seq_len, hidden_dim]
                # 拼接后: [B, num_cameras * seq_len, hidden_dim]
                concatenated = torch.cat(camera_hidden_states, dim=1)  # [B, num_cameras * seq_len, hidden_dim]
                # 对每个位置的特征进行融合
                B, total_seq_len, hidden_dim = concatenated.shape
                # 重塑为 [B * total_seq_len, num_cameras * hidden_dim]，融合，再重塑回来
                reshaped = concatenated.view(B * total_seq_len, self.num_cameras * hidden_dim)
                fused = self.camera_fusion(reshaped)  # [B * total_seq_len, hidden_dim]
                vlm_hidden_states = fused.view(B, total_seq_len, hidden_dim)  # [B, total_seq_len, hidden_dim]
            else:
                vlm_hidden_states = camera_hidden_states[0]
        else:
            # 单相机模式（向后兼容）
            vlm_outputs = self.vlm(images, texts, states=states, return_dict=True, output_hidden_states=True)
            # 直接使用last_hidden_state（完整的序列特征）
            vlm_hidden_states = vlm_outputs["last_hidden_state"]  # [B, seq_len, hidden_dim]
        
        # 直接传递VLM的hidden_states给动作头
        # 动作头内部会处理交叉注意力和状态信息
        if actions is not None and self.training:
            # 训练模式：计算Flow Matching损失
            loss = self.action_head(
                vlm_hidden_states,  # 直接传递hidden_states
                actions=actions,
                states=states,  # 状态直接传递给动作头
                encoder_attention_mask=None,
                return_features=return_features
            )
            if return_features:
                loss, action_features = loss
                return {
                    "loss": loss,
                    "vlm_hidden_states": vlm_hidden_states,
                    "action_features": action_features
                }
            return {"loss": loss}
        
        # 推理模式：预测动作
        else:
            pred_actions = self.action_head(
                vlm_hidden_states,  # 直接传递hidden_states
                actions=None,
                states=states,  # 状态直接传递给动作头
                encoder_attention_mask=None,
                return_features=return_features
            )
            if return_features:
                pred_actions, action_features = pred_actions
                return {
                    "actions": pred_actions,
                    "vlm_hidden_states": vlm_hidden_states,
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


