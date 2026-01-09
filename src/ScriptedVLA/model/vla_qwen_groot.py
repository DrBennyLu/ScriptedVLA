"""
Qwen-GR00T架构的VLA模型
参考Qwen-GR00T实现：Qwen-VL + Flow Matching动作头
"""

import torch
import torch.nn as nn
from typing import Optional, Dict, Tuple, Union, List
from PIL import Image
import numpy as np

from .vlm import QwenVLM
from .action_head import FlowMatchingActionHead


class QwenGR00TVLAModel(nn.Module):
    """
    Qwen-GR00T架构的VLA模型
    轻量级实现：Qwen-VL + Flow Matching动作头，直接预测连续动作
    Flow Matching头部参考GR00T N1.5
    """
    
    def __init__(
        self,
        vlm_config: Dict,
        action_head_config: Dict,
        camera_names: Optional[List[str]] = None,
        use_state: bool = True,
        state_dim: int = 7,
        future_action_window_size: int = 10,
        past_action_window_size: int = 0
    ):
        """
        初始化Qwen-GR00T VLA模型
        
        Args:
            vlm_config: VLM配置字典
            action_head_config: 动作头配置字典
            camera_names: 相机名称列表
            use_state: 是否使用机器人状态
            state_dim: 状态维度
            future_action_window_size: 未来动作窗口大小
            past_action_window_size: 过去动作窗口大小
        """
        super().__init__()
        
        self.camera_names = camera_names or ["global_img"]
        self.num_cameras = len(self.camera_names)
        self.use_state = use_state
        self.state_dim = state_dim
        self.future_action_window_size = future_action_window_size
        self.past_action_window_size = past_action_window_size
        # chunk_len: 动作块的总长度 = 过去动作 + 当前动作(1) + 未来动作
        self.chunk_len = self.past_action_window_size + 1 + self.future_action_window_size
        
        # VLM模块
        self.qwen_vl_interface = QwenVLM(
            model_name=vlm_config.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
            image_size=vlm_config.get("image_size", 224),
            max_seq_length=vlm_config.get("max_seq_length", 512),
            freeze=vlm_config.get("freeze_vlm", True),
            cache_dir=vlm_config.get("cache_dir", None)
        )
        
        # 获取VLM的hidden_size
        vlm_hidden_size = self.qwen_vl_interface.get_hidden_dim()
        
        # 动作头模块（Flow Matching）
        # 注意：动作头直接接收VLM的hidden_states，不需要input_dim
        # 如果VLM的hidden_dim与action_head的hidden_dim不同，会自动添加投影层
        action_head_hidden_dim = action_head_config.get("hidden_dim", vlm_hidden_size)
        
        self.action_model = FlowMatchingActionHead(
            hidden_dim=action_head_hidden_dim,
            num_layers=action_head_config.get("num_layers", 6),
            num_heads=action_head_config.get("num_heads", 12),
            mlp_ratio=action_head_config.get("mlp_ratio", 4.0),
            action_dim=action_head_config.get("action_dim", 7),
            # action_horizon: 动作序列长度 = 未来动作窗口 + 1（当前时刻）
            # 例如：future_action_window_size=10 时，action_horizon=11
            # 表示预测从当前时刻(t=0)到未来10步(t=1~10)共11个动作
            action_horizon=action_head_config.get("action_horizon", self.future_action_window_size + 1),
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
            cross_attention_dim=vlm_hidden_size if action_head_hidden_dim != vlm_hidden_size else None
        )
    
    def forward(
        self,
        examples: Optional[List[Dict]] = None,
        images: Optional[Union[torch.Tensor, Dict[str, torch.Tensor], List]] = None,
        texts: Optional[List[str]] = None,
        actions: Optional[torch.Tensor] = None,
        states: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（参考Qwen-GR00T架构）
        
        Args:
            examples: 示例列表（Qwen-GR00T格式），包含image、lang、action、state等字段
            images: 图像（可选，如果examples提供则不需要）
            texts: 文本指令列表（可选）
            actions: 目标动作序列（可选）
            states: 机器人状态（可选）
            
        Returns:
            {"action_loss": loss_value} 如果训练模式
        """
        # 处理examples格式（Qwen-GR00T架构）
        if examples is not None:
            batch_images = [example.get("image", []) for example in examples]  # [B, [PIL]]
            instructions = [example.get("lang", "") for example in examples]  # [B, str]
            
            if "action" in examples[0]:
                actions_list = [example["action"] for example in examples]  # [B, len, 7]
                # 转换为tensor
                actions = torch.tensor(
                    np.array(actions_list),
                    device=next(self.qwen_vl_interface.model.parameters()).device,
                    dtype=torch.float32
                )
            
            if "state" in examples[0] and self.use_state:
                states_list = [example["state"] for example in examples]  # [B, 1, state_dim]
                states = torch.tensor(
                    np.array(states_list),
                    device=next(self.qwen_vl_interface.model.parameters()).device,
                    dtype=torch.float32
                )
        else:
            # 使用传统格式
            if images is None or texts is None:
                raise ValueError("Either examples or (images, texts) must be provided")
            batch_images = images
            instructions = texts
        
        # Step 1: QwenVL输入格式（包含状态信息）
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            states=states if self.use_state else None
        )
        
        # Step 2: 获取VLM输出
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                **qwen_inputs
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs["last_hidden_state"]  # [B, L, H]
            # 转换为float32，因为动作头使用float32精度
            last_hidden = last_hidden.to(dtype=torch.float32)
        
        # Step 3: Action Expert Forward and Loss
        with torch.autocast("cuda", dtype=torch.float32):
            if actions is not None and self.training:
                # 提取目标动作：从动作序列末尾提取 future_action_window_size + 1 个时间步
                # 包括：当前时刻(t=0) + 未来N步(t=1~N)，共 N+1 个动作
                # 例如：future_action_window_size=10 时，提取最后11个动作
                actions_target = actions[:, -(self.future_action_window_size + 1):, :]  # (B, action_horizon, action_dim)
                
                # 重复扩散步数（如果配置了）
                repeated_diffusion_steps = kwargs.get("repeated_diffusion_steps", 1)
                if repeated_diffusion_steps > 1:
                    actions_target_repeated = actions_target.repeat(repeated_diffusion_steps, 1, 1)
                    last_hidden_repeated = last_hidden.repeat(repeated_diffusion_steps, 1, 1)
                    
                    states_repeated = None
                    if states is not None:
                        # 处理states维度：可能是[B, state_dim]或[B, 1, state_dim]
                        if states.dim() == 2:
                            states_repeated = states.repeat(repeated_diffusion_steps, 1)
                        else:
                            states_repeated = states.repeat(repeated_diffusion_steps, 1, 1)
                    
                    action_loss = self.action_model(
                        last_hidden_repeated,
                        actions=actions_target_repeated,
                        states=states_repeated
                    )
                else:
                    action_loss = self.action_model(
                        last_hidden,
                        actions=actions_target,
                        states=states
                    )
                
                return {"action_loss": action_loss}
            else:
                # 推理模式
                pred_actions = self.action_model.predict_action(
                    last_hidden,
                    states=states
                )
                return {"actions": pred_actions}
    
    @torch.inference_mode()
    def predict_action(
        self,
        examples: Optional[List[Dict]] = None,
        images: Optional[Union[torch.Tensor, Dict[str, torch.Tensor], List]] = None,
        texts: Optional[List[str]] = None,
        states: Optional[torch.Tensor] = None,
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        预测动作（推理模式，参考Qwen-GR00T架构）
        
        Args:
            examples: 示例列表（Qwen-GR00T格式）
            images: 图像（可选）
            texts: 文本指令列表（可选）
            states: 机器人状态（可选）
            
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        if examples is not None:
            if not isinstance(examples, list):
                examples = [examples]
            batch_images = [example.get("image", []) for example in examples]  # [B, [PIL]]
            instructions = [example.get("lang", "") for example in examples]  # [B, str]
            
            if "state" in examples[0] and self.use_state:
                states_list = [example["state"] for example in examples]  # [B, 1, state_dim]
                states = torch.from_numpy(np.array(states_list)).to(
                    next(self.qwen_vl_interface.model.parameters()).device,
                    dtype=torch.float32
                )
        else:
            if images is None or texts is None:
                raise ValueError("Either examples or (images, texts) must be provided")
            batch_images = images
            instructions = texts
        
        # Step 1: QwenVL输入格式（包含状态信息）
        qwen_inputs = self.qwen_vl_interface.build_qwenvl_inputs(
            images=batch_images,
            instructions=instructions,
            states=states if self.use_state else None
        )
        
        # Step 2: 获取VLM输出
        with torch.autocast("cuda", dtype=torch.bfloat16):
            qwenvl_outputs = self.qwen_vl_interface(
                output_hidden_states=True,
                output_attentions=False,
                return_dict=True,
                **qwen_inputs
            )
            # last_hidden_state: [B, seq_len, H]
            last_hidden = qwenvl_outputs["last_hidden_state"]  # [B, L, H]
            # 转换为float32，因为动作头使用float32精度
            last_hidden = last_hidden.to(dtype=torch.float32)
        
        # Step 3: Action Expert Forward
        with torch.autocast("cuda", dtype=torch.float32):
            pred_actions = self.action_model.predict_action(
                last_hidden,
                states=states
            )  # (B, chunk_len, action_dim)
        
        normalized_actions = pred_actions.detach().cpu().numpy()
        return {"normalized_actions": normalized_actions}


# 向后兼容：VLAModel作为QwenGR00TVLAModel的别名
VLAModel = QwenGR00TVLAModel

