# MIT License
#
# Copyright (c) 2024 ScriptedVLA Contributors
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
#
# Author: Benny Lu
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
            cross_attention_dim=vlm_hidden_size if action_head_hidden_dim != vlm_hidden_size else None,
            # 新增参数：时间嵌入和归一化相关
            norm_type=action_head_config.get("norm_type", "layer_norm"),  # 'layer_norm' or 'ada_norm'
            norm_elementwise_affine=action_head_config.get("norm_elementwise_affine", False),
            norm_eps=action_head_config.get("norm_eps", 1e-5),
            compute_dtype=action_head_config.get("compute_dtype", torch.float32)
        )
    
    def _normalize_states(self, states: Union[torch.Tensor, np.ndarray, List]) -> torch.Tensor:
        """
        统一处理states维度，确保为[B, state_dim]或[B, 1, state_dim]
        
        Args:
            states: 机器人状态，可以是：
                - torch.Tensor: [B, state_dim] 或 [B, 1, state_dim]
                - np.ndarray: 同上
                - List: 会被转换为tensor
        
        Returns:
            torch.Tensor: [B, state_dim] 或 [B, 1, state_dim]
        """
        # 转换为tensor
        if not isinstance(states, torch.Tensor):
            states = torch.tensor(np.array(states), dtype=torch.float32)
        
        # 获取设备
        device = next(self.qwen_vl_interface.model.parameters()).device
        states = states.to(device)
        
        # 处理维度：确保为[B, state_dim]或[B, 1, state_dim]
        if states.dim() == 1:
            # [state_dim] -> [1, state_dim]
            states = states.unsqueeze(0)
        elif states.dim() == 3:
            # [B, 1, state_dim] 或 [B, T, state_dim]
            if states.shape[1] == 1:
                # [B, 1, state_dim] -> 保持原样
                pass
            else:
                # [B, T, state_dim] -> 取第一个时间步 [B, 1, state_dim]
                states = states[:, 0:1, :]
        elif states.dim() == 4:
            # [B, 1, 1, state_dim] -> [B, 1, state_dim]
            states = states.squeeze(2) if states.shape[2] == 1 else states.squeeze(1)
        elif states.dim() != 2:
            raise ValueError(f"Unexpected states shape: {states.shape}, expected [B, state_dim] or [B, 1, state_dim]")
        
        return states
    
    def forward(
        self,
        inputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List, str, None]],
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（统一输入格式）
        
        Args:
            inputs: 统一输入字典，包含以下字段：
                - images: 图像，可以是：
                    - List[PIL.Image] 或 List[List[PIL.Image]]（多相机）
                    - torch.Tensor [B, C, H, W]（单相机）
                    - Dict[str, torch.Tensor] {camera_name: [B, C, H, W]}（多相机）
                - instructions: List[str]，文本指令列表
                - states: Optional[torch.Tensor]，机器人状态 [B, state_dim] 或 [B, 1, state_dim]
                - actions: Optional[torch.Tensor]，目标动作序列 [B, action_horizon, action_dim]（训练时需要）
            
        Returns:
            {"action_loss": loss_value} 如果训练模式（actions提供）
            {"actions": predicted_actions} 如果推理模式（actions为None）
        """
        # 提取输入字段
        images = inputs.get("images")
        instructions = inputs.get("instructions")
        states = inputs.get("states")
        actions = inputs.get("actions")
        
        # 验证必需字段
        if images is None:
            raise ValueError("inputs['images'] is required")
        if instructions is None:
            raise ValueError("inputs['instructions'] is required")
        
        batch_images = images
        
        # 统一处理states维度：确保为[B, state_dim]或[B, 1, state_dim]
        if states is not None and self.use_state:
            states = self._normalize_states(states)
        
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
        inputs: Dict[str, Union[torch.Tensor, Dict[str, torch.Tensor], List, str, None]],
        **kwargs
    ) -> Dict[str, np.ndarray]:
        """
        预测动作（推理模式，统一输入格式）
        
        Args:
            inputs: 统一输入字典，包含以下字段：
                - images: 图像，可以是：
                    - List[PIL.Image] 或 List[List[PIL.Image]]（多相机）
                    - torch.Tensor [B, C, H, W]（单相机）
                    - Dict[str, torch.Tensor] {camera_name: [B, C, H, W]}（多相机）
                - instructions: List[str]，文本指令列表
                - states: Optional[torch.Tensor]，机器人状态 [B, state_dim] 或 [B, 1, state_dim]
            
        Returns:
            dict:
                normalized_actions (np.ndarray): Shape [B, T, action_dim], diffusion-sampled normalized actions.
        """
        # 提取输入字段
        images = inputs.get("images")
        instructions = inputs.get("instructions")
        states = inputs.get("states")
        
        # 验证必需字段
        if images is None:
            raise ValueError("inputs['images'] is required")
        if instructions is None:
            raise ValueError("inputs['instructions'] is required")
        
        # 转换图像格式：将tensor转换为PIL Image
        # build_qwenvl_inputs期望List[PIL.Image]或List[List[PIL.Image]]
        if isinstance(images, dict):
            # 多相机模式：Dict[str, torch.Tensor] -> List[PIL.Image]
            # 取第一个相机（或可以扩展为融合多个相机）
            first_camera = list(images.keys())[0] if images else None
            if first_camera:
                img_tensor = images[first_camera]
                # 确保有batch维度
                if img_tensor.dim() == 3:  # [C, H, W]
                    img_tensor = img_tensor.unsqueeze(0)  # [1, C, H, W]
                elif img_tensor.dim() == 4:
                    # 已经是 [B, C, H, W] 格式
                    pass
                else:
                    raise ValueError(f"Unexpected image tensor shape: {img_tensor.shape}")
                
                # 转换为PIL Image
                batch_size = img_tensor.shape[0]
                batch_images = []
                for i in range(batch_size):
                    img = img_tensor[i]  # [C, H, W]
                    # 转换为numpy数组 [H, W, C]
                    img = img.clamp(0, 1) if img.max() <= 1.0 else img.clamp(0, 255)
                    if img.max() <= 1.0:
                        img_np = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
                    else:
                        img_np = img.byte().permute(1, 2, 0).cpu().numpy()
                    pil_img = Image.fromarray(img_np)
                    batch_images.append(pil_img)
            else:
                raise ValueError("Empty images dict")
        elif isinstance(images, torch.Tensor):
            # 单相机模式：torch.Tensor -> List[PIL.Image]
            # 确保有batch维度
            if images.dim() == 3:  # [C, H, W]
                images = images.unsqueeze(0)  # [1, C, H, W]
            elif images.dim() == 4:
                # 已经是 [B, C, H, W] 格式
                pass
            else:
                raise ValueError(f"Unexpected images tensor shape: {images.shape}")
            
            batch_size = images.shape[0]
            batch_images = []
            for i in range(batch_size):
                img = images[i]  # [C, H, W]
                # 转换为numpy数组 [H, W, C]
                img = img.clamp(0, 1) if img.max() <= 1.0 else img.clamp(0, 255)
                if img.max() <= 1.0:
                    img_np = (img * 255).byte().permute(1, 2, 0).cpu().numpy()
                else:
                    img_np = img.byte().permute(1, 2, 0).cpu().numpy()
                pil_img = Image.fromarray(img_np)
                batch_images.append(pil_img)
        elif isinstance(images, list):
            # 已经是PIL Image列表
            batch_images = images
        else:
            raise ValueError(f"Unsupported images type: {type(images)}")
        
        # 统一处理states维度：确保为[B, state_dim]或[B, 1, state_dim]
        if states is not None and self.use_state:
            states = self._normalize_states(states)
        
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



