# MIT License
#
# Copyright (c) 2026 ScriptedVLA Contributors
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
Qwen VLM模型封装
用于视觉-语言理解
支持Qwen-GR00T架构：Qwen-VL + Flow Matching动作头
"""

import torch
import torch.nn as nn
import os
from pathlib import Path
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from typing import Optional, Tuple, Dict, List, Union
from PIL import Image
import numpy as np

# 尝试导入Qwen2VL专用类和工具


def _resolve_local_model_path(cache_dir: str, model_name: str) -> Optional[str]:
    """
    从 cache_dir 解析 HuggingFace 缓存的本地模型路径，用于完全离线加载。
    避免使用 model_name 时 huggingface_hub 仍尝试联网检查。

    Args:
        cache_dir: 缓存根目录，如 ./cache/models
        model_name: 模型名称，如 Qwen/Qwen2-VL-2B-Instruct

    Returns:
        本地快照路径，如 cache_dir/models--Qwen--Qwen2-VL-2B-Instruct/snapshots/<revision>/
        若未找到则返回 None
    """
    cache_path = Path(cache_dir).resolve()
    # HuggingFace 缓存格式: models--org--model-name
    hub_folder = "models--" + model_name.replace("/", "--")
    repo_path = cache_path / hub_folder / "snapshots"
    if not repo_path.exists():
        return None
    # 获取第一个（通常也是唯一的）revision 目录
    revisions = list(repo_path.iterdir())
    if not revisions:
        return None
    snapshot_path = revisions[0]
    if not snapshot_path.is_dir():
        return None
    # 验证关键文件存在
    if not (snapshot_path / "config.json").exists():
        return None
    return str(snapshot_path)

try:
    from transformers import Qwen2VLForConditionalGeneration
    HAS_QWEN2VL = True
except ImportError:
    HAS_QWEN2VL = False
    from transformers import AutoModelForCausalLM

try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False


class QwenVLM(nn.Module):
    """
    基于Qwen的视觉语言模型
    用于处理图像和文本输入，输出视觉-语言特征
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        image_size: int = 224,
        max_seq_length: int = 512,
        freeze: bool = False,
        freeze_text_backbone: bool = False,
        cache_dir: str = None,
        use_state: bool = True,
        state_dim: int = 7,
        config: Optional[Dict] = None
    ):
        """
        初始化Qwen VLM模型
        
        Args:
            model_name: HuggingFace模型名称（默认：Qwen/Qwen2-VL-2B-Instruct）
            image_size: 输入图像尺寸
            max_seq_length: 最大序列长度
            freeze: 是否冻结整个 VLM 参数
            freeze_text_backbone: 是否仅冻结文本主干（language_model + lm_head），
                Vision Encoder 保持可训练；仅在 freeze=False 时生效
            cache_dir: 模型缓存目录，如果指定则从缓存加载
            use_state: 是否使用机器人本体信息（关节角度等）
            state_dim: 机器人状态维度
            config: 配置字典（可选），如果提供则优先使用配置字典中的值
        """
        super().__init__()
        
        # 如果提供了配置字典，优先使用配置字典中的值
        local_model_path = None
        if config is not None:
            model_name = config.get("model_name", model_name)
            image_size = config.get("image_size", image_size)
            max_seq_length = config.get("max_seq_length", max_seq_length)
            freeze = config.get("freeze_vlm", freeze)
            freeze_text_backbone = config.get("freeze_text_backbone", freeze_text_backbone)
            cache_dir = config.get("cache_dir", cache_dir)
            local_model_path = config.get("local_model_path", None)
            use_state = config.get("use_state", use_state)
            state_dim = config.get("state_dim", state_dim)
        
        self.model_name = model_name
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        self.cache_dir = cache_dir
        self.use_state = use_state
        self.state_dim = state_dim
        
        # 解析本地路径以实现完全离线加载（避免 huggingface_hub 联网）
        load_path = model_name
        if cache_dir or local_model_path:
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            if local_model_path and Path(local_model_path).exists():
                load_path = str(Path(local_model_path).resolve())
                print(f"Loading Qwen VLM from local_model_path (offline): {load_path}")
            elif cache_dir:
                resolved = _resolve_local_model_path(cache_dir, model_name)
                if resolved:
                    load_path = resolved
                    print(f"Loading Qwen VLM from local cache (offline): {load_path}")
            else:
                print(f"Loading Qwen VLM model from cache: {model_name} (cache: {cache_dir})")
                print("  Offline mode: local_files_only=True")
        else:
            print(f"Loading Qwen VLM model: {model_name}")

        try:
            processor_kwargs = {"trust_remote_code": True}
            if load_path != model_name:
                # 使用本地路径，直接加载，无需联网
                self.processor = AutoProcessor.from_pretrained(load_path, trust_remote_code=True)
            else:
                processor_kwargs = {
                    "cache_dir": cache_dir,
                    "trust_remote_code": True
                }
                if cache_dir:
                    processor_kwargs["local_files_only"] = True
                self.processor = AutoProcessor.from_pretrained(model_name, **processor_kwargs)
        except Exception as e:
            print(f"Warning: Failed to load processor, trying tokenizer: {e}")
            tokenizer_kwargs = {"trust_remote_code": True}
            if load_path != model_name:
                self.processor = AutoTokenizer.from_pretrained(load_path, trust_remote_code=True)
            else:
                tokenizer_kwargs = {"cache_dir": cache_dir, "trust_remote_code": True}
                if cache_dir:
                    tokenizer_kwargs["local_files_only"] = True
                self.processor = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
        
        # 加载模型（优先使用Qwen2VLForConditionalGeneration）
        if HAS_QWEN2VL and "Qwen2-VL" in model_name:
            try:
                if load_path != model_name:
                    # 使用本地路径，完全离线加载
                    self._use_device_map = False
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(
                        load_path,
                        torch_dtype="auto",
                        trust_remote_code=True
                    )
                else:
                    load_kwargs = {
                        "cache_dir": cache_dir,
                        "torch_dtype": "auto",
                        "trust_remote_code": True
                    }
                    if cache_dir:
                        load_kwargs["local_files_only"] = True
                    if cache_dir is None:
                        load_kwargs["device_map"] = "auto"
                        self._use_device_map = True
                    else:
                        self._use_device_map = False
                    self.model = Qwen2VLForConditionalGeneration.from_pretrained(model_name, **load_kwargs)
                print(f"✓ 成功加载模型为 Qwen2VLForConditionalGeneration")
                if self._use_device_map:
                    print(f"  使用 device_map='auto'，模型已自动分配到设备")
                else:
                    print(f"  模型将在后续手动移动到指定设备")
            except Exception as e:
                print(f"✗ Qwen2VLForConditionalGeneration 加载失败: {e}")
                print("尝试使用 AutoModelForCausalLM...")
                self._use_device_map = False
                try:
                    from transformers import AutoModelForCausalLM
                    if load_path != model_name:
                        self.model = AutoModelForCausalLM.from_pretrained(
                            load_path,
                            torch_dtype=torch.float32,
                            trust_remote_code=True
                        )
                    else:
                        fallback_kwargs = {
                            "cache_dir": cache_dir,
                            "torch_dtype": torch.float32,
                            "trust_remote_code": True
                        }
                        if cache_dir:
                            fallback_kwargs["local_files_only"] = True
                        self.model = AutoModelForCausalLM.from_pretrained(
                            model_name,
                            **fallback_kwargs
                        )
                    print(f"✓ 使用 AutoModelForCausalLM 加载")
                except Exception as e2:
                    raise RuntimeError(
                        f"无法加载模型: Qwen2VLForConditionalGeneration 失败 ({e}), "
                        f"AutoModelForCausalLM 也失败 ({e2})"
                    )
        else:
            try:
                from transformers import AutoModelForCausalLM
                if load_path != model_name:
                    self.model = AutoModelForCausalLM.from_pretrained(
                        load_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                else:
                    model_kwargs = {
                        "cache_dir": cache_dir,
                        "torch_dtype": torch.float32,
                        "trust_remote_code": True
                    }
                    if cache_dir:
                        model_kwargs["local_files_only"] = True
                    self.model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        **model_kwargs
                    )
                print(f"✓ 使用 AutoModelForCausalLM 加载")
                self._use_device_map = False
            except Exception as e:
                print(f"✗ AutoModelForCausalLM 加载失败: {e}")
                if load_path != model_name:
                    self.model = AutoModel.from_pretrained(
                        load_path,
                        torch_dtype=torch.float32,
                        trust_remote_code=True
                    )
                else:
                    automodel_kwargs = {
                        "cache_dir": cache_dir,
                        "torch_dtype": torch.float32,
                        "trust_remote_code": True
                    }
                    if cache_dir:
                        automodel_kwargs["local_files_only"] = True
                    self.model = AutoModel.from_pretrained(
                        model_name,
                        **automodel_kwargs
                    )
                print(f"⚠ 使用 AutoModel 加载，可能不支持 generate 方法")
                self._use_device_map = False
        
        self.apply_vlm_freeze_policy(freeze_vlm=freeze, freeze_text_backbone=freeze_text_backbone)
        
        # 获取隐藏层维度
        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            self.hidden_dim = self.model.config.d_model
        else:
            # 默认值
            self.hidden_dim = 1536
            print(f"Warning: Could not determine hidden_dim, using default: {self.hidden_dim}")
    
    @classmethod
    def from_config(cls, config: Dict, use_state: bool = True, state_dim: int = 7):
        """
        从配置字典创建QwenVLM实例
        
        Args:
            config: VLM配置字典，应包含以下键：
                - model_name: HuggingFace模型名称
                - image_size: 输入图像尺寸
                - max_seq_length: 最大序列长度
                - freeze_vlm: 是否冻结VLM参数
                - cache_dir: 模型缓存目录（可选）
            use_state: 是否使用机器人本体信息（关节角度等），如果配置中有use_state则优先使用配置中的值
            state_dim: 机器人状态维度，如果配置中有state_dim则优先使用配置中的值
            
        Returns:
            QwenVLM实例
        """
        # 从配置中读取值，如果配置中没有则使用默认值
        return cls(
            model_name=config.get("model_name", "Qwen/Qwen2-VL-2B-Instruct"),
            image_size=config.get("image_size", 224),
            max_seq_length=config.get("max_seq_length", 512),
            freeze=config.get("freeze_vlm", False),
            freeze_text_backbone=config.get("freeze_text_backbone", False),
            cache_dir=config.get("cache_dir", None),
            use_state=config.get("use_state", use_state),
            state_dim=config.get("state_dim", state_dim)
        )

    @staticmethod
    def _set_module_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
        if module is None:
            return
        for param in module.parameters():
            param.requires_grad = requires_grad

    def apply_vlm_freeze_policy(
        self,
        freeze_vlm: bool = False,
        freeze_text_backbone: bool = False,
    ) -> None:
        """
        应用 VLM 参数冻结策略。

        - freeze_vlm=True: 冻结整个 Qwen2-VL（含 Vision Encoder 与文本主干）
        - freeze_vlm=False 且 freeze_text_backbone=True: 仅冻结 language_model 与 lm_head，
          Vision Encoder 保持可训练（tokenizer/processor 本身无可训练参数）
        - 两者均为 False: VLM 全量可训练
        """
        if freeze_vlm:
            for param in self.model.parameters():
                param.requires_grad = False
            print("VLM: all parameters frozen")
            return

        for param in self.model.parameters():
            param.requires_grad = True

        if not freeze_text_backbone:
            print("VLM: all backbone parameters trainable")
            return

        frozen_parts = []
        backbone = getattr(self.model, "model", None)
        if backbone is not None:
            if hasattr(backbone, "language_model"):
                self._set_module_requires_grad(backbone.language_model, False)
                frozen_parts.append("language_model")
            if hasattr(backbone, "visual"):
                self._set_module_requires_grad(backbone.visual, True)
        if hasattr(self.model, "lm_head") and self.model.lm_head is not None:
            self._set_module_requires_grad(self.model.lm_head, False)
            frozen_parts.append("lm_head")

        if not frozen_parts:
            print("Warning: freeze_text_backbone=True but no language modules found; VLM left fully trainable")
        else:
            print(f"VLM: frozen {', '.join(frozen_parts)}; visual encoder trainable")

    def summarize_backbone_trainability(self) -> Dict[str, Tuple[int, int]]:
        """返回各主干子模块 (trainable_params, total_params) 统计。"""
        groups = {
            "visual": 0,
            "language_model": 0,
            "lm_head": 0,
            "other": 0,
        }
        group_totals = dict(groups)

        for name, param in self.model.named_parameters():
            trainable = int(param.numel()) if param.requires_grad else 0
            total = int(param.numel())
            if name.startswith("model.visual."):
                key = "visual"
            elif name.startswith("model.language_model."):
                key = "language_model"
            elif name.startswith("lm_head."):
                key = "lm_head"
            else:
                key = "other"
            groups[key] += trainable
            group_totals[key] += total

        return {k: (groups[k], group_totals[k]) for k in groups}
        
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        texts: Optional[List[str]] = None,
        states: Optional[torch.Tensor] = None,
        return_dict: bool = True,
        output_hidden_states: bool = True,
        output_attentions: bool = False,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播（支持机器人本体信息）
        
        Args:
            images: 图像张量 [B, C, H, W]，值域[0, 1]（可选，如果提供了kwargs则不需要）
            texts: 文本列表，可选
            states: 机器人本体信息（关节角度等）[B, state_dim] 或 [B, 1, state_dim]，可选
            return_dict: 是否返回字典格式
            output_hidden_states: 是否输出所有层的hidden states
            output_attentions: 是否输出注意力权重
            **kwargs: 可以直接传入build_qwenvl_inputs的输出（用于Qwen-GR00T架构）
            
        Returns:
            包含视觉-语言特征的字典，包括：
                - hidden_states: 所有层的hidden states（如果output_hidden_states=True）
                - last_hidden_state: 最后一层的hidden state [B, seq_len, hidden_dim]
                - features: 全局特征 [B, hidden_dim]（向后兼容）
        """
        # 如果提供了kwargs（来自build_qwenvl_inputs），直接使用
        if kwargs and any(k in kwargs for k in ['input_ids', 'pixel_values', 'images']):
            inputs = kwargs
        else:
            # 否则使用传统方式处理
            if images is None:
                raise ValueError("Either images or kwargs must be provided")
            batch_size = images.shape[0]
            device = next(self.model.parameters()).device
            
            # 准备输入
            if texts is None:
                texts = [""] * batch_size
            
            # 将机器人本体信息添加到文本指令中
            if self.use_state and states is not None:
                # 处理states维度
                if states.dim() == 3:
                    # [B, 1, state_dim] -> [B, state_dim]
                    states_2d = states[:, 0, :]
                elif states.dim() == 2:
                    # [B, state_dim]
                    states_2d = states
                else:
                    raise ValueError(f"Unexpected states shape: {states.shape}")
                
                # 将状态信息格式化为文本并添加到指令中
                texts_with_state = []
                for i, text in enumerate(texts):
                    state_values = states_2d[i].cpu().numpy() if isinstance(states_2d[i], torch.Tensor) else states_2d[i]
                    # 格式化状态信息为文本
                    state_str = ", ".join([f"{val:.3f}" for val in state_values])
                    if text:
                        # 将状态信息添加到指令中
                        enhanced_text = f"{text}\n[Robot State: {state_str}]"
                    else:
                        enhanced_text = f"[Robot State: {state_str}]"
                    texts_with_state.append(enhanced_text)
                texts = texts_with_state
            
            # 处理输入：对于Qwen2-VL，应该使用build_qwenvl_inputs方法
            # 将图像转换为PIL格式
            if images.max() <= 1.0:
                images_uint8 = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
            else:
                images_uint8 = images.byte().permute(0, 2, 3, 1).cpu().numpy()
            
            pil_images = [Image.fromarray(img) for img in images_uint8]
            
            # 使用build_qwenvl_inputs方法构建输入（这是Qwen2-VL推荐的方式）
            try:
                inputs = self.build_qwenvl_inputs(
                    images=pil_images,
                    instructions=texts,
                    states=states
                )
            except Exception as e:
                # 如果build_qwenvl_inputs失败，尝试直接使用processor
                print(f"Warning: build_qwenvl_inputs failed, trying processor directly: {e}")
                try:
                    # 构建messages格式
                    messages = []
                    for img, text in zip(pil_images, texts):
                        message = {
                            "role": "user",
                            "content": []
                        }
                        if img is not None:
                            message["content"].append({"type": "image", "image": img})
                        if text:
                            message["content"].append({"type": "text", "text": text})
                        messages.append(message)
                    
                    # 使用processor处理
                    if hasattr(self.processor, 'apply_chat_template'):
                        # 对每个message单独处理，确保batch维度正确
                        texts = []
                        for msg in messages:
                            text = self.processor.apply_chat_template(
                                [msg],  # 单个message作为列表
                                tokenize=False,
                                add_generation_prompt=False
                            )
                            texts.append(text)
                        
                        if HAS_QWEN_VL_UTILS:
                            image_inputs, video_inputs = process_vision_info(messages)
                        else:
                            # 备用方法：直接从messages中提取图像
                            image_inputs = []
                            for msg in messages:
                                for item in msg.get("content", []):
                                    if item.get("type") == "image":
                                        image_inputs.append(item["image"])
                            video_inputs = []
                        
                        inputs = self.processor(
                            text=texts,  # 使用列表而不是单个字符串
                            images=image_inputs if image_inputs else None,
                            videos=video_inputs if video_inputs else None,
                            padding=True,
                            return_tensors="pt"
                        )
                    else:
                        # 回退到简单方式
                        inputs = self.processor(
                            text=texts,
                            images=pil_images,
                            return_tensors="pt",
                            padding=True
                        )
                    
                    # 将输入移到模型设备
                    device = next(self.model.parameters()).device
                    inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                             for k, v in inputs.items()}
                except Exception as e2:
                    raise RuntimeError(f"Failed to prepare inputs: {e2}")
        
        # 获取模型输出
        # 在推理时，确保模型处于eval模式以提高输出一致性
        # 如果模型在训练模式，dropout等层会引入随机性
        was_training = self.model.training
        if was_training:
            self.model.eval()
        
        try:
            with torch.set_grad_enabled(self.training):
                outputs = self.model(
                    **inputs,
                    output_hidden_states=output_hidden_states,
                    output_attentions=output_attentions,
                    return_dict=True
                )
        except Exception as e:
            # 如果模型调用失败，打印详细错误信息
            print(f"Warning: Model forward failed: {e}")
            print(f"  Input keys: {list(inputs.keys())}")
            if 'input_ids' in inputs:
                print(f"  input_ids shape: {inputs['input_ids'].shape}")
            if 'pixel_values' in inputs:
                print(f"  pixel_values shape: {inputs['pixel_values'].shape}")
            # Qwen2VLForConditionalGeneration没有vision_model属性，直接抛出错误
            raise RuntimeError(
                f"Model forward failed. This might be due to incorrect input format. "
                f"Please ensure images and texts are properly processed using build_qwenvl_inputs. "
                f"Original error: {e}"
            )
        finally:
            # 恢复原始训练状态
            if was_training:
                self.model.train()
        
        # 提取hidden states
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 所有层的hidden states
            all_hidden_states = outputs.hidden_states  # List of [B, seq_len, hidden_dim]
            last_hidden_state = all_hidden_states[-1]  # [B, seq_len, hidden_dim]
        elif hasattr(outputs, 'last_hidden_state'):
            last_hidden_state = outputs.last_hidden_state
            all_hidden_states = [last_hidden_state] if output_hidden_states else None
        else:
            # 如果都没有，使用模型的输出
            last_hidden_state = outputs[0] if isinstance(outputs, tuple) else outputs
            all_hidden_states = [last_hidden_state] if output_hidden_states else None
        
        # 确保last_hidden_state是3D张量
        if len(last_hidden_state.shape) == 2:
            last_hidden_state = last_hidden_state.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 对序列维度进行平均池化，得到全局特征（向后兼容）
        visual_language_features = last_hidden_state.mean(dim=1)  # [B, hidden_dim]
        
        if return_dict:
            result = {
                "last_hidden_state": last_hidden_state,  # [B, seq_len, hidden_dim]
                "features": visual_language_features,  # [B, hidden_dim] (向后兼容)
                "full_features": last_hidden_state,  # [B, seq_len, hidden_dim] (向后兼容)
            }
            if output_hidden_states and all_hidden_states is not None:
                result["hidden_states"] = all_hidden_states  # List of [B, seq_len, hidden_dim]
            if output_attentions and hasattr(outputs, 'attentions'):
                result["attentions"] = outputs.attentions
            return result
        else:
            return visual_language_features
    
    def get_hidden_dim(self) -> int:
        """返回隐藏层维度"""
        return self.hidden_dim
    
    def build_qwenvl_inputs(
        self,
        images: Union[List[Image.Image], List[List[Image.Image]]],
        instructions: List[str],
        states: Optional[Union[torch.Tensor, np.ndarray, List]] = None
    ) -> Dict:
        """
        构建Qwen VL输入格式（参考Qwen-GR00T架构，支持机器人本体信息）
        
        Args:
            images: 图像列表，可以是：
                - List[Image.Image]: 单相机图像列表 [B]
                - List[List[Image.Image]]: 多相机图像列表 [B, num_cameras]
            instructions: 指令文本列表 [B]
            states: 机器人本体信息（关节角度等），可以是：
                - torch.Tensor: [B, state_dim] 或 [B, 1, state_dim]
                - np.ndarray: [B, state_dim] 或 [B, 1, state_dim]
                - List: [B] 个状态数组
            
        Returns:
            包含处理后的输入的字典，可直接传递给Qwen VL模型
        """
        batch_size = len(instructions)
        
        # 处理图像格式
        if images and len(images) > 0:
            batch_images = images
            # if isinstance(images[0], list):
            #     # 多相机模式：取第一个相机（或可以扩展为融合多个相机）
            #     batch_images = [img_list[0] if img_list else None for img_list in images]
            # else:
            #     # 单相机模式
            #     batch_images = images
        else:
            raise ValueError(f"Empty images list or None")
        
        # 处理机器人本体信息（如果提供）
        if self.use_state and states is not None:
            # 转换为numpy数组
            if isinstance(states, torch.Tensor):
                states_np = states.detach().cpu().numpy()
            elif isinstance(states, np.ndarray):
                states_np = states
            elif isinstance(states, list):
                states_np = np.array(states)
            else:
                raise TypeError(f"Unsupported states type: {type(states)}")
            
            # 处理维度
            if states_np.ndim == 3:
                # [B, 1, state_dim] -> [B, state_dim]
                states_np = states_np[:, 0, :]
            elif states_np.ndim == 2:
                # [B, state_dim]
                pass
            else:
                raise ValueError(f"Unexpected states shape: {states_np.shape}")
            
            # 将状态信息添加到指令中，使用标准化格式以提高一致性
            enhanced_instructions = []
            for i, instruction in enumerate(instructions):
                state_values = states_np[i]
                # 格式化状态信息为文本，使用固定格式以提高一致性
                # 使用固定的小数位数（3位）和格式，确保每次输出一致
                # 对状态值进行排序以确保格式一致性（如果需要）
                state_str = ", ".join([f"{val:.3f}" for val in state_values])
                # 使用标准化的prompt格式，明确指示这是机器人状态信息
                # 格式：指令文本 + 标准化的状态信息标记（使用固定格式）
                instruction_clean = instruction.strip() if instruction else ""
                if instruction_clean:
                    # 如果指令不为空，使用标准格式（确保换行符一致）
                    enhanced_text = f"{instruction_clean}\n[Robot State: {state_str}]"
                else:
                    # 如果指令为空，只提供状态信息
                    enhanced_text = f"[Robot State: {state_str}]"
                enhanced_instructions.append(enhanced_text)
            instructions = enhanced_instructions
        
        # 构建messages格式（Qwen2-VL标准格式）
        # 使用标准化的格式以提高一致性
        messages = []
        for imgs, instruction in zip(batch_images, instructions):
            message = {
                "role": "user",
                "content": []
            }
            
            # 添加图像（如果存在）
            if imgs is not None:
                #message["content"].append({
                #    "type": "image",
                #    "image": img
                #})
                message["content"] = [{"type":"image", "image":img} for img in imgs]
                
            # 添加文本指令
            if instruction:
                message["content"].append({
                    "type": "text",
                    "text": instruction
                })
            
            messages.append(message)
        
        # 使用processor处理
        try:
            # 步骤1: 使用apply_chat_template处理文本
            # 注意：apply_chat_template对每个message单独处理，返回列表
            # 设置add_generation_prompt=False，因为我们只是提取hidden states，不生成文本
            # 这样可以提高输出一致性
            texts = []
            for msg in messages:
                text = self.processor.apply_chat_template(
                    [msg],  # 单个message作为列表
                    tokenize=False,
                    add_generation_prompt=True
                )
                texts.append(text)
            
            # 步骤2: 处理图像信息
            if HAS_QWEN_VL_UTILS:
                image_inputs, video_inputs = process_vision_info(messages)
            else:
                # 备用方法：直接从messages中提取图像
                image_inputs = []
                for msg in messages:
                    for item in msg.get("content", []):
                        if item.get("type") == "image":
                            image_inputs.append(item["image"])
                video_inputs = []
            
            # 步骤3: 使用processor处理输入
            inputs = self.processor(
                text=texts,
                images=image_inputs if image_inputs else None,
                videos=video_inputs if video_inputs else None,
                padding=True,
                return_tensors="pt"
            )
            
            # 将输入移到模型设备
            device = next(self.model.parameters()).device
            inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                     for k, v in inputs.items()}
            
            return inputs
            
        except Exception as e:
            # 回退到简单格式
            print(f"Warning: build_qwenvl_inputs failed, using fallback: {e}")
            try:
                inputs = self.processor(
                    text=instructions,
                    images=batch_images,
                    return_tensors="pt",
                    padding=True
                )
                device = next(self.model.parameters()).device
                inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
                return inputs
            except Exception as e2:
                raise RuntimeError(f"Failed to build Qwen VL inputs: {e2}")


