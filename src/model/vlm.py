"""
Qwen VLM模型封装
用于视觉-语言理解
"""

import torch
import torch.nn as nn
from transformers import AutoModel, AutoProcessor, AutoTokenizer
from typing import Optional, Tuple, Dict
from PIL import Image


class QwenVLM(nn.Module):
    """
    基于Qwen的视觉语言模型
    用于处理图像和文本输入，输出视觉-语言特征
    """
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen-VL-Chat",
        image_size: int = 224,
        max_seq_length: int = 512,
        freeze: bool = False
    ):
        """
        初始化Qwen VLM模型
        
        Args:
            model_name: HuggingFace模型名称
            image_size: 输入图像尺寸
            max_seq_length: 最大序列长度
            freeze: 是否冻结模型参数
        """
        super().__init__()
        
        self.model_name = model_name
        self.image_size = image_size
        self.max_seq_length = max_seq_length
        
        # 加载Qwen VLM模型和处理器
        print(f"Loading Qwen VLM model: {model_name}")
        try:
            # 尝试加载processor
            self.processor = AutoProcessor.from_pretrained(model_name)
        except:
            # 如果没有processor，使用tokenizer
            self.processor = AutoTokenizer.from_pretrained(model_name)
        
        # 加载模型
        self.model = AutoModel.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True
        )
        
        # 冻结参数（如果需要）
        if freeze:
            for param in self.model.parameters():
                param.requires_grad = False
            print("VLM parameters frozen")
        
        # 获取隐藏层维度
        if hasattr(self.model.config, 'hidden_size'):
            self.hidden_dim = self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            self.hidden_dim = self.model.config.d_model
        else:
            # 默认值
            self.hidden_dim = 768
            print(f"Warning: Could not determine hidden_dim, using default: {self.hidden_dim}")
        
    def forward(
        self,
        images: torch.Tensor,
        texts: Optional[list] = None,
        return_dict: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            images: 图像张量 [B, C, H, W]，值域[0, 1]
            texts: 文本列表，可选
            return_dict: 是否返回字典格式
            
        Returns:
            包含视觉-语言特征的字典
        """
        batch_size = images.shape[0]
        device = next(self.model.parameters()).device
        
        # 准备输入
        if texts is None:
            texts = [""] * batch_size
        
        # 处理输入
        # 将图像转换为PIL格式（如果需要）
        if images.max() <= 1.0:
            images_uint8 = (images * 255).byte().permute(0, 2, 3, 1).cpu().numpy()
        else:
            images_uint8 = images.byte().permute(0, 2, 3, 1).cpu().numpy()
        
        pil_images = [Image.fromarray(img) for img in images_uint8]
        
        try:
            # 尝试使用processor处理
            if hasattr(self.processor, 'image_processor'):
                inputs = self.processor(
                    text=texts,
                    images=pil_images,
                    return_tensors="pt",
                    padding=True
                )
            else:
                # 使用tokenizer处理文本，图像单独处理
                text_inputs = self.processor(
                    texts,
                    return_tensors="pt",
                    padding=True
                )
                # 图像需要单独处理
                inputs = {**text_inputs}
                inputs['pixel_values'] = images.to(device)
        except Exception as e:
            # 如果processor失败，手动构建输入
            print(f"Warning: Processor failed, using manual input preparation: {e}")
            text_inputs = self.processor(
                texts,
                return_tensors="pt",
                padding=True
            )
            inputs = {**text_inputs}
            inputs['pixel_values'] = images.to(device)
        
        # 将输入移到模型设备
        inputs = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        
        # 获取模型输出
        with torch.set_grad_enabled(self.training):
            try:
                outputs = self.model(**inputs, output_hidden_states=True)
            except Exception as e:
                # 如果模型调用失败，尝试不同的输入格式
                print(f"Warning: Model forward failed, trying alternative: {e}")
                # 只使用图像特征
                if 'pixel_values' in inputs:
                    outputs = self.model.vision_model(pixel_values=inputs['pixel_values'])
                else:
                    raise e
        
        # 提取视觉-语言特征
        if hasattr(outputs, 'hidden_states') and outputs.hidden_states is not None:
            # 使用最后一层的隐藏状态
            features = outputs.hidden_states[-1]  # [B, seq_len, hidden_dim]
        elif hasattr(outputs, 'last_hidden_state'):
            features = outputs.last_hidden_state
        elif hasattr(outputs, 'pooler_output'):
            features = outputs.pooler_output.unsqueeze(1)  # [B, 1, hidden_dim]
        else:
            # 如果都没有，使用模型的输出
            features = outputs[0] if isinstance(outputs, tuple) else outputs
        
        # 确保features是3D张量
        if len(features.shape) == 2:
            features = features.unsqueeze(1)  # [B, 1, hidden_dim]
        
        # 对序列维度进行平均池化，得到全局特征
        visual_language_features = features.mean(dim=1)  # [B, hidden_dim]
        
        if return_dict:
            return {
                "features": visual_language_features,
                "full_features": features
            }
        else:
            return visual_language_features
    
    def get_hidden_dim(self) -> int:
        """返回隐藏层维度"""
        return self.hidden_dim


