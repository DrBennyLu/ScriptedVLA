"""
Qwen VLM机器人能力测评脚本
测试物体识别、空间感知、因果推理等能力
"""
import os
# 在代码最顶部添加（必须在import torch之前）
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import torch
import argparse
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from typing import Dict, List, Tuple, Optional
import json
from datetime import datetime

import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from ScriptedVLA.model.vlm import QwenVLM
from ScriptedVLA.utils import load_config, get_model_config

# 尝试导入Qwen2VL专用工具
try:
    from qwen_vl_utils import process_vision_info
    HAS_QWEN_VL_UTILS = True
except ImportError:
    HAS_QWEN_VL_UTILS = False
    # 不在这里打印警告，因为可能不是所有情况都需要


class VLMCapabilityEvaluator:
    """VLM能力评估器"""
    
    def __init__(
        self,
        model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
        device: str = None,
        test_image_path: str = "test_imgs/test_img_1.jpg",
        cache_dir: str = "./cache/models"
    ):
        """
        初始化评估器
        
        Args:
            model_name: 模型名称
            device: 设备（cuda/cpu），None表示自动选择
            test_image_path: 测试图像路径，默认为 test_imgs/test_img_1.jpg
            cache_dir: 模型缓存目录，默认为 ./cache/models
        """
        self.model_name = model_name
        self.test_image_path = Path(test_image_path)
        self.cache_dir = cache_dir
        
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)


        print(f"初始化VLM模型: {model_name}")
        print(f"使用设备: {self.device}")
        if cache_dir:
            print(f"使用缓存目录: {cache_dir}")
        
        # 检查测试图像是否存在
        if not self.test_image_path.exists():
            print(f"警告: 测试图像文件不存在: {self.test_image_path}")
            print(f"请确保 {self.test_image_path} 文件存在，或使用 --test-image 参数指定其他图像")
            # 创建目录（如果不存在）
            self.test_image_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 加载模型（从缓存目录）
        self.vlm = QwenVLM(
            model_name=model_name,
            image_size=448,  # 使用更大的图像尺寸以获得更好的效果
            max_seq_length=512,
            freeze=False,
            cache_dir=cache_dir
        )
        
        # 处理设备分配
        # 如果模型使用了device_map="auto"，需要检查模型实际所在的设备
        if hasattr(self.vlm, '_use_device_map') and self.vlm._use_device_map:
            # 获取模型实际所在的设备
            model_device = next(self.vlm.model.parameters()).device
            print(f"  模型实际所在设备: {model_device}")
            # 如果用户指定的设备与模型设备不一致，需要移动模型
            if str(model_device) != str(self.device):
                if self.device == "cpu":
                    # 用户指定使用CPU，强制移动模型到CPU
                    print(f"  将模型从 {model_device} 移动到 {self.device}")
                    self.vlm.model = self.vlm.model.to(self.device)
                    self._use_device_map = False  # 不再使用自动设备映射
                else:
                    # 用户指定使用CUDA，但模型在CPU，移动到CUDA
                    print(f"  将模型从 {model_device} 移动到 {self.device}")
                    self.vlm.model = self.vlm.model.to(self.device)
                    self._use_device_map = False
        else:
            # 手动指定设备
            self.vlm = self.vlm.to(self.device)
            self._use_device_map = False
        
        self.vlm.eval()
        
        print("模型加载完成！\n")
    
    def generate_text_response(
        self,
        image: Image.Image,
        prompt: str,
        max_new_tokens: int = 100
    ) -> str:
        """
        生成文本响应
        
        Args:
            image: 输入图像
            prompt: 提示文本
            max_new_tokens: 最大生成token数
            
        Returns:
            模型响应文本
        """
        try:
            # 直接使用transformers的processor和model
            processor = self.vlm.processor
            model = self.vlm.model
            
            # 准备输入
            # Qwen2-VL使用标准的消息格式
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": image},
                        {"type": "text", "text": prompt}
                    ]
                }
            ]
            
            # 使用processor准备输入（按照官方示例）
            try:
                # 步骤1: 使用apply_chat_template处理文本（按照官方示例）
                text = processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
                
                # 步骤2: 处理图像信息（使用qwen_vl_utils.process_vision_info）
                if HAS_QWEN_VL_UTILS:
                    image_inputs, video_inputs = process_vision_info(messages)
                else:
                    # 备用方法：直接从messages中提取图像
                    image_inputs = []
                    video_inputs = []
                    for msg in messages:
                        if "content" in msg:
                            for item in msg["content"]:
                                if item.get("type") == "image":
                                    image_inputs.append(item["image"])
                
                # 步骤3: 使用processor处理输入（按照官方示例）
                inputs = processor(
                    text=[text],
                    images=image_inputs if image_inputs else None,
                    videos=video_inputs if video_inputs else None,
                    padding=True,
                    return_tensors="pt"
                )
            except Exception as e:
                print(f"警告: 使用官方方法失败，尝试备用方法: {e}")
                # 回退到简单格式
                try:
                    inputs = processor(
                        text=prompt,
                        images=image,
                        return_tensors="pt",
                        padding=True
                    )
                except:
                    # 如果processor不支持，使用更基础的方式
                    inputs = processor(
                        prompt,
                        images=image,
                        return_tensors="pt"
                    )
            
            # 将输入移到设备（按照官方示例：inputs = inputs.to("cuda")）
            # 确定模型所在的设备
            if hasattr(self.vlm, 'model'):
                model_device = next(self.vlm.model.parameters()).device
            else:
                model_device = torch.device(self.device)
            
            # 将输入移到模型所在的设备
            if isinstance(inputs, dict):
                inputs = {k: v.to(model_device) if isinstance(v, torch.Tensor) else v 
                         for k, v in inputs.items()}
            else:
                inputs = inputs.to(model_device)
            
            # 生成响应（按照官方示例）
            with torch.no_grad():
                # 检查模型是否有generate方法
                if not hasattr(model, 'generate'):
                    model_type = type(model).__name__
                    raise AttributeError(
                        f"模型 {model_type} 没有 generate 方法。\n"
                        f"请确保使用 Qwen2VLForConditionalGeneration 或 AutoModelForCausalLM 加载模型。\n"
                        f"当前模型类型: {model_type}"
                    )
                
                # 生成（按照官方示例）
                generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)
                
                # 解码 - 提取生成的部分（按照官方示例）
                generated_ids_trimmed = [
                    out_ids[len(in_ids):] 
                    for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                
                generated_text = processor.batch_decode(
                    generated_ids_trimmed,
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False  # 按照官方示例使用False
                )[0]
            
            # 返回生成的文本（官方示例直接返回解码后的文本）
            return generated_text.strip()
            
        except Exception as e:
            print(f"生成响应时出错: {e}")
            import traceback
            traceback.print_exc()
            # 如果生成失败，尝试使用特征提取方式
            try:
                # 使用VLM的特征提取功能作为备选
                image_array = np.array(image.resize((448, 448)))
                image_tensor = torch.from_numpy(image_array).permute(2, 0, 1).float() / 255.0
                image_tensor = image_tensor.unsqueeze(0).to(self.device)
                
                outputs = self.vlm(image_tensor, texts=[prompt], return_dict=True)
                return f"[特征提取成功，但无法生成文本。错误: {str(e)}]"
            except:
                return f"[生成失败: {str(e)}]"
    
    def test_object_recognition(self) -> Dict:
        """
        测试1: 物体识别能力
        测试模型能否识别图像中的物体
        """
        print("=" * 60)
        print("测试1: 物体识别能力")
        print("=" * 60)
        
        results = []
        
        # 测试1.1: 简单物体识别
        print("\n1.1 简单物体识别测试")
        test_image = self._load_test_image()
        prompt = "请描述图像中有什么物体，并列出它们的名称。"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "简单物体识别",
            "prompt": prompt,
            "response": response,
            "expected_keywords": ["网球", "饭盒", "机器人", "卡皮巴拉"]
        })
        
        # 测试1.2: 颜色识别
        print("\n1.2 颜色识别测试")
        test_image = self._load_test_image()
        prompt = "图像中有哪些颜色？请详细描述。"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "颜色识别",
            "prompt": prompt,
            "response": response,
            "expected_keywords": ["红色", "绿色", "棕色", "银色", "red", "green", "brown", "silver"]
        })
        
        # 测试1.3: 数量统计
        print("\n1.3 数量统计测试")
        test_image = self._load_test_image()
        prompt = "图像中有多少个物体？请数一数。"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "数量统计",
            "prompt": prompt,
            "response": response,
            "expected_count": 4  # 图像中有4个物体
        })
        
        return {
            "category": "物体识别",
            "results": results,
            "summary": f"完成 {len(results)} 个子测试"
        }
    
    def test_spatial_perception(self) -> Dict:
        """
        测试2: 空间感知能力
        测试模型对空间关系的理解
        """
        print("\n" + "=" * 60)
        print("测试2: 空间感知能力")
        print("=" * 60)
        
        results = []
        
        # 测试2.1: 位置关系
        print("\n2.1 位置关系测试")
        test_image = self._load_test_image()
        prompt = "描述图像中物体的空间位置关系，比如哪个在左边、右边、上面、下面。"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "位置关系",
            "prompt": prompt,
            "response": response,
            "expected_keywords": ["左边", "右边", "上方", "下方","前方", "后方", "left", "right", "top", "bottom", "front", "back"]
        })
        
        # 测试2.2: 距离判断
        print("\n2.2 距离判断测试")
        test_image = self._load_test_image()
        prompt = "图像中哪些物体距离较近？哪些距离较远？"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "距离判断",
            "prompt": prompt,
            "response": response
        })
        
        # 测试2.3: 方向判断
        print("\n2.3 方向判断测试")
        test_image = self._load_test_image()
        prompt = "描述图像中物体的排列方向，比如水平排列、垂直排列等。"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "方向判断",
            "prompt": prompt,
            "response": response
        })
        
        return {
            "category": "空间感知",
            "results": results,
            "summary": f"完成 {len(results)} 个子测试"
        }
    
    def test_causal_reasoning(self) -> Dict:
        """
        测试3: 因果推理能力
        测试模型根据图文进行因果推理的能力
        """
        print("\n" + "=" * 60)
        print("测试3: 因果推理能力")
        print("=" * 60)
        
        results = []
        
        # 测试3.1: 动作-结果推理
        print("\n3.1 动作-结果推理测试")
        test_image = self._load_test_image()
        prompt = "根据图像内容，如果绿色物体被移动到银色物体外面，会发生什么？请推理可能的结果。"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "动作-结果推理",
            "prompt": prompt,
            "response": response
        })
        
        # 测试3.2: 场景理解
        print("\n3.2 场景理解测试")
        test_image = self._load_test_image("test_img_2.jpg")
        prompt = "分析这个场景，描述可能发生的前因后果。"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "场景理解",
            "prompt": prompt,
            "response": response
        })
        
        # 测试3.3: 逻辑推理
        print("\n3.3 逻辑推理测试")
        test_image = self._load_test_image("test_img_2.jpg")
        prompt = "观察图像中的物体排列，如果按照某种规律，下一个应该是什么？请说明推理过程。"
        
        response = self.generate_text_response(test_image, prompt)
        print(f"提示: {prompt}")
        print(f"响应: {response}")
        
        results.append({
            "test": "逻辑推理",
            "prompt": prompt,
            "response": response
        })
        
        return {
            "category": "因果推理",
            "results": results,
            "summary": f"完成 {len(results)} 个子测试"
        }
    
    def _load_test_image(self, image_path: Optional[str] = None) -> Image.Image:
        """
        加载测试图像
        
        Args:
            image_path: 图像路径（可选），如果为None则使用默认路径（test_img_1.jpg）
                       如果提供相对路径（如 "test_img_2.jpg"），则相对于test_imgs目录
                       如果提供绝对路径，则直接使用该路径
        
        Returns:
            测试图像（PIL Image）
        """
        # 如果指定了路径，使用指定路径；否则使用默认路径
        if image_path is None:
            target_path = self.test_image_path
        else:
            # 如果提供的是相对路径，相对于test_image_path的父目录
            if Path(image_path).is_absolute():
                target_path = Path(image_path)
            else:
                target_path = self.test_image_path.parent / image_path
        
        if not target_path.exists():
            raise FileNotFoundError(
                f"测试图像文件不存在: {target_path}\n"
                f"请确保文件存在，或使用 --test-image 参数指定其他图像路径"
            )
        
        try:
            image = Image.open(target_path)
            # 转换为RGB格式（如果图像是RGBA或其他格式）
            if image.mode != 'RGB':
                image = image.convert('RGB')
            print(f"已加载测试图像: {target_path}")
            return image
        except Exception as e:
            raise Exception(f"加载测试图像失败: {e}")
    
    def _create_test_image_objects(self) -> Image.Image:
        """创建物体识别测试图像"""
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制不同形状的物体
        # 圆形
        draw.ellipse([50, 50, 150, 150], fill='red', outline='black', width=2)
        # 方形
        draw.rectangle([200, 50, 300, 150], fill='blue', outline='black', width=2)
        # 三角形
        draw.polygon([(350, 150), (400, 50), (450, 150)], fill='green', outline='black', width=2)
        # 矩形
        draw.rectangle([50, 200, 200, 300], fill='yellow', outline='black', width=2)
        
        return img
    
    def _create_test_image_colors(self) -> Image.Image:
        """创建颜色识别测试图像"""
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制不同颜色的区域
        draw.rectangle([0, 0, 150, 448], fill='red')
        draw.rectangle([150, 0, 300, 448], fill='green')
        draw.rectangle([300, 0, 448, 448], fill='blue')
        
        return img
    
    def _create_test_image_counting(self) -> Image.Image:
        """创建数量统计测试图像"""
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        
        # 绘制5个圆形
        positions = [(100, 100), (200, 100), (300, 100), (150, 200), (250, 200)]
        for pos in positions:
            draw.ellipse([pos[0]-30, pos[1]-30, pos[0]+30, pos[1]+30], 
                        fill='blue', outline='black', width=2)
        
        return img
    
    def _create_test_image_spatial(self) -> Image.Image:
        """创建空间关系测试图像"""
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        
        # 左边物体
        draw.ellipse([50, 150, 150, 250], fill='red', outline='black', width=2)
        # 右边物体
        draw.rectangle([300, 150, 400, 250], fill='blue', outline='black', width=2)
        # 上方物体
        draw.polygon([(200, 50), (250, 100), (150, 100)], fill='green', outline='black', width=2)
        # 下方物体
        draw.rectangle([175, 300, 275, 400], fill='yellow', outline='black', width=2)
        
        return img
    
    def _create_test_image_distance(self) -> Image.Image:
        """创建距离判断测试图像"""
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        
        # 近距离物体（两个圆形靠近）
        draw.ellipse([100, 100, 150, 150], fill='red', outline='black', width=2)
        draw.ellipse([160, 100, 210, 150], fill='red', outline='black', width=2)
        
        # 远距离物体（两个圆形远离）
        draw.ellipse([50, 300, 100, 350], fill='blue', outline='black', width=2)
        draw.ellipse([350, 300, 400, 350], fill='blue', outline='black', width=2)
        
        return img
    
    def _create_test_image_direction(self) -> Image.Image:
        """创建方向判断测试图像"""
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        
        # 水平排列
        for i in range(5):
            x = 50 + i * 80
            draw.ellipse([x, 150, x+40, 190], fill='red', outline='black', width=2)
        
        # 垂直排列
        for i in range(4):
            y = 250 + i * 50
            draw.rectangle([350, y, 390, y+40], fill='blue', outline='black', width=2)
        
        return img
    
    def _create_test_image_causal(self) -> Image.Image:
        """创建因果推理测试图像"""
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        
        # 红色物体
        draw.ellipse([100, 200, 180, 280], fill='red', outline='black', width=3)
        # 蓝色物体
        draw.rectangle([300, 200, 380, 280], fill='blue', outline='black', width=3)
        # 箭头表示移动方向
        draw.line([180, 240, 300, 240], fill='black', width=3)
        draw.polygon([(300, 240), (285, 230), (285, 250)], fill='black')
        
        return img
    
    def _create_test_image_scene(self) -> Image.Image:
        """创建场景理解测试图像"""
        img = Image.new('RGB', (448, 448), color='lightgray')
        draw = ImageDraw.Draw(img)
        
        # 创建一个简单的场景：桌子上的物体
        # 桌子
        draw.rectangle([50, 300, 400, 350], fill='brown', outline='black', width=2)
        # 桌子上的物体
        draw.ellipse([150, 250, 200, 300], fill='red', outline='black', width=2)  # 苹果
        draw.rectangle([250, 250, 300, 300], fill='yellow', outline='black', width=2)  # 盒子
        
        return img
    
    def _create_test_image_logic(self) -> Image.Image:
        """创建逻辑推理测试图像"""
        img = Image.new('RGB', (448, 448), color='white')
        draw = ImageDraw.Draw(img)
        
        # 创建有规律的排列：圆形-方形-圆形-方形-?
        patterns = [
            (100, 200, 'circle', 'red'),
            (180, 200, 'square', 'blue'),
            (260, 200, 'circle', 'red'),
            (340, 200, 'square', 'blue'),
        ]
        
        for x, y, shape, color in patterns:
            if shape == 'circle':
                draw.ellipse([x-30, y-30, x+30, y+30], fill=color, outline='black', width=2)
            else:
                draw.rectangle([x-30, y-30, x+30, y+30], fill=color, outline='black', width=2)
        
        # 下一个位置（问号）
        draw.text((420, 185), "?", fill='black')
        
        return img
    
    def run_all_tests(self, save_results: bool = True) -> Dict:
        """
        运行所有测试
        
        Args:
            save_results: 是否保存结果到文件
            
        Returns:
            所有测试结果
        """
        print("\n" + "=" * 60)
        print("开始VLM能力测评")
        print("=" * 60)
        print(f"模型: {self.model_name}")
        print(f"设备: {self.device}")
        print(f"时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 60)
        
        all_results = {
            "model_name": self.model_name,
            "device": str(self.device),
            "timestamp": datetime.now().isoformat(),
            "tests": []
        }
        
        # 运行所有测试
        try:
            test1 = self.test_object_recognition()
            all_results["tests"].append(test1)
        except Exception as e:
            print(f"\n物体识别测试失败: {e}")
            all_results["tests"].append({
                "category": "物体识别",
                "error": str(e)
            })
        
        try:
            test2 = self.test_spatial_perception()
            all_results["tests"].append(test2)
        except Exception as e:
            print(f"\n空间感知测试失败: {e}")
            all_results["tests"].append({
                "category": "空间感知",
                "error": str(e)
            })
        
        try:
            test3 = self.test_causal_reasoning()
            all_results["tests"].append(test3)
        except Exception as e:
            print(f"\n因果推理测试失败: {e}")
            all_results["tests"].append({
                "category": "因果推理",
                "error": str(e)
            })
        
        # 保存结果
        if save_results:
            output_file = f"vlm_evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(all_results, f, indent=2, ensure_ascii=False)
            print(f"\n结果已保存到: {output_file}")
        
        # 打印总结
        print("\n" + "=" * 60)
        print("测评完成！")
        print("=" * 60)
        print(f"完成测试类别: {len(all_results['tests'])}")
        for test in all_results["tests"]:
            if "error" not in test:
                print(f"  - {test['category']}: {test.get('summary', 'N/A')}")
            else:
                print(f"  - {test['category']}: 测试失败")
        
        return all_results


def main():
    parser = argparse.ArgumentParser(description="Evaluate VLM Capabilities")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model name (default: Qwen/Qwen2-VL-2B-Instruct)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to use (default: auto)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file (optional)"
    )
    parser.add_argument(
        "--test-image",
        type=str,
        default="test_imgs/test_img_1.jpg",
        help="Path to test image file (default: test_imgs/test_img_1.jpg)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/models",
        help="Cache directory for model storage (default: ./cache/models)"
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results to file"
    )
    
    args = parser.parse_args()
    
    # 如果提供了配置文件，尝试从中读取模型名称
    model_name = args.model
    if args.config and Path(args.config).exists():
        try:
            config = load_config(args.config)
            model_config = get_model_config(config)
            vlm_config = model_config.get("vlm", {})
            if "model_name" in vlm_config:
                model_name = vlm_config["model_name"]
                print(f"从配置文件读取模型名称: {model_name}")
        except Exception as e:
            print(f"读取配置文件失败，使用命令行参数: {e}")
    
    try:
        evaluator = VLMCapabilityEvaluator(
            model_name=model_name,
            device=args.device,
            test_image_path=args.test_image,
            cache_dir=args.cache_dir
        )
        
        results = evaluator.run_all_tests(save_results=not args.no_save)
        
        return 0
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())

