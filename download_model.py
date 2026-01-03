"""
下载Qwen VLM模型
支持下载Qwen2-VL-2B-Instruct等模型
"""

import argparse
import os
from pathlib import Path
from transformers import AutoModel, AutoProcessor, AutoTokenizer
import torch


def download_model(
    model_name: str = "Qwen/Qwen2-VL-2B-Instruct",
    cache_dir: str = None,
    force_download: bool = False
):
    """
    下载Qwen VLM模型
    
    Args:
        model_name: HuggingFace模型名称
        cache_dir: 缓存目录（可选）
        force_download: 是否强制重新下载
    """
    print(f"开始下载模型: {model_name}")
    print("=" * 60)
    
    # 设置缓存目录
    if cache_dir:
        cache_dir = Path(cache_dir)
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # 下载processor
        print("\n1. 下载Processor...")
        try:
            processor = AutoProcessor.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                force_download=force_download,
                trust_remote_code=True
            )
            print(f"   ✓ Processor下载完成")
        except Exception as e:
            print(f"   ⚠ Processor下载失败，尝试使用Tokenizer: {e}")
            processor = AutoTokenizer.from_pretrained(
                model_name,
                cache_dir=cache_dir,
                force_download=force_download,
                trust_remote_code=True
            )
            print(f"   ✓ Tokenizer下载完成")
        
        # 下载模型
        print("\n2. 下载模型...")
        model = AutoModel.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            force_download=force_download,
            trust_remote_code=True,
            torch_dtype=torch.float32
        )
        print(f"   ✓ 模型下载完成")
        
        # 显示模型信息
        print("\n3. 模型信息:")
        print(f"   - 模型名称: {model_name}")
        if hasattr(model, 'config'):
            config = model.config
            if hasattr(config, 'hidden_size'):
                print(f"   - 隐藏层维度: {config.hidden_size}")
            if hasattr(config, 'vocab_size'):
                print(f"   - 词汇表大小: {config.vocab_size}")
            if hasattr(config, 'num_attention_heads'):
                print(f"   - 注意力头数: {config.num_attention_heads}")
        
        # 计算模型大小
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"   - 总参数量: {total_params:,} ({total_params / 1e9:.2f}B)")
        print(f"   - 可训练参数: {trainable_params:,}")
        
        print("\n" + "=" * 60)
        print("✓ 模型下载完成！")
        print(f"\n模型已保存到缓存目录，可以在代码中使用:")
        print(f'  model_name = "{model_name}"')
        
        return model, processor
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的解决方案:")
        print("1. 检查网络连接")
        print("2. 确认模型名称正确")
        print("3. 检查HuggingFace访问权限")
        print("4. 尝试使用VPN或代理")
        raise


def main():
    parser = argparse.ArgumentParser(description="Download Qwen VLM Model")
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen2-VL-2B-Instruct",
        help="Model name (default: Qwen/Qwen2-VL-2B-Instruct)"
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default="./cache/models",
        help="Cache directory for model storage (default: ./cache/models)"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        default=False,
        help="Force re-download even if model exists"
    )
    
    args = parser.parse_args()
    
    try:
        download_model(
            model_name=args.model,
            cache_dir=args.cache_dir,
            force_download=args.force
        )
    except Exception as e:
        print(f"\n错误: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

