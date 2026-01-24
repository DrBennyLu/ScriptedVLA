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
自动下载Hugging Face数据集
下载 k1000dai/libero-object-smolvla 数据集到 ./dataset/libero_object
"""

import os
import sys
from pathlib import Path
import argparse


def download_libero_object_dataset(
    output_dir: str = "./dataset/libero_object"
) -> Path:
    """
    从Hugging Face下载 libero-object-smolvla 数据集
    
    Args:
        output_dir: 输出目录，默认为 ./dataset/libero_object
        
    Returns:
        数据集路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始下载数据集: k1000dai/libero-object-smolvla")
    print(f"目标目录: {output_path.absolute()}")
    print("=" * 60)
    
    # 检查并安装 datasets 库
    try:
        from datasets import load_dataset
    except ImportError:
        print("datasets 库未安装，正在安装...")
        os.system(f"{sys.executable} -m pip install datasets")
        from datasets import load_dataset
    
    try:
        # 从 Hugging Face 下载数据集
        print("\n正在从 Hugging Face 下载数据集...")
        dataset = load_dataset(
            "k1000dai/libero-object-smolvla",
            cache_dir=str(output_path),
            download_mode="reuse_cache_if_exists"
        )
        
        print(f"\n✓ 数据集下载成功！")
        print(f"数据集信息:")
        print(f"  - 数据集路径: {output_path.absolute()}")
        
        # 显示数据集结构信息
        if isinstance(dataset, dict):
            print(f"  - 数据分割: {list(dataset.keys())}")
            for split_name, split_data in dataset.items():
                print(f"    - {split_name}: {len(split_data)} 条样本")
        else:
            print(f"  - 样本数量: {len(dataset)}")
        
        # 如果数据集有特征信息，显示一些基本信息
        if hasattr(dataset, 'features'):
            print(f"  - 特征: {list(dataset.features.keys())}")
        
        print("\n数据集已准备就绪，可以用于训练。")
        
        return output_path
        
    except Exception as e:
        print(f"\n✗ 下载失败: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接问题")
        print("  2. Hugging Face 认证问题（某些数据集需要登录）")
        print("  3. 数据集名称或路径不正确")
        print("\n建议:")
        print("  - 检查网络连接")
        print("  - 尝试使用 Hugging Face CLI 登录: huggingface-cli login")
        print("  - 确认数据集名称: k1000dai/libero-object-smolvla")
        raise


def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="下载 k1000dai/libero-object-smolvla 数据集"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./dataset/libero_object",
        help="输出目录（默认: ./dataset/libero_object）"
    )
    
    args = parser.parse_args()
    
    try:
        path = download_libero_object_dataset(output_dir=args.output)
        print(f"\n{'=' * 60}")
        print(f"数据集已下载到: {path.absolute()}")
        print(f"{'=' * 60}")
    except Exception as e:
        print(f"\n下载过程出错: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
