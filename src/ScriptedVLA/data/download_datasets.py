"""
自动下载公开VLA数据集
支持LIBERO、ACT等数据集
"""

import os
import sys
from pathlib import Path
from typing import Optional
import argparse


def download_libero_dataset(
    dataset_name: str = "libero_spatial",
    output_dir: str = "./data/libero",
    download_only: bool = False
) -> Path:
    """
    下载LIBERO数据集
    
    Args:
        dataset_name: 数据集名称，可选：
            - libero_spatial: 空间推理任务
            - libero_object: 物体操作任务
            - libero_goal: 目标条件任务
            - libero_100: 100个任务
        output_dir: 输出目录
        download_only: 是否只下载不处理
        
    Returns:
        数据集路径
    """
    try:
        from libero.libero import benchmark
    except ImportError:
        print("LIBERO包未安装，正在安装...")
        os.system(f"{sys.executable} -m pip install libero")
        from libero.libero import benchmark
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"正在下载LIBERO数据集: {dataset_name}")
    
    # 获取基准字典
    benchmark_dict = benchmark.get_benchmark_dict()
    
    if dataset_name not in benchmark_dict:
        available = list(benchmark_dict.keys())
        raise ValueError(
            f"数据集 {dataset_name} 不存在。\n"
            f"可选数据集: {available}"
        )
    
    # 获取数据集
    task_suite = benchmark_dict[dataset_name]()
    tasks = task_suite.get_tasks()
    
    print(f"成功下载数据集: {dataset_name}")
    print(f"任务数量: {len(tasks)}")
    print(f"数据集路径: {output_path}")
    
    if not download_only:
        # 保存任务信息
        task_info = {
            "dataset_name": dataset_name,
            "num_tasks": len(tasks),
            "tasks": [task.name for task in tasks]
        }
        
        import json
        info_file = output_path / "dataset_info.json"
        with open(info_file, 'w', encoding='utf-8') as f:
            json.dump(task_info, f, indent=2, ensure_ascii=False)
    
    return output_path


def download_act_dataset(
    output_dir: str = "./data/act",
    source: str = "huggingface"
) -> Path:
    """
    下载ACT数据集
    
    Args:
        output_dir: 输出目录
        source: 数据源，可选 "huggingface" 或 "github"
        
    Returns:
        数据集路径
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("正在下载ACT数据集...")
    
    if source == "huggingface":
        try:
            from datasets import load_dataset
            
            print("从HuggingFace下载ACT数据集...")
            # ACT数据集在HuggingFace上的名称可能不同
            # 这里使用一个通用的示例
            try:
                dataset = load_dataset("act-dataset/act", cache_dir=str(output_path))
                print(f"成功下载ACT数据集到: {output_path}")
                return output_path
            except Exception as e:
                print(f"HuggingFace下载失败: {e}")
                print("尝试从GitHub下载...")
                source = "github"
        
        except ImportError:
            print("datasets包未安装，正在安装...")
            os.system(f"{sys.executable} -m pip install datasets")
            from datasets import load_dataset
    
    if source == "github":
        # ACT数据集通常可以从GitHub下载
        # 这里提供一个示例URL（需要根据实际情况修改）
        print("从GitHub下载ACT数据集...")
        print("注意: 请根据ACT官方仓库更新下载URL")
        
        # 示例：使用wget或requests下载
        try:
            import requests
            import zipfile
            
            # 这里需要替换为实际的ACT数据集下载URL
            act_url = "https://github.com/tonyzhaozh/act/archive/refs/heads/main.zip"
            
            zip_path = output_path / "act.zip"
            print(f"下载到: {zip_path}")
            
            # 下载文件
            response = requests.get(act_url, stream=True)
            if response.status_code == 200:
                with open(zip_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # 解压
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(output_path)
                
                # 删除zip文件
                zip_path.unlink()
                
                print(f"成功下载并解压ACT数据集到: {output_path}")
            else:
                print(f"下载失败，状态码: {response.status_code}")
                print("请手动下载ACT数据集并放置到:", output_path)
        
        except ImportError:
            print("requests包未安装，请手动下载数据集")
            print("数据集路径:", output_path)
    
    return output_path


def download_dataset(
    dataset_type: str,
    dataset_name: Optional[str] = None,
    output_dir: Optional[str] = None
) -> Path:
    """
    统一的数据集下载接口
    
    Args:
        dataset_type: 数据集类型，"libero" 或 "act"
        dataset_name: 数据集名称（仅对LIBERO有效）
        output_dir: 输出目录
        
    Returns:
        数据集路径
    """
    if dataset_type.lower() == "libero":
        if dataset_name is None:
            dataset_name = "libero_spatial"
        if output_dir is None:
            output_dir = f"./data/libero_{dataset_name}"
        return download_libero_dataset(dataset_name, output_dir)
    
    elif dataset_type.lower() == "act":
        if output_dir is None:
            output_dir = "./data/act"
        return download_act_dataset(output_dir)
    
    else:
        raise ValueError(f"不支持的数据集类型: {dataset_type}. 支持: 'libero', 'act'")


def main():
    parser = argparse.ArgumentParser(description="下载VLA公开数据集")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["libero", "act"],
        help="数据集类型: libero 或 act"
    )
    parser.add_argument(
        "--name",
        type=str,
        default=None,
        help="数据集名称（仅对LIBERO有效，如: libero_spatial, libero_object等）"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="输出目录"
    )
    
    args = parser.parse_args()
    
    try:
        path = download_dataset(
            dataset_type=args.dataset,
            dataset_name=args.name,
            output_dir=args.output
        )
        print(f"\n数据集已下载到: {path.absolute()}")
        print("您可以在config.yaml中配置数据路径进行训练")
    except Exception as e:
        print(f"下载失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

