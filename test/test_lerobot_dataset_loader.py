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
轻量化的LeRobot数据集加载测试脚本
用于快速验证数据集是否可以正常加载
"""

import sys
from pathlib import Path
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def test_import_lerobot():
    """测试lerobot库是否可以正常导入"""
    print("=" * 60)
    print("测试1: 导入lerobot库")
    print("=" * 60)
    try:
        from lerobot.datasets.lerobot_dataset import LeRobotDataset
        print("[OK] lerobot库导入成功")
        return True, LeRobotDataset
    except Exception as e:
        print(f"[FAIL] lerobot库导入失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_load_info_json(dataset_path: Path):
    """测试加载info.json文件"""
    print("\n" + "=" * 60)
    print("测试2: 加载info.json文件")
    print("=" * 60)
    try:
        info_file = dataset_path / "meta" / "info.json"
        if not info_file.exists():
            # 尝试根目录
            info_file = dataset_path / "info.json"
        
        if not info_file.exists():
            print(f"[FAIL] 无法找到info.json文件: {dataset_path}")
            return False, None
        
        print(f"  找到info.json: {info_file}")
        with open(info_file, 'r', encoding='utf-8') as f:
            info = json.load(f)
        
        print(f"[OK] info.json加载成功")
        print(f"  数据集版本: {info.get('codebase_version', 'unknown')}")
        print(f"  FPS: {info.get('fps', 'unknown')}")
        print(f"  总episodes: {info.get('total_episodes', 'unknown')}")
        print(f"  总frames: {info.get('total_frames', 'unknown')}")
        
        # 检查数据路径格式
        if 'data_path' in info:
            print(f"  数据路径格式: {info['data_path']}")
        
        return True, info
    except Exception as e:
        print(f"[FAIL] info.json加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_create_dataset(dataset_path: Path, LeRobotDataset):
    """测试创建LeRobotDataset对象"""
    print("\n" + "=" * 60)
    print("测试3: 创建LeRobotDataset对象")
    print("=" * 60)
    try:
        dataset_dir = Path(dataset_path).resolve()
        dataset_name = dataset_dir.name
        root_path_str = str(dataset_dir)
        
        print(f"  数据集路径: {dataset_dir}")
        print(f"  数据集名称: {dataset_name}")
        print(f"  Root路径: {root_path_str}")
        
        # 简单的delta_timestamps
        delta_timestamps = {"action": [0.1, 0.2, 0.3, 0.4, 0.5]}
        
        print(f"  创建LeRobotDataset对象...")
        dataset = LeRobotDataset(
            repo_id=dataset_name,
            root=root_path_str,
            delta_timestamps=delta_timestamps
        )
        
        print(f"[OK] LeRobotDataset创建成功")
        print(f"  数据集长度: {len(dataset)}")
        return True, dataset
    except Exception as e:
        print(f"[FAIL] LeRobotDataset创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None


def test_get_item(dataset):
    """测试获取数据集样本"""
    print("\n" + "=" * 60)
    print("测试4: 获取数据集样本")
    print("=" * 60)
    try:
        print(f"  尝试获取第一个样本...")
        sample = dataset[0]
        
        print(f"[OK] 样本获取成功")
        print(f"  样本键: {list(sample.keys())}")
        
        # 检查各个字段
        for key, value in sample.items():
            if hasattr(value, 'shape'):
                print(f"  {key}: shape={value.shape}, dtype={value.dtype}")
            elif isinstance(value, (list, tuple)):
                print(f"  {key}: type={type(value)}, length={len(value)}")
            else:
                print(f"  {key}: type={type(value)}")
        
        return True
    except Exception as e:
        print(f"[FAIL] 样本获取失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """主函数"""
    print("LeRobot数据集加载测试")
    print("=" * 60)
    
    # 测试1: 导入lerobot
    success, LeRobotDataset = test_import_lerobot()
    if not success:
        print("\n[ERROR] 测试失败: 无法导入lerobot库")
        print("\n建议:")
        print("1. 检查pandas库是否损坏，尝试重新安装: pip uninstall pandas && pip install pandas")
        print("2. 检查Python环境是否正常")
        print("3. 尝试重新安装lerobot: pip uninstall lerobot && pip install lerobot==0.3.3")
        return
    
    # 默认数据集路径
    default_dataset_path = Path("./dataset/libero_object/")
    
    # 检查数据集路径
    if len(sys.argv) > 1:
        dataset_path = Path(sys.argv[1])
    else:
        dataset_path = default_dataset_path
    
    if not dataset_path.exists():
        print(f"\n[ERROR] 数据集路径不存在: {dataset_path}")
        print(f"\n使用方法: python test_lerobot_dataset_loader.py [数据集路径]")
        print(f"默认路径: {default_dataset_path}")
        return
    
    # 测试2: 加载info.json
    success, info = test_load_info_json(dataset_path)
    if not success:
        print("\n[ERROR] 测试失败: 无法加载info.json")
        return
    
    # 测试3: 创建数据集对象
    success, dataset = test_create_dataset(dataset_path, LeRobotDataset)
    if not success:
        print("\n[ERROR] 测试失败: 无法创建LeRobotDataset对象")
        return
    
    # 测试4: 获取样本
    success = test_get_item(dataset)
    if not success:
        print("\n[ERROR] 测试失败: 无法获取数据集样本")
        return
    
    # 所有测试通过
    print("\n" + "=" * 60)
    print("[SUCCESS] 所有测试通过！数据集可以正常加载")
    print("=" * 60)


if __name__ == "__main__":
    main()
