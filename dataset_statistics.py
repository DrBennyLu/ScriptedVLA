"""
数据集统计和筛选工具
展示如何使用层次化数据结构
"""

import argparse
from pathlib import Path
import json

from ScriptedVLA.data import (
    VLADataset,
    filter_dataset_by_hierarchy,
    get_dataset_statistics
)
from ScriptedVLA.utils import load_config, get_data_config


def main():
    parser = argparse.ArgumentParser(description="Dataset Statistics and Filtering Tool")
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--task",
        type=str,
        nargs="+",
        default=None,
        help="Filter by task names"
    )
    parser.add_argument(
        "--episode",
        type=int,
        nargs="+",
        default=None,
        help="Filter by episode IDs"
    )
    parser.add_argument(
        "--step",
        type=int,
        nargs="+",
        default=None,
        help="Filter by step IDs"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file for statistics (JSON format)"
    )
    
    args = parser.parse_args()
    
    # 加载配置
    try:
        config = load_config(args.config)
        data_config = get_data_config(config)
        camera_names = data_config.get("cameras", {}).get("names", ["global_img", "left_wrist_img"])
        use_state = data_config.get("robot_state", {}).get("use_state", True)
        state_dim = data_config.get("robot_state", {}).get("state_dim", 7)
        image_size = data_config.get("image_size", 224)
    except:
        # 使用默认值
        camera_names = ["global_img", "left_wrist_img"]
        use_state = True
        state_dim = 7
        image_size = 224
    
    # 加载数据集
    print(f"Loading dataset from: {args.data_path}")
    dataset = VLADataset(
        data_path=args.data_path,
        image_size=image_size,
        camera_names=camera_names,
        use_state=use_state,
        state_dim=state_dim
    )
    
    print(f"\n原始数据集: {len(dataset)} 个样本")
    
    # 获取统计信息
    stats = get_dataset_statistics(dataset)
    
    print("\n" + "=" * 60)
    print("数据集统计信息")
    print("=" * 60)
    print(f"总样本数: {stats['total_samples']}")
    print(f"任务数量: {len(stats['tasks'])}")
    print(f"Episode数量: {len(stats['episodes'])}")
    print(f"Step数量: {len(stats['steps'])}")
    
    print("\n任务列表:")
    for task in stats['tasks']:
        episodes = stats['task_episode_map'].get(task, [])
        print(f"  - {task}: {len(episodes)} episodes")
    
    print("\n任务-Episode映射:")
    for task, episodes in stats['task_episode_map'].items():
        print(f"  {task}: {episodes}")
    
    # 筛选数据集（如果指定了筛选条件）
    if args.task or args.episode or args.step:
        print("\n" + "=" * 60)
        print("筛选数据集")
        print("=" * 60)
        
        filtered_dataset = filter_dataset_by_hierarchy(
            dataset,
            task_names=args.task,
            episode_ids=args.episode,
            step_ids=args.step
        )
        
        print(f"筛选条件:")
        if args.task:
            print(f"  任务: {args.task}")
        if args.episode:
            print(f"  Episode: {args.episode}")
        if args.step:
            print(f"  Step: {args.step}")
        print(f"筛选后样本数: {len(filtered_dataset)}")
        
        # 显示筛选后的统计信息
        if hasattr(filtered_dataset, 'indices'):
            filtered_tasks = set()
            filtered_episodes = set()
            for idx in filtered_dataset.indices:
                sample = dataset.samples[idx]
                filtered_tasks.add(sample.get('task_name', 'unknown'))
                filtered_episodes.add((sample.get('task_name', 'unknown'), sample.get('episode_id', -1)))
            print(f"筛选后任务: {len(filtered_tasks)} ({sorted(filtered_tasks)})")
            print(f"筛选后Episodes: {len(filtered_episodes)}")
    
    # 保存统计信息（如果指定）
    if args.output:
        output_path = Path(args.output)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n统计信息已保存到: {output_path}")


if __name__ == "__main__":
    main()

