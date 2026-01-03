"""
创建虚拟数据集用于测试
"""

import argparse
from src.ScriptedVLA.data.dataset import create_dummy_dataset


def main():
    parser = argparse.ArgumentParser(description="Create dummy dataset for testing")
    parser.add_argument(
        "--output",
        type=str,
        default="./dataset/train",
        help="Output directory for training data"
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=100,
        help="Number of samples to create"
    )
    parser.add_argument(
        "--val_output",
        type=str,
        default="./dataset/val",
        help="Output directory for validation data"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=20,
        help="Number of validation samples"
    )
    parser.add_argument(
        "--cameras",
        type=str,
        nargs="+",
        default=["global_img", "left_wrist_img"],
        help="Camera names (e.g., global_img left_wrist_img)"
    )
    parser.add_argument(
        "--use_state",
        action="store_true",
        default=True,
        help="Include robot state information"
    )
    parser.add_argument(
        "--no_state",
        dest="use_state",
        action="store_false",
        help="Disable robot state information"
    )
    parser.add_argument(
        "--state_dim",
        type=int,
        default=7,
        help="Robot state dimension"
    )
    parser.add_argument(
        "--action_dim",
        type=int,
        default=7,
        help="Action dimension"
    )
    parser.add_argument(
        "--num_tasks",
        type=int,
        default=3,
        help="Number of tasks"
    )
    parser.add_argument(
        "--episodes_per_task",
        type=int,
        default=5,
        help="Number of episodes per task"
    )
    parser.add_argument(
        "--steps_per_episode",
        type=int,
        default=10,
        help="Number of steps per episode"
    )
    
    args = parser.parse_args()
    
    print("Creating dummy training dataset...")
    create_dummy_dataset(
        args.output,
        args.num_samples,
        camera_names=args.cameras,
        use_state=args.use_state,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_tasks=args.num_tasks,
        episodes_per_task=args.episodes_per_task,
        steps_per_episode=args.steps_per_episode
    )
    
    print("\nCreating dummy validation dataset...")
    create_dummy_dataset(
        args.val_output,
        args.val_samples,
        camera_names=args.cameras,
        use_state=args.use_state,
        state_dim=args.state_dim,
        action_dim=args.action_dim,
        num_tasks=max(1, args.num_tasks // 2),  # 验证集使用更少的任务
        episodes_per_task=args.episodes_per_task,
        steps_per_episode=args.steps_per_episode
    )
    
    print("\nDone! You can now use these datasets for training.")


if __name__ == "__main__":
    main()

