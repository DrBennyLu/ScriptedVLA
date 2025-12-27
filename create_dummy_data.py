"""
创建虚拟数据集用于测试
"""

import argparse
from src.data import create_dummy_dataset


def main():
    parser = argparse.ArgumentParser(description="Create dummy dataset for testing")
    parser.add_argument(
        "--output",
        type=str,
        default="./data/train",
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
        default="./data/val",
        help="Output directory for validation data"
    )
    parser.add_argument(
        "--val_samples",
        type=int,
        default=20,
        help="Number of validation samples"
    )
    
    args = parser.parse_args()
    
    print("Creating dummy training dataset...")
    create_dummy_dataset(args.output, args.num_samples)
    
    print("\nCreating dummy validation dataset...")
    create_dummy_dataset(args.val_output, args.val_samples)
    
    print("\nDone! You can now use these datasets for training.")


if __name__ == "__main__":
    main()

