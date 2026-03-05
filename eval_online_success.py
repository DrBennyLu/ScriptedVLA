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
在线仿真成功率评估脚本（Validation: Online Simulation Success Rate）

加载指定 checkpoint，在 PickPlace 仿真中运行多局 episode，统计任务成功率与平均推理轮数。
用于验证阶段与 loss 一起评估策略质量（loss 低但成功率不升时可据此调整数据或训练）。
"""

import argparse
from pathlib import Path

from online_simulation_inference import run_online_simulation


def main():
    parser = argparse.ArgumentParser(
        description="Eval trained VLA: run N episodes in pick-place sim, report success rate and mean rounds."
    )
    parser.add_argument("--checkpoint_dir", type=str, default="./checkpoints", help="Checkpoint directory")
    parser.add_argument("--config", type=str, default="config.yaml", help="Config path")
    parser.add_argument("--device", type=str, default=None, help="cuda or cpu")
    parser.add_argument("--num_episodes", type=int, default=5, help="Number of episodes to run")
    parser.add_argument("--max_rounds", type=int, default=50, help="Max inference rounds per episode")
    parser.add_argument("--seed", type=int, default=42, help="Base random seed (episode i uses seed + i)")
    parser.add_argument("--no_gui", action="store_true", help="Disable GUI (headless eval)")
    parser.add_argument("--instruction", type=str, default="Pick up the red cube and place it in the box.")
    parser.add_argument("--chunk_steps", type=int, default=None, help="Steps per round (receding horizon)")
    parser.add_argument("--first_step_alpha", type=float, default=0.3, help="First-step blend alpha")
    args = parser.parse_args()

    results = []
    for ep in range(args.num_episodes):
        ep_seed = (args.seed + ep) if args.seed is not None else None
        done, rounds = run_online_simulation(
            checkpoint_dir=args.checkpoint_dir,
            config_path=args.config,
            device=args.device,
            use_gui=not args.no_gui,
            seed=ep_seed,
            instruction=args.instruction,
            max_inference_rounds=args.max_rounds,
            smooth_first_step=True,
            first_step_alpha=args.first_step_alpha,
            chunk_execution_steps=args.chunk_steps,
            debug_print_ranges=False,
            quiet=True,
        )
        results.append((done, rounds))

    success_count = sum(1 for d, _ in results if d)
    rounds_list = [r for _, r in results]
    mean_rounds = sum(rounds_list) / len(rounds_list) if rounds_list else 0
    success_rate = success_count / len(results) if results else 0

    print("=" * 60)
    print("Online simulation success evaluation")
    print("=" * 60)
    print(f"  Episodes:     {len(results)}")
    print(f"  Successes:    {success_count}")
    print(f"  Success rate: {success_rate:.2%}")
    print(f"  Mean rounds:  {mean_rounds:.1f}")
    if rounds_list:
        min_r = min(rounds_list)
        max_r = max(rounds_list)
        print(f"  Rounds range: [{min_r}, {max_r}]")
    print("=" * 60)


if __name__ == "__main__":
    main()
