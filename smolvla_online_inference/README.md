# SmolVLA 在线仿真推理

使用 [LeRobot](https://github.com/huggingface/lerobot) 官方 SmolVLA 模型，在 PickPlace 仿真环境中进行实时闭环推理。

## 前置条件

1. 使用 `test/test_episode_data_collection.py` 采集 LeRobot 格式数据集（含 `top_image`, `wrist_image`, `observation.state`, `action`）
2. 使用 `lerobot-train` 训练 SmolVLA，例如：

```bash
cd lerobot && lerobot-train \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id=${HF_USER}/mydataset \
  --batch_size=64 \
  --steps=20000 \
  --output_dir=outputs/train/my_smolvla \
  --job_name=my_smolvla_training \
  --policy.device=cuda \
  --wandb.enable=true
```

## 安装

```bash
pip install -r smolvla_online_inference/requirements.txt
```

或单独安装：

```bash
pip install "lerobot[smolvla]"
```

本项目依赖父目录的 `simulator` 模块（PickPlaceEnv），需在 ScriptedVLA 项目根目录下运行。

## 运行

在 ScriptedVLA 项目根目录下执行：

```bash
# 默认 checkpoint 与红方块任务
python smolvla_online_inference/run_inference.py

# 指定 checkpoint 目录
python smolvla_online_inference/run_inference.py --checkpoint_dir outputs/train/my_smolvla

# 蓝方块任务
python smolvla_online_inference/run_inference.py --instruction "Pick up the blue cube and place it in the box."

# 无 GUI（DIRECT 仿真）
python smolvla_online_inference/run_inference.py --no_gui

# Receding horizon：每轮只执行 5 步
python smolvla_online_inference/run_inference.py --chunk_steps 5 --max_rounds 30

# 禁用第一步平滑
python smolvla_online_inference/run_inference.py --no_smooth_first_step
```

## 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--checkpoint_dir` | `outputs/train/my_smolvla` | lerobot 训练输出目录 |
| `--device` | 自动 | cuda 或 cpu |
| `--no_gui` | False | 禁用 GUI，使用 DIRECT 仿真 |
| `--instruction` | "Pick up the red cube..." | 任务指令 |
| `--max_rounds` | 50 | 最大推理轮数 |
| `--step_delay` | 0.02 | 每步仿真延时（秒） |
| `--chunk_steps` | None | 每轮执行步数（None=整段 chunk） |
| `--no_smooth_first_step` | False | 禁用第一步 EE 平滑 |
| `--first_step_alpha` | 0.3 | 第一步平滑系数 |

## 目录说明

本文件夹与主项目隔离，仅包含 SmolVLA 推理相关脚本，不修改 `test/`、`src/`、`simulator/` 等现有代码。`run_inference.py` 通过 `sys.path` 导入父项目的 `PickPlaceEnv`。
