# ScriptedVLA + RL Token + TD3 训练流程（终端命令）

> 所有 ScriptedVLA 命令在 **ScriptedVLA 仓库根目录** 执行（`cd /home/lxg/ai_models/ScriptedVLA`）。  
> 评测 / rollout / 在线采集前，需先在 **LIBERO 环境** 启动 WebSocket 仿真服务（另开终端）：

```bash
# 终端 A：LIBERO 仓库（conda libero 环境）
cd /home/lxg/ai_models/LIBERO
python scripts/libero_ws_server.py --suite libero_object
```

```bash
# 终端 B：ScriptedVLA 仓库
cd /home/lxg/ai_models/ScriptedVLA
source .venv/bin/activate   # 或 uv sync && uv run ...
```

训练 / 评测前请确认 `libero/config_libero_object.yaml` 中 `dataset.task_index`、`training.save_dir`、`train_rl_token`、`train_rl_td3` 等路径与当前任务一致（默认示例为 **task_index=6**）。

---

## 1. Libero Object 数据集全量微调（VLA）

使用 `config_libero_object.yaml` 对 LeRobot 格式数据 `./dada/libero-object` 做全参数微调（`freeze_vlm: false`）：

```bash
bash libero/scripts/train_libero_object.sh
```

等价命令：

```bash
uv run python train.py \
  --config libero/config_libero_object.yaml \
  --dataset_path ./dada/libero-object
```

可选：10 个 task 预训练 + 单 task posttrain 两阶段流程：

```bash
bash libero/scripts/train_libero_object_pretrain.sh
bash libero/scripts/train_libero_object_posttrain.sh ./checkpoints/libero_object_pretrain/checkpoint_step_60000.pt
```

---

## 2. Rollout Video — 查看 VLA 训练效果

需 WebSocket 服务已启动。保存每条 rollout 的 MP4 到 `./results/rollout_videos/`：

```bash
python -m libero.libero_ws_eval_video \
  --config libero/config_libero_object.yaml \
  --checkpoint-dir ./checkpoints/libero_object_full_0527_200K \
  --task-id 2 \
  --num-rollouts 5 \
  --video-dir ./results/vla_rollout_videos \
  --camera both \
  --chunk-steps 10
```

---

## 3. Eval — 查看 VLA 整体成功率

不录视频，仅统计 closed-loop 成功率（`init_id=0..num_rollouts-1`）：

```bash
python -m libero.libero_ws_eval \
  --config libero/config_libero_object.yaml \
  --checkpoint-dir ./checkpoints/libero_object_full_0527_200K \
  --task-id 2 \
  --num-rollouts 50 \
  --chunk-steps 10
```

终端末尾输出：`[eval] summary: success_rate=XX.XX% (N rollouts)`。

全 benchmark 多 task 评测（可选）：

```bash
python -m libero.libero_benchmark_eval \
  --config libero/config_libero_object.yaml \
  --checkpoint-dir ./checkpoints/libero_object_full_0527_200K
```

---

## 4. 训练 RL Token 提取器

先在 yaml 的 `train_rl_token` 块配置 `vla_checkpoint`、`save_dir` 等，然后：

```bash
python train_rl_token.py --config libero/config_libero_object.yaml
```

### 4.1 余弦相似度 — 测试 RL Token 提取效果

在数据集单条 rollout 上，计算各帧 `z_rl[t]` 与首帧 `z_rl[0]` 的余弦相似度并绘图。  
训练有效时，除 `t=0` 外相似度应明显低于 1.0（脚本默认要求 `min < 0.99`）。

默认从 yaml `train_rl_token` 读取 VLA / RL Token 路径（RL 权重默认 `<save_dir>/rl_token_final.pt`）。

**仅跑 rollout 余弦相似度测试（推荐）：**

```bash
python test/test_rl_token_generation.py \
  --config libero/config_libero_object.yaml \
  --rollout-cosine-only
```

指定 episode、抽帧与输出图路径：

```bash
python test/test_rl_token_generation.py \
  --config libero/config_libero_object.yaml \
  --rollout-cosine-only \
  --episode-id 0 \
  --frame-stride 1 \
  --max-frames 200 \
  --output ./results/rl_token_task2_ep0_rollout_cosine.png
```

**稳定性 + 余弦相似度一并测试：**

```bash
python test/test_rl_token_generation.py --config libero/config_libero_object.yaml
```

终端关注：`cos to frame0 (t>0): min=..., max=..., mean=..., std=...`；  
曲线图默认保存到 `./results/rl_token_task{N}_ep{episode_id}_rollout_cosine.png`。

---

## 5. 提取数据集 Buffer Cache（离线 expert replay）

从 LeRobot 数据集预计算 `z_rl` 特征，生成 `replay_cache.pt`（供 TD3 离线训练）。  
需已训练好 VLA + RL Token（yaml 中 `train_rl_td3.vla_checkpoint` / `rl_token_checkpoint` 已指向有效权重）：

```bash
python train_rl_td3.py \
  --config libero/config_libero_object.yaml \
  --build-replay-only \
  --replay-cache ./data/replay_buffers/libero_object_task2/replay_cache.pt \
  --rebuild-cache
```

若缓存已存在且 meta 匹配，可去掉 `--rebuild-cache` 直接复用。

---

## 6. 提取 VLA 在线 Rollout Buffer Cache

在仿真中跑 VLA closed-loop，采集**成功** episode 写入 `vla_success_replay_cache.pt`（支持断点续采）。  
推理与 `libero_ws_eval` 一致：从 VLA checkpoint 加载 normalizer，对 WebSocket state 做 `align_joint_angles` + `normalize_state` 后再预测动作 / 编码 `z_rl`。

```bash
python -m libero.libero_ws_collect_vla_success_replay \
  --config libero/config_libero_object.yaml \
  --task-id 2 \
  --target-success-episodes 400
```

默认输出：`./data/replay_buffers/libero_object_task6/vla_success_replay_cache.pt`。

---

## （6 → 7 之间）离线训练 TD3

混合 expert cache + VLA success cache 训练 TD3 actor/critic：

```bash
python train_rl_td3.py \
  --config libero/config_libero_object.yaml \
  --vla-success-sample-ratio 0.5
```

权重默认保存到 `./checkpoints/libero_object_rl_td3_task6_0603/td3_agent_step_*.pt`。

---

## 7. Rollout Video — 查看 TD3 + RL 训练效果

VLA + RL Token + TD3 闭环 rollout，保存 MP4 与逐步 Q 曲线：

```bash
python -m libero.libero_ws_eval_td3_video \
  --config libero/config_libero_object.yaml \
  --td3-checkpoint ./checkpoints/libero_object_rl_td3_task2_0604/td3_agent_step_10000.pt \
  --task-id 2 \
  --num-rollouts 5 \
  --video-dir ./results/rl_td3_rollout_videos_0604 \
  --camera both \
  --chunk-steps 10
```

也可用 `--td3-step 10000` 代替 `--td3-checkpoint`（从 yaml `train_rl_td3.checkpoint.save_dir` 解析路径）。

---

## 8. Eval — 查看 TD3 + RL 整体成功率

当前 TD3 评测与 step 7 共用 `libero_ws_eval_td3_video`（会同时写 summary.json）。  
增大 `--num-rollouts` 做统计评测：

```bash
python -m libero.libero_ws_eval_td3_video \
  --config libero/config_libero_object.yaml \
  --td3-checkpoint ./checkpoints/libero_object_rl_td3_task6_0603/td3_agent_step_10000.pt \
  --task-id 6 \
  --num-rollouts 20 \
  --video-dir ./results/rl_td3_eval_0603 \
  --camera both \
  --chunk-steps 10
```

终端末尾输出：`[eval_td3_video] summary: success_rate=XX.XX% (N rollouts)`；  
详细结果：`./results/rl_td3_eval_0603/<timestamp>/summary.json`。

---

## 流程速查


| 步骤  | 目的                   | 关键命令                                                                 |
| --- | -------------------- | -------------------------------------------------------------------- |
| 0   | 启动仿真                 | `LIBERO`: `python scripts/libero_ws_server.py --suite libero_object` |
| 1   | VLA 全量微调             | `bash libero/scripts/train_libero_object.sh`                         |
| 2   | VLA rollout 视频       | `python -m libero.libero_ws_eval_video ...`                          |
| 3   | VLA 成功率              | `python -m libero.libero_ws_eval ...`                                |
| 4   | RL Token             | `python train_rl_token.py ...`                                       |
| 4.1 | RL Token 余弦相似度       | `python test/test_rl_token_generation.py --rollout-cosine-only ...`  |
| 5   | 数据集 replay cache     | `python train_rl_td3.py --build-replay-only ...`                     |
| 6   | VLA 成功 rollout cache | `python -m libero.libero_ws_collect_vla_success_replay ...`          |
| —   | 离线 TD3 训练            | `python train_rl_td3.py --vla-success-sample-ratio 0.6`              |
| 7   | TD3 rollout 视频       | `python -m libero.libero_ws_eval_td3_video ...`                      |
| 8   | TD3 成功率              | `python -m libero.libero_ws_eval_td3_video ... --num-rollouts 20`    |


