#!/usr/bin/env python3
"""Apply DDP support to ScriptedVLA/train.py. Writes after each stage (resumable)."""

from __future__ import annotations

import re
from pathlib import Path

PATH = Path("/data/share/1919650160032350208/users/lxg/ScriptedVLA/train.py")

HELPERS = '''
def _ddp_enabled() -> bool:
    return "WORLD_SIZE" in os.environ and int(os.environ["WORLD_SIZE"]) > 1


def setup_distributed():
    """Initialize torch.distributed when launched via torchrun."""
    if not _ddp_enabled():
        return {
            "enabled": False,
            "rank": 0,
            "local_rank": 0,
            "world_size": 1,
            "is_main": True,
        }

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    if not torch.distributed.is_initialized():
        torch.distributed.init_process_group(backend=backend)
    return {
        "enabled": True,
        "rank": rank,
        "local_rank": local_rank,
        "world_size": world_size,
        "is_main": rank == 0,
    }


def cleanup_distributed():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()
        torch.distributed.destroy_process_group()


def unwrap_model(model):
    return model.module if isinstance(model, torch.nn.parallel.DistributedDataParallel) else model


def dist_print(is_main: bool, *args, **kwargs):
    if is_main:
        print(*args, **kwargs)


def dist_barrier(enabled: bool):
    if enabled and torch.distributed.is_initialized():
        torch.distributed.barrier()

'''


def save(text: str) -> None:
    PATH.write_text(text, encoding="utf-8")
    compile(text, str(PATH), "exec")


def main() -> None:
    text = PATH.read_text(encoding="utf-8")

    if "DistributedSampler" not in text:
        text = text.replace(
            "from torch.utils.data import DataLoader\n",
            "from torch.utils.data import DataLoader, DistributedSampler\n"
            "from torch.nn.parallel import DistributedDataParallel as DDP\n",
            1,
        )
        save(text)
        print("stage imports")

    if "def setup_distributed(" not in text:
        text = text.replace("def set_seed(seed: int):", HELPERS + "def set_seed(seed: int):", 1)
        save(text)
        print("stage helpers")

    if "dist_info = setup_distributed()" not in text:
        old = (
            '    print("=" * 60)\n'
            '    print("LeRobot数据集训练")\n'
            '    print("=" * 60)\n\n'
            "    if not HAS_LEROBOT:\n"
        )
        new = (
            "    dist_info = setup_distributed()\n"
            '    is_main = dist_info["is_main"]\n'
            '    local_rank = dist_info["local_rank"]\n'
            '    world_size = dist_info["world_size"]\n\n'
            '    dist_print(is_main, "=" * 60)\n'
            '    dist_print(is_main, "LeRobot数据集训练")\n'
            "    dist_print(is_main, f\"DDP enabled={dist_info['enabled']} "
            "world_size={world_size} local_rank={local_rank}\")\n"
            '    dist_print(is_main, "=" * 60)\n\n'
            "    if not HAS_LEROBOT:\n"
        )
        if old not in text:
            raise RuntimeError("train header not found")
        text = text.replace(old, new, 1)
        save(text)
        print("stage train_start")

    if "model = DDP(" not in text:
        pat = re.compile(
            r'    device = torch\.device\("cuda" if torch\.cuda\.is_available\(\) else "cpu"\)\n'
            r"    model = model\.to\(device\)\n"
            r'    print\(f"\s*模型已移动到设备: \{device\}"\)\n'
        )
        repl = (
            '    if dist_info["enabled"]:\n'
            '        device = torch.device(f"cuda:{local_rank}")\n'
            "    else:\n"
            '        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")\n'
            "    model = model.to(device)\n"
            '    if dist_info["enabled"]:\n'
            "        model = DDP(\n"
            "            model,\n"
            "            device_ids=[local_rank],\n"
            "            output_device=local_rank,\n"
            "            find_unused_parameters=True,\n"
            "        )\n"
            "    dist_print(is_main, f\"  模型已移动到设备: {device} "
            f"(ddp={{dist_info['enabled']}})\")\n"
        )
        text2, n = pat.subn(repl, text, count=1)
        if n != 1:
            raise RuntimeError(f"device_ddp n={n}")
        text = text2
        save(text)
        print("stage device_ddp")

    if "train_sampler = None" not in text:
        old = "    train_loader = DataLoader(lerobot_dataset, **dataloader_kwargs)\n"
        if old not in text:
            raise RuntimeError("dataloader line missing")
        text = text.replace(
            old,
            "    train_sampler = None\n"
            '    if dist_info["enabled"]:\n'
            "        train_sampler = DistributedSampler(\n"
            "            lerobot_dataset,\n"
            "            num_replicas=world_size,\n"
            '            rank=dist_info["rank"],\n'
            '            shuffle=dataloader_kwargs.get("shuffle", True),\n'
            "            drop_last=False,\n"
            "        )\n"
            '        dataloader_kwargs["shuffle"] = False\n'
            '        dataloader_kwargs["sampler"] = train_sampler\n'
            "    train_loader = DataLoader(lerobot_dataset, **dataloader_kwargs)\n"
            "    dist_print(is_main, f\"  per-GPU batch_size={batch_size}, "
            "global_batch≈{batch_size * world_size}\")\n",
            1,
        )
        save(text)
        print("stage sampler")

    if "disable=not is_main" not in text:
        text2, n = re.subn(
            r"    progress_bar = tqdm\(range\(start_step, max_steps\), "
            r'initial=start_step, total=max_steps, desc="Training"\)\n',
            "    progress_bar = tqdm(\n"
            "        range(start_step, max_steps),\n"
            "        initial=start_step,\n"
            "        total=max_steps,\n"
            '        desc="Training",\n'
            "        disable=not is_main,\n"
            "    )\n",
            text,
            count=1,
        )
        if n != 1:
            raise RuntimeError(f"tqdm n={n}")
        text = text2
        save(text)
        print("stage tqdm")

    if "train_sampler.set_epoch" not in text:
        old = (
            "        except StopIteration:\n"
            "            loader_iter = iter(train_loader)\n"
            "            batch = next(loader_iter)\n"
        )
        if old not in text:
            raise RuntimeError("StopIteration block missing")
        text = text.replace(
            old,
            "        except StopIteration:\n"
            "            if train_sampler is not None:\n"
            "                train_sampler.set_epoch(step + 1)\n"
            "            loader_iter = iter(train_loader)\n"
            "            batch = next(loader_iter)\n",
            1,
        )
        save(text)
        print("stage sampler_epoch")

    # unwrap for I/O / attribute access (avoid matching helper def)
    if "save_checkpoint(unwrap_model(model)," not in text:
        text = text.replace("save_checkpoint(model,", "save_checkpoint(unwrap_model(model),")
        save(text)
        print("stage unwrap_save")
    if "unwrap_model(model).load_state_dict" not in text and "model.load_state_dict" in text:
        text = text.replace("model.load_state_dict", "unwrap_model(model).load_state_dict")
        save(text)
        print("stage unwrap_load")
    if "unwrap_model(model).qwen_vl_interface" not in text:
        text = text.replace("model.qwen_vl_interface", "unwrap_model(model).qwen_vl_interface")
        text = text.replace(
            "model.action_model.parameters()",
            "unwrap_model(model).action_model.parameters()",
        )
        save(text)
        print("stage unwrap_attrs")
    if "load_init_checkpoint" in text and "unwrap_model(model)," not in text.split("load_init_checkpoint")[1][:200]:
        text2, n = re.subn(
            r"(load_init_checkpoint\(\s*\n\s*init_path,\s*\n\s*)model,",
            r"\1unwrap_model(model),",
            text,
            count=1,
        )
        if n == 1:
            text = text2
            save(text)
            print("stage unwrap_init")
    if "evaluate(unwrap_model(model)," not in text:
        text = text.replace("evaluate(model,", "evaluate(unwrap_model(model),")
        text = text.replace(
            "evaluate(\n                model,",
            "evaluate(\n                unwrap_model(model),",
        )
        save(text)
        print("stage unwrap_eval")

    # Gate checkpoint writes: checkpoint_path = ...
    if "DDP_SAVE_GATE" not in text:
        old = (
            '            checkpoint_path = save_dir / f"checkpoint_step_{step + 1}.pt"\n'
            "            save_checkpoint(\n"
            "                unwrap_model(model), optimizer, scheduler, 0, loss_value, checkpoint_path,"
        )
        # trailing may vary; do flexible regex
        pat = re.compile(
            r'([ \t]*)checkpoint_path = save_dir / f"checkpoint_step_\{step \+ 1\}\.pt"\n'
            r"([ \t]*)save_checkpoint\(\n"
            r"([ \t]*)(unwrap_model\(model\)|model), optimizer, scheduler, 0, loss_value, checkpoint_path,",
        )

        def repl(m: re.Match[str]) -> str:
            ind = m.group(1)
            return (
                f"{ind}# DDP_SAVE_GATE\n"
                f"{ind}if is_main:\n"
                f'{ind}    checkpoint_path = save_dir / f"checkpoint_step_{{step + 1}}.pt"\n'
                f"{ind}    save_checkpoint(\n"
                f"{ind}        unwrap_model(model), optimizer, scheduler, 0, loss_value, checkpoint_path,"
            )

        text2, n = pat.subn(repl, text, count=1)
        if n != 1:
            # fallback: only wrap assignment + call start without assuming unwrap already
            pat2 = re.compile(
                r'([ \t]*)checkpoint_path = save_dir / f"checkpoint_step_\{step \+ 1\}\.pt"\n'
                r"([ \t]*)save_checkpoint\(\n"
                r"([ \t]*)(.+?), optimizer, scheduler, 0, loss_value, checkpoint_path,",
            )

            def repl2(m: re.Match[str]) -> str:
                ind = m.group(1)
                model_expr = m.group(4)
                if "unwrap_model" not in model_expr:
                    model_expr = "unwrap_model(model)"
                return (
                    f"{ind}# DDP_SAVE_GATE\n"
                    f"{ind}if is_main:\n"
                    f'{ind}    checkpoint_path = save_dir / f"checkpoint_step_{{step + 1}}.pt"\n'
                    f"{ind}    save_checkpoint(\n"
                    f"{ind}        {model_expr}, optimizer, scheduler, 0, loss_value, checkpoint_path,"
                )

            text2, n = pat2.subn(repl2, text, count=1)
            if n != 1:
                raise RuntimeError(f"save_gate failed n={n}")
        text = text2
        # Need to indent the remaining args of save_checkpoint and following print.
        # Find DDP_SAVE_GATE and indent consecutive lines that were body of save until blank/outdent.
        lines = text.splitlines(keepends=True)
        out = []
        i = 0
        while i < len(lines):
            out.append(lines[i])
            if "# DDP_SAVE_GATE" in lines[i]:
                # next: if is_main / checkpoint_path / save_checkpoint( / first arg line already done
                i += 1
                # copy until we pass the save_checkpoint opening and first arg
                while i < len(lines) and (
                    "if is_main:" in lines[i]
                    or "checkpoint_path =" in lines[i]
                    or "save_checkpoint(" in lines[i]
                    or "unwrap_model(model), optimizer" in lines[i]
                ):
                    out.append(lines[i])
                    i += 1
                # indent remaining args until closing paren line of save_checkpoint
                while i < len(lines):
                    ln = lines[i]
                    stripped = ln.lstrip(" ")
                    # original body under `if (step...)` is typically 12 spaces
                    if ln.startswith("            ") and not ln.startswith("                "):
                        # likely sibling of checkpoint_path - should become inside if is_main
                        if stripped.startswith(("global_step=", "normalizer=", ")")) or stripped.startswith(")"):
                            out.append("    " + ln)
                            i += 1
                            if stripped.startswith(")"):
                                break
                            continue
                        if "print(" in ln or "progress_bar" in ln:
                            out.append("    " + ln)
                            i += 1
                            continue
                        break
                    if ln.startswith("                "):
                        # already deeper (kwargs) - add 4 more for if is_main
                        out.append("    " + ln)
                        i += 1
                        continue
                    break
                out.append("            dist_barrier(dist_info['enabled'])\n")
                continue
            i += 1
        text = "".join(out)
        save(text)
        print("stage save_gate")

    idx = text.rfind("    return model, losses")
    if idx < 0:
        raise RuntimeError("return missing")
    if "cleanup_distributed()" not in text[max(0, idx - 200) : idx]:
        text = text[:idx] + "    cleanup_distributed()\n" + text[idx:]
        save(text)
        print("stage cleanup")

    print("DDP_PATCH_OK")


if __name__ == "__main__":
    main()
