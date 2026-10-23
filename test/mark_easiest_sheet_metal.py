#!/usr/bin/env python3
"""Mark the easiest-to-grasp sheet metal using Qwen2-VL-2B.

Reuses VLMCapabilityEvaluator.generate_text_response() from
evaluate_vlm_capabilities.py, then draws a red center dot on the image.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image, ImageDraw

TEST_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(TEST_DIR))
sys.path.insert(0, str(TEST_DIR.parent / "src"))
sys.path.insert(0, str(TEST_DIR.parent))

from evaluate_vlm_capabilities import VLMCapabilityEvaluator  # noqa: E402

GRASP_PROMPT = """这是一张工业场景俯视图。画面中心料箱里堆叠着多片钣金件，请作为具身机器人大脑完成抓取点选择。

任务：找出当前状态下最容易被机械臂夹爪抓取的那一片钣金。
判断标准（按优先级）：
1. 位于料堆顶部、遮挡最少、露出面积最大
2. 姿态相对平整、边缘/中心可夹
3. 不要选夹爪、线缆、料笼金属框、背景墙壁或其他非钣金物体

请只输出一个 JSON 对象，不要输出其它文字。坐标使用 Qwen2-VL 惯例：相对整张图像的 0-1000 归一化整数。
格式：
{"reason":"一句话说明为何这片最容易抓","bbox_2d":[x1,y1,x2,y2],"center":[cx,cy]}
"""


def _to_floats(values: List[Any]) -> Optional[List[float]]:
    out: List[float] = []
    for v in values:
        try:
            out.append(float(v))
        except (TypeError, ValueError):
            return None
    return out


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    if fence:
        text = fence.group(1)
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            obj = json.loads(text[start : end + 1])
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            return None
    return None


def _looks_normalized(vals: List[float], width: int, height: int) -> bool:
    """Prompt asks for Qwen 0-1000 coords; treat as pixels only if a value exceeds 1000."""
    del width, height
    if any(v < 0 for v in vals):
        return False
    return max(vals) <= 1000.0 and max(vals) > 1.5


def _to_pixels(
    vals: List[float],
    width: int,
    height: int,
    n: int,
) -> Optional[Tuple[int, ...]]:
    if len(vals) < n:
        return None
    vals = vals[:n]
    if all(v <= 1.0 for v in vals):
        xs = [v * width if i % 2 == 0 else v * height for i, v in enumerate(vals)]
    elif _looks_normalized(vals, width, height):
        xs = [v / 1000.0 * width if i % 2 == 0 else v / 1000.0 * height for i, v in enumerate(vals)]
    else:
        xs = list(vals)
    ints = [int(round(v)) for v in xs]
    if n == 2:
        cx = max(0, min(width - 1, ints[0]))
        cy = max(0, min(height - 1, ints[1]))
        return (cx, cy)
    x1, y1, x2, y2 = ints[:4]
    x1 = max(0, min(width - 1, x1))
    x2 = max(0, min(width - 1, x2))
    y1 = max(0, min(height - 1, y1))
    y2 = max(0, min(height - 1, y2))
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return (x1, y1, x2, y2)


def parse_grasp_response(
    text: str, width: int, height: int
) -> Dict[str, Any]:
    reason = ""
    bbox_px: Optional[Tuple[int, int, int, int]] = None
    center_px: Optional[Tuple[int, int]] = None
    raw_bbox = None
    raw_center = None

    obj = _extract_json_object(text)
    if obj:
        reason = str(obj.get("reason") or obj.get("explanation") or "").strip()
        for key in ("bbox_2d", "bbox", "box"):
            if key in obj and isinstance(obj[key], (list, tuple)) and len(obj[key]) >= 4:
                raw_bbox = _to_floats(list(obj[key])[:4])
                break
        for key in ("center", "point", "center_2d"):
            if key in obj and isinstance(obj[key], (list, tuple)) and len(obj[key]) >= 2:
                raw_center = _to_floats(list(obj[key])[:2])
                break

    if raw_bbox is None:
        box_tag = re.search(
            r"<\|box_start\|>\s*\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?\s*,\s*\(?\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*\)?\s*<\|box_end\|>",
            text,
        )
        if box_tag:
            raw_bbox = _to_floats(list(box_tag.groups()))
        else:
            m = re.search(
                r"\[(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\s*,\s*(\d+(?:\.\d+)?)\]",
                text,
            )
            if m:
                raw_bbox = _to_floats(list(m.groups()))

    if raw_center is None:
        m = re.search(
            r'"center"\s*:\s*\[\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)\s*\]',
            text,
        )
        if m:
            raw_center = _to_floats(list(m.groups()))

    if raw_bbox is not None:
        parsed = _to_pixels(raw_bbox, width, height, 4)
        if parsed is not None:
            bbox_px = parsed  # type: ignore[assignment]
    if raw_center is not None:
        parsed_c = _to_pixels(raw_center, width, height, 2)
        if parsed_c is not None:
            center_px = parsed_c  # type: ignore[assignment]
    if bbox_px is not None:
        x1, y1, x2, y2 = bbox_px
        bbox_center = ((x1 + x2) // 2, (y1 + y2) // 2)
        if center_px is None:
            center_px = bbox_center
        else:
            cx, cy = center_px
            inside = (x1 <= cx <= x2) and (y1 <= cy <= y2)
            near_image_center = abs(cx - width / 2) < 8 and abs(cy - height / 2) < 8
            if (not inside) or near_image_center:
                center_px = bbox_center

    return {
        "reason": reason,
        "raw_bbox": raw_bbox,
        "raw_center": raw_center,
        "bbox_px": list(bbox_px) if bbox_px else None,
        "center_px": list(center_px) if center_px else None,
    }


def draw_mark(
    image: Image.Image,
    bbox_px: Optional[List[int]],
    center_px: Optional[List[int]],
) -> Image.Image:
    out = image.copy().convert("RGB")
    draw = ImageDraw.Draw(out)
    if bbox_px and len(bbox_px) == 4:
        x1, y1, x2, y2 = bbox_px
        draw.rectangle([x1, y1, x2, y2], outline=(255, 220, 0), width=3)
    if center_px and len(center_px) == 2:
        cx, cy = center_px
        r = max(10, min(out.size) // 80)
        draw.ellipse([cx - r - 3, cy - r - 3, cx + r + 3, cy + r + 3], outline=(255, 255, 255), width=4)
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(220, 20, 20), outline=(255, 255, 255), width=2)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Mark easiest-to-grasp sheet metal with Qwen2-VL")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2-VL-2B-Instruct")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--test-image", type=str, required=True)
    parser.add_argument("--cache-dir", type=str, default="./cache/models")
    parser.add_argument("--output", type=str, default="")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    args = parser.parse_args()

    image_path = Path(args.test_image).expanduser().resolve()
    if not image_path.is_file():
        print(f"ERROR: image not found: {image_path}", file=sys.stderr)
        return 1

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = image_path.with_name(image_path.stem + "_grasp_marked.png")
    json_path = output_path.with_suffix(".json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Qwen VLM easiest-grasp sheet-metal test")
    print(f"model: {args.model}")
    print(f"device: {args.device}")
    print(f"image: {image_path}")
    print(f"cache: {args.cache_dir}")
    print(f"time: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    evaluator = VLMCapabilityEvaluator(
        model_name=args.model,
        device=args.device,
        test_image_path=str(image_path),
        cache_dir=args.cache_dir,
    )
    image = evaluator._load_test_image(str(image_path))
    width, height = image.size
    print(f"image size: {width}x{height}")

    print("\n--- prompt ---")
    print(GRASP_PROMPT)
    response = evaluator.generate_text_response(
        image, GRASP_PROMPT, max_new_tokens=args.max_new_tokens
    )
    print("\n--- model response ---")
    print(response)
    print("--- end response ---")

    parsed = parse_grasp_response(response, width, height)
    print("\n--- parsed ---")
    print(json.dumps(parsed, ensure_ascii=False, indent=2))

    if not parsed.get("center_px"):
        print("ERROR: failed to parse a grasp center from model output", file=sys.stderr)
        payload = {
            "ok": False,
            "error": "no_center",
            "model": args.model,
            "image": str(image_path),
            "image_size": [width, height],
            "response": response,
            "parsed": parsed,
        }
        json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"wrote debug json: {json_path}")
        return 2

    marked = draw_mark(image, parsed.get("bbox_px"), parsed.get("center_px"))
    marked.save(output_path)
    payload = {
        "ok": True,
        "model": args.model,
        "device": args.device,
        "image": str(image_path),
        "output": str(output_path),
        "image_size": [width, height],
        "prompt": GRASP_PROMPT,
        "response": response,
        "parsed": parsed,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
    }
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nMARKED_OK {output_path}")
    print(f"JSON_OK {json_path}")
    print(f"center_px={parsed['center_px']} bbox_px={parsed.get('bbox_px')}")
    print(f"reason={parsed.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
