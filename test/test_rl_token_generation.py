"""
测试 RL token 生成效果与稳定性。
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

import torch
import torch.nn.functional as F

from ScriptedVLA.model.rl_token import RLTokenBottleneck


def test_rl_token_basic_shape_and_finite():
    print("=" * 60)
    print("测试1: RL token 形状与数值稳定性")
    print("=" * 60)
    bsz, seq_len, hid, rl_dim = 4, 32, 128, 64
    module = RLTokenBottleneck(
        input_dim=hid,
        model_dim=128,
        num_encoder_layers=2,
        num_decoder_layers=2,
        num_heads=8,
        rl_token_dim=rl_dim,
    )
    z = torch.randn(bsz, seq_len, hid)
    out = module.reconstruction_loss(z)
    zrl = out["z_rl"]
    assert zrl.shape == (bsz, rl_dim), f"shape mismatch: {zrl.shape}"
    assert torch.isfinite(zrl).all(), "z_rl contains NaN/Inf"
    assert torch.isfinite(out["loss"]).all(), "loss contains NaN/Inf"
    print(f"✓ z_rl shape: {zrl.shape}, loss={out['loss'].item():.6f}")
    return True


def test_rl_token_not_collapsed():
    print("\n" + "=" * 60)
    print("测试2: RL token 非塌缩检查")
    print("=" * 60)
    bsz, seq_len, hid = 8, 24, 96
    module = RLTokenBottleneck(input_dim=hid, model_dim=96, rl_token_dim=48)
    z = torch.randn(bsz, seq_len, hid)
    zrl = module.encode(z)
    std_mean = zrl.std(dim=0).mean().item()
    assert std_mean > 1e-6, f"z_rl collapsed, std_mean={std_mean}"
    print(f"✓ z_rl std(dim=0).mean()={std_mean:.6e}")
    return True


def test_rl_token_same_input_consistency():
    print("\n" + "=" * 60)
    print("测试3: eval 模式同输入一致性")
    print("=" * 60)
    module = RLTokenBottleneck(input_dim=64, model_dim=64, rl_token_dim=32, dropout=0.0)
    module.eval()
    z = torch.randn(2, 16, 64)
    with torch.no_grad():
        a = module.encode(z)
        b = module.encode(z)
    max_diff = (a - b).abs().max().item()
    assert max_diff < 1e-7, f"inconsistent eval outputs, max_diff={max_diff}"
    print(f"✓ max diff={max_diff:.3e}")
    return True


def test_rl_token_input_separability():
    print("\n" + "=" * 60)
    print("测试4: 不同输入可分性（余弦相似度）")
    print("=" * 60)
    module = RLTokenBottleneck(input_dim=80, model_dim=80, rl_token_dim=40, dropout=0.0)
    module.eval()
    z1 = torch.randn(1, 20, 80)
    z2 = torch.randn(1, 20, 80)
    with torch.no_grad():
        r1 = module.encode(z1)
        r2 = module.encode(z2)
    cos = F.cosine_similarity(r1, r2).item()
    assert abs(cos) < 0.999999, f"two different inputs produced nearly identical token, cos={cos}"
    print(f"✓ cosine_similarity={cos:.6f}")
    return True


def run_all_tests():
    results = []
    results.append(("shape_and_finite", test_rl_token_basic_shape_and_finite()))
    results.append(("not_collapsed", test_rl_token_not_collapsed()))
    results.append(("same_input_consistency", test_rl_token_same_input_consistency()))
    results.append(("input_separability", test_rl_token_input_separability()))

    print("\n" + "=" * 60)
    print("RL token 测试总结")
    print("=" * 60)
    passed = sum(1 for _, ok in results if ok)
    for name, ok in results:
        print(f"  {name}: {'✓ 通过' if ok else '✗ 失败'}")
    print(f"\n总计: {passed}/{len(results)} 通过")
    return passed == len(results)


if __name__ == "__main__":
    ok = run_all_tests()
    raise SystemExit(0 if ok else 1)
