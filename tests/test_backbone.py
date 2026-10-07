import torch

from beliefnet.encoders.frozen_backbone import FrozenBackboneConfig, pool_hidden


def test_config_defaults_target_turing():
    cfg = FrozenBackboneConfig()
    assert cfg.compute_dtype == "float16"      # no bf16 on TITAN RTX
    assert cfg.attn_implementation == "sdpa"   # no FlashAttention-2 on Turing
    assert cfg.torch_dtype == torch.float16


def test_config_from_yaml(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("model_name: tiny\nload_in_4bit: false\npooling: mean\n")
    cfg = FrozenBackboneConfig.from_yaml(p)
    assert cfg.model_name == "tiny"
    assert cfg.load_in_4bit is False
    assert cfg.pooling == "mean"


def test_config_from_yaml_rejects_unknown_keys(tmp_path):
    p = tmp_path / "b.yaml"
    p.write_text("not_a_field: 1\n")
    try:
        FrozenBackboneConfig.from_yaml(p)
        assert False, "expected ValueError for unknown key"
    except ValueError:
        pass


def test_pool_mean_ignores_padding():
    hidden = torch.tensor([[[1.0, 1.0], [3.0, 3.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    out = pool_hidden(hidden, mask, "mean")
    assert torch.allclose(out, torch.tensor([[2.0, 2.0]]))


def test_pool_last_selects_last_real_token():
    hidden = torch.tensor([[[1.0, 1.0], [5.0, 7.0], [99.0, 99.0]]])
    mask = torch.tensor([[1, 1, 0]])
    out = pool_hidden(hidden, mask, "last")
    assert torch.allclose(out, torch.tensor([[5.0, 7.0]]))


def test_pool_shapes():
    B, T, H = 4, 6, 8
    hidden = torch.randn(B, T, H)
    mask = torch.ones(B, T, dtype=torch.long)
    assert pool_hidden(hidden, mask, "mean").shape == (B, H)
    assert pool_hidden(hidden, mask, "last").shape == (B, H)
