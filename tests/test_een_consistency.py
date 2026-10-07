import torch

from beliefnet.models.legacy import StatementEnergyBaseline
from beliefnet.perception.set_baselines import AllPairsClassifier


def _model_and_input():
    torch.manual_seed(0)
    m = StatementEnergyBaseline(hidden_size=32, latent_dim=16, mlp_hidden=32, n_steps=15)
    b, k, h = 4, 5, 32
    emb = torch.randn(b, k, h)
    mask = torch.ones(b, k, dtype=torch.bool)
    mask[0, 4] = False     # padding
    mask[1, 3:] = False
    return m, emb, mask


def test_forward_shape_and_finite():
    m, emb, mask = _model_and_input()
    out = m(emb, mask)
    assert out.shape == (4,)
    assert torch.isfinite(out).all()


def test_equilibrium_reduces_energy():
    m, emb, mask = _model_and_input()
    traj = m.energy_trajectory(emb, mask)
    assert traj[-1] <= traj[0] + 1e-6


def test_gradients_flow():
    m, emb, mask = _model_and_input()
    m(emb, mask).sum().backward()
    grads = [p.grad for p in m.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)


def test_smaller_than_all_pairs_baseline():
    een = StatementEnergyBaseline(hidden_size=896, latent_dim=64)
    ap = AllPairsClassifier(896)
    assert een.num_params() < ap.num_params()
