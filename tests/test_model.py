import pytest

torch = pytest.importorskip("torch")

from mtt.model import ModelConfig, TinyGPT, greedy_generate  # noqa: E402


def test_default_size_is_5_to_10_million_parameters():
    m = TinyGPT(ModelConfig(vocab_size=2100))
    assert 5e6 <= m.num_parameters() <= 10e6


def test_forward_shapes_and_loss():
    m = TinyGPT(ModelConfig(vocab_size=100, block_size=32, n_layer=2, d_model=32, n_head=2))
    idx = torch.randint(0, 100, (3, 10))
    assert m(idx).shape == (3, 10, 100) and m(idx, last_only=True).shape == (3, 1, 100)
    mask = torch.zeros(3, 10, dtype=torch.bool)
    mask[:, 5:] = True
    assert m.compute_loss(idx, mask).item() > 0


def test_model_is_causal():
    m = TinyGPT(ModelConfig(vocab_size=50, block_size=16, n_layer=2, d_model=32, n_head=2)).eval()
    a = torch.randint(0, 50, (1, 8))
    b = a.clone()
    b[0, -1] = (b[0, -1] + 1) % 50
    assert torch.allclose(m(a)[:, :-1], m(b)[:, :-1], atol=1e-5)


def test_greedy_generate_returns_one_result_per_prompt():
    m = TinyGPT(ModelConfig(vocab_size=50, block_size=24, n_layer=1, d_model=16, n_head=2))
    prompts = [[1, 5, 6], [1, 7], [1, 8, 9]]
    out = greedy_generate(m, prompts, eos_id=2, max_new_tokens=5)
    assert len(out) == 3 and all(1 <= len(o) <= 5 for o in out)
