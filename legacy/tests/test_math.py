from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch
from math_utils import cosine_matrix, evidence_separation, average_states


def test_cosine_matrix_identity():
    x = torch.tensor([[1., 0.], [0., 1.], [1., 1.]], dtype=torch.float64)
    s = cosine_matrix(x)
    assert torch.allclose(s.diagonal(), torch.ones(3, dtype=torch.float64))
    assert torch.isclose(s[0, 1], torch.tensor(0., dtype=torch.float64))
    assert torch.isclose(s[0, 2], torch.tensor(2 ** -0.5, dtype=torch.float64))


def test_evidence_separation_with_true_updates():
    x = torch.tensor([[2., 2., 0.], [2., 2., 0.], [-2., 1., 0.], [0., -2., 1.], [0., 0., 1.]], dtype=torch.float64)
    e = evidence_separation(x, [0, 1])
    assert e['attack_pair_raw_cos'] > 0.999
    assert -1 <= e['attack_pair_residual_cos'] <= 1
    assert e['benign_pair_count'] == 3


def test_fedavg_state_average():
    s0 = {'A': torch.tensor([1., 2.])}
    s1 = {'A': torch.tensor([3., 6.])}
    out = average_states([s0, s1])
    assert torch.equal(out['A'], torch.tensor([2., 4.]))


def test_effective_gram_matches_explicit_lora_matrix():
    from math_utils import effective_lora_gram
    torch.manual_seed(33)
    global_state = {}
    states = [{}, {}]
    for i in range(4):
        stem = f'model.layers.{i}.self_attn.q_proj'
        k_a = stem + '.lora_A.weight'
        k_b = stem + '.lora_B.weight'
        global_state[k_a] = torch.randn(2, 5, dtype=torch.float32)
        global_state[k_b] = torch.randn(7, 2, dtype=torch.float32)
        for s in states:
            s[k_a] = global_state[k_a] + 0.03 * torch.randn(2, 5)
            s[k_b] = global_state[k_b] + 0.03 * torch.randn(7, 2)
    gram, _ = effective_lora_gram(states, global_state, selected_module_count=4)
    explicit = []
    for s in states:
        diff = []
        for i in range(4):
            prefix = f'model.layers.{i}.self_attn.q_proj'
            key_a = prefix + '.lora_A.weight'
            key_b = prefix + '.lora_B.weight'
            m = s[key_b] @ s[key_a] - global_state[key_b] @ global_state[key_a]
            diff.append(m.reshape(-1))
        explicit.append(torch.cat(diff).double())
    x = torch.stack(explicit)
    assert torch.allclose(gram, x @ x.T, atol=1e-6, rtol=1e-5)
