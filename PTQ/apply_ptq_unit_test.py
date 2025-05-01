import torch
import pytest
from types import SimpleNamespace

def is_ternary(tensor, alpha, tol=1e-5):
    # Checks if all weights are in {-alpha, 0, alpha} within tolerance
    vals = torch.unique(tensor)
    allowed = torch.tensor([-alpha, 0.0, alpha], device=tensor.device)
    return all(any(torch.isclose(v, a, atol=tol) for a in allowed) for v in vals)

def test_applyPTQ_ternary_weight_and_activation():
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = torch.nn.Linear(8, 8)

        def forward(self, x):
            return self.linear(x)

    model = DummyModel()
    tokenizer = SimpleNamespace()
    tokenizer.encode = lambda x, return_tensors=None: torch.randint(0, 100, (1, 8))

    quant_model = applyPTQ(
        model=model,
        tokenizer=tokenizer,
        calibration_input="Dummy calibration sentence.",
        act_quant=True,
        deterministic=True,
    )

    quantized_found = False
    for name, module in quant_model.named_modules():
        if isinstance(module, BitLinear):
            quantized_found = True
            # Test for ternary weights
            alpha = module.alpha.item()
            assert is_ternary(module.weight.data, alpha), f"{name} weights not ternary!"

            # Test for activation quant
            if module.act_quant:
                assert hasattr(module, 'act_scale'), f"{name} missing act_scale"
                assert module.act_scale is not None, f"{name} has None act_scale"
                assert module.act_scale.item() > 0, f"{name} act_scale not calibrated"

    assert quantized_found, "No BitLinear modules were found in model!"
