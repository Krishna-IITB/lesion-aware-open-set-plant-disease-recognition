from pathlib import Path

import torch

from plant_ood.training.checkpoint import load_checkpoint, save_checkpoint


def test_checkpoint_round_trip(tmp_path: Path) -> None:
    model = torch.nn.Linear(3, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    path = tmp_path / "model.pt"
    original = model.weight.detach().clone()
    save_checkpoint(path, model, optimizer, epoch=4, metadata={"seed": 13})
    with torch.no_grad():
        model.weight.zero_()
    epoch, metadata = load_checkpoint(path, model, optimizer)
    assert epoch == 4
    assert metadata == {"seed": 13}
    assert torch.equal(model.weight, original)
