import torch

from src.arc.dataset import ArcPairDataset, build_puzzle_id_map, collate
from src.arc.synthetic import make_overfit_tasks
from src.spark.config import merge_configs
from src.spark.losses import act_loss
from src.spark.recursive import SparkTRM


def test_overfit_synthetic_few_steps():
    arch, _ = merge_configs("configs/spark_debug.yaml")
    tasks = make_overfit_tasks()
    id_map = build_puzzle_id_map(tasks)
    arch.num_puzzle_identifiers = max(id_map.values()) + 1
    ds = ArcPairDataset(tasks, id_map, augment=False, translational_aug=False, seed=0)
    model = SparkTRM(arch)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    batch = collate([ds[i] for i in range(len(ds))])
    first = None
    last = None
    for _ in range(15):
        carry = model.initial_carry(batch["tokens"].size(0), batch["tokens"].device)
        carry, logits, halt = model(batch["tokens"], batch["puzzle_id"], carry)
        loss, metrics = act_loss(logits, batch["labels"], halt, kind="softmax_cross_entropy")
        if first is None:
            first = metrics["loss"]
        last = metrics["loss"]
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert last < first, (first, last)
