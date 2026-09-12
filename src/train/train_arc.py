"""Train Spark TRM on ARC (Phase 1) or a probe config (Phase 2)."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.arc.dataset import ArcPairDataset, build_puzzle_id_map, collate
from src.arc.io import load_arc_dir
from src.arc.split import choose_holdout, load_holdout, write_holdout
from src.spark.config import SparkConfig, TrainConfig, dump_config, merge_configs
from src.spark.ema import EMA
from src.spark.losses import act_loss
from src.spark.recursive import SparkTRM, count_parameters


def pick_device(name: str) -> torch.device:
    if name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    if name == "cuda":
        print("CUDA not available, using CPU")
    return torch.device("cpu")


def split_tasks(data_dir: Path, train_cfg: TrainConfig, holdout_path: Path):
    train_tasks = load_arc_dir(data_dir / "training", split="training")
    if not train_tasks:
        train_tasks = load_arc_dir(data_dir / "data" / "training", split="training")
    eval_tasks = load_arc_dir(data_dir / "evaluation", split="evaluation")
    if not eval_tasks:
        eval_tasks = load_arc_dir(data_dir / "data" / "evaluation", split="evaluation")
    if holdout_path.exists():
        holdout_ids = load_holdout(holdout_path)
    else:
        holdout_ids = set(choose_holdout([t.task_id for t in eval_tasks], train_cfg.holdout_fraction))
        write_holdout(holdout_path, sorted(holdout_ids))
    holdout = [t for t in eval_tasks if t.task_id in holdout_ids]
    eval_rest = [t for t in eval_tasks if t.task_id not in holdout_ids]
    fit = list(train_tasks)
    if train_cfg.train_on_eval_demos:
        fit.extend(eval_rest)
    return fit, holdout, holdout_ids


@torch.no_grad()
def evaluate(model: SparkTRM, loader: DataLoader, device: torch.device, train_cfg: TrainConfig) -> dict:
    model.eval()
    totals = {"loss": 0.0, "exact": 0.0, "cell_acc": 0.0, "n": 0}
    for batch in loader:
        tokens = batch["tokens"].to(device)
        labels = batch["labels"].to(device)
        puzzle = batch["puzzle_id"].to(device)
        carry = model.initial_carry(tokens.size(0), device)
        logits = halt = None
        for _ in range(model.config.halt_max_steps):
            carry, logits, halt = model(tokens, puzzle, carry)
            if bool(carry.halted.all()):
                break
        loss, metrics = act_loss(logits, labels, halt, kind=train_cfg.loss_type)
        b = tokens.size(0)
        totals["n"] += b
        totals["loss"] += metrics["loss"] * b
        totals["exact"] += metrics["exact"] * b
        totals["cell_acc"] += metrics["cell_acc"] * b
    n = max(totals["n"], 1)
    return {k: (totals[k] / n if k != "n" else totals[k]) for k in totals}


def save_ckpt(path: Path, model: SparkTRM, ema: EMA | None, step: int, extra: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "step": step,
        "model": model.state_dict(),
        "ema": None if ema is None else ema.state_dict(),
        "extra": extra,
    }
    torch.save(payload, path)


def train(arch: SparkConfig, train_cfg: TrainConfig, *, synthetic: bool = False) -> None:
    device = pick_device(train_cfg.device)
    out_dir = Path(train_cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    dump_config(arch, train_cfg, out_dir / "config.yaml")

    if synthetic:
        from src.arc.synthetic import make_overfit_tasks

        tasks = make_overfit_tasks()
        holdout = tasks
        fit = tasks
        holdout_ids = {t.task_id for t in tasks}
    else:
        fit, holdout, holdout_ids = split_tasks(Path(train_cfg.data_dir), train_cfg, Path("data/holdout_eval_v1.json"))

    id_map = build_puzzle_id_map(fit + holdout)
    arch.num_puzzle_identifiers = max(arch.num_puzzle_identifiers, max(id_map.values()) + 1)

    model = SparkTRM(arch).to(device)
    counts = count_parameters(model)
    print(f"device={device}  core={counts['millions_core']}M  total={counts['millions_total']}M  n_train_pairs={sum(len(t.train) for t in fit)}")
    if counts["millions_core"] < 5 or counts["millions_core"] > 16:
        print("warning: core params outside the 7–15M target band (ok for debug configs)")

    train_ds = ArcPairDataset(fit, id_map, augment=not synthetic, translational_aug=train_cfg.translational_aug, seed=train_cfg.seed)
    eval_ds = ArcPairDataset(holdout, id_map, augment=False, translational_aug=False, seed=0)
    train_loader = DataLoader(train_ds, batch_size=train_cfg.batch_size, shuffle=True, collate_fn=collate, drop_last=False)
    eval_loader = DataLoader(eval_ds, batch_size=train_cfg.batch_size, shuffle=False, collate_fn=collate)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "puzzle_emb" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    groups = [{"params": decay, "lr": train_cfg.lr, "weight_decay": train_cfg.weight_decay}]
    if no_decay:
        groups.append(
            {"params": no_decay, "lr": train_cfg.puzzle_emb_lr, "weight_decay": train_cfg.puzzle_emb_weight_decay}
        )
    opt = AdamW(groups, betas=train_cfg.betas)
    ema = EMA(model, train_cfg.ema_decay) if train_cfg.ema else None

    step = 0
    t0 = time.time()
    log_path = out_dir / "train.jsonl"
    model.train()
    while step < train_cfg.max_steps:
        for batch in train_loader:
            tokens = batch["tokens"].to(device)
            labels = batch["labels"].to(device)
            puzzle = batch["puzzle_id"].to(device)
            carry = model.initial_carry(tokens.size(0), device)
            step += 1
            warm = max(train_cfg.warmup_steps, 1)
            scale = min(1.0, step / warm)
            for i, group in enumerate(opt.param_groups):
                base = train_cfg.puzzle_emb_lr if i == 1 else train_cfg.lr
                group["lr"] = base * scale
            last_metrics = {}
            for _ in range(arch.halt_max_steps):
                model.train()
                carry, logits, halt = model(tokens, puzzle, carry)
                loss, last_metrics = act_loss(logits, labels, halt, kind=train_cfg.loss_type)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                opt.step()
                if ema is not None:
                    ema.update(model)
                if bool(carry.halted.all()):
                    break
            if step % train_cfg.log_every == 0:
                elapsed = time.time() - t0
                row = {"step": step, "secs": round(elapsed, 1), **last_metrics}
                print(
                    f"step {step:6d}  loss={last_metrics['loss']:.4f}  "
                    f"exact={last_metrics['exact']:.3f}  cell={last_metrics['cell_acc']:.3f}"
                )
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(row) + "\n")
            if step % train_cfg.eval_every == 0:
                eval_model = ema.shadow if ema is not None else model
                metrics = evaluate(eval_model, eval_loader, device, train_cfg)
                print(f"eval step {step}  exact={metrics['exact']:.3f}  cell={metrics['cell_acc']:.3f}  loss={metrics['loss']:.4f}")
                with open(log_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps({"step": step, "split": "eval", **metrics}) + "\n")
                model.train()
            if step % train_cfg.save_every == 0:
                save_ckpt(out_dir / f"step_{step}.pt", model, ema, step, {"params": counts, "holdout": sorted(holdout_ids)})
            if step >= train_cfg.max_steps:
                break
    save_ckpt(out_dir / "last.pt", model, ema, step, {"params": counts, "holdout": sorted(holdout_ids)})
    print(f"done. checkpoints in {out_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Spark TRM (Phase 1/2)")
    p.add_argument("--arch", default="configs/spark7.yaml")
    p.add_argument("--train", default="configs/train_arc.yaml")
    p.add_argument("--synthetic", action="store_true", help="overfit the 4 synthetic tasks")
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--device", default=None)
    p.add_argument("--out-dir", default=None)
    p.add_argument("--data-dir", default=None)
    p.add_argument("--halt-max-steps", type=int, default=None)
    p.add_argument("--L-cycles", type=int, default=None)
    p.add_argument("--H-cycles", type=int, default=None)
    p.add_argument("--L-layers", type=int, default=None)
    p.add_argument("--pos", default=None, help="rope | rope_2d | learned | none")
    p.add_argument("--mlp-t", action="store_true")
    p.add_argument("--no-puzzle-emb", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    arch, train_cfg = merge_configs(args.arch, args.train)
    if args.max_steps is not None:
        train_cfg.max_steps = args.max_steps
    if args.batch_size is not None:
        train_cfg.batch_size = args.batch_size
    if args.device is not None:
        train_cfg.device = args.device
    if args.out_dir is not None:
        train_cfg.out_dir = args.out_dir
    if args.data_dir is not None:
        train_cfg.data_dir = args.data_dir
    if args.halt_max_steps is not None:
        arch.halt_max_steps = args.halt_max_steps
    if args.L_cycles is not None:
        arch.L_cycles = args.L_cycles
    if args.H_cycles is not None:
        arch.H_cycles = args.H_cycles
    if args.L_layers is not None:
        arch.L_layers = args.L_layers
    if args.pos is not None:
        arch.pos_encodings = args.pos
    if args.mlp_t:
        arch.mlp_t = True
    if args.no_puzzle_emb:
        arch.use_puzzle_emb = False
        arch.puzzle_emb_len = 0
    train(arch, train_cfg, synthetic=args.synthetic)


if __name__ == "__main__":
    main()
