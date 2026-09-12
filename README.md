# Spark — tiny recursive reasoner (7–15M)

Phase 1–2 of [PLAN.md](PLAN.md): a TRM-style model in the **7–15M** band for ARC-AGI.

## Setup

```bash
pip install -r requirements.txt
python scripts/download_arc.py
python scripts/count_params.py
```

## Colab

Open [`notebooks/spark_colab.ipynb`](notebooks/spark_colab.ipynb) → Runtime → GPU.

If this repo on GitHub does not yet contain `src/`, set **CODE_SOURCE** to `upload_zip` and upload a zip of the local folder (must include `src/` and `configs/`). Start with **RUN_MODE = overfit**.

## Phase 1 — reproduce TRM

Overfit sanity (CPU, tiny debug net):

```bash
python -m src.train.train_arc --arch configs/spark_debug.yaml --synthetic --max-steps 200 --device cpu --out-dir artifacts/overfit --batch-size 4
```

Full 7M run (needs a GPU):

```bash
python -m src.train.train_arc --arch configs/spark7.yaml --train configs/train_arc.yaml --out-dir artifacts/spark7
```

`spark-15` is the same recipe, wider hidden size, still inside 15M:

```bash
python -m src.train.train_arc --arch configs/spark15.yaml --out-dir artifacts/spark15
```

## Phase 2 — one change per run

Configs in `configs/probes/`:

| Config | Probe |
|---|---|
| `rope_2d.yaml` | axial 2D RoPE |
| `layers4.yaml` | 4 layers, `n=3` |
| `n2.yaml` / `n12.yaml` | inner recursion depth |
| `mlp_mixer.yaml` | attention-free mixer |
| `no_halt.yaml` | always 16 outer steps |
| `no_puzzle_emb.yaml` | drop identity embeddings |

```bash
python -m src.train.train_arc --arch configs/probes/rope_2d.yaml --out-dir artifacts/probe_rope2d
```

## Tests

```bash
pytest -q
```

Gate **G0** is ≥ 40% ARC-AGI-1 public eval (or within 5pp of a public TRM checkpoint) with `spark-7`. Do not grow the model until that lands. Later phases (DSL, 100M LM) are parked in PLAN.md.
