# Tiny Reasoning Model — Research and Build Plan

**Goal:** train a **7M–15M** TRM-style recursive grid model that is competitive on **ARC-AGI-1** and makes material progress on **ARC-AGI-2**.

**Active scope:** **Phase 1 + Phase 2 only.** Reproduce a ~7M TRM, then run one-at-a-time architecture probes still inside 15M. A 50–100M language model, DSL search, and Kaggle packaging are parked until G0/G1.

**Status:** living plan. Update the decision log when a gate passes, fails, or changes the design.

**Default hardware assumption:** 1 consumer/pro GPU (12–24 GB). The 7M model is the default; 15M is a width probe, not a new project.

---

## 0. Thesis

A 7M–15M model **cannot** beat frontier LLMs by memorizing the internet. It **can** beat them on ARC-style tasks if it has the right inductive bias and spends compute at test time.

The evidence:

- **TRM (7M)** reports **45%** ARC-AGI-1 and **8%** ARC-AGI-2 — better than many much larger LLMs — by recursing a tiny network over a latent + answer state, with heavy augmentation and deep supervision.
- **HRM (27M)** reports ~40% ARC-AGI-1. Independent ARC Prize verification: **32%** ARC-AGI-1, **2%** ARC-AGI-2 on semi-private. Ablations show the **outer refinement loop**, **augmentations**, and **per-task adaptation** matter more than the “hierarchical brain” story.
- **NVARC (2025 Kaggle 1st)** reached **24%** ARC-AGI-2 private by combining synthetic data, test-time training, and a TRM-style component — not by scaling an LLM.
- **BARC** showed **induction (write a program)** and **transduction (directly predict the grid)** solve different tasks. Ensembling them is the strongest published recipe on ARC-AGI-1 public eval (~55–62%).

So the plan is **not** “train a tiny ChatGPT.” For Phase 1–2 it is:

1. A **recursive grid reasoner** (transduction) in the TRM family, **7M baseline, 15M max**.
2. **Training-time augmentations** (D4 + color perm, start at 300).
3. **Architecture probes** that change one knob at a time (Phase 2).

Test-time voting, TTT, a DSL, and a language model wait until G0 is green.

ARC-AGI-2 at **85%** (the prize line) is a moonshot. Nobody is close with a small model. This plan still aims at it, but only after staged gates. Declaring 85% ARC-AGI-2 as Phase 1 is how the project dies.

---

## 1. What “beat” means

“Beat ARC-AGI” is ambiguous. Use these definitions and never mix them.

| Label | Metric | What it actually is |
|---|---|---|
| Public eval | pass@2 on public evaluation tasks | Useful for iteration. Easy to overfit. |
| Semi-private | ARC Prize holdout | The real scientific score. |
| Private / Kaggle | Hidden competition set | The prize score. Offline, compute-capped. |
| Human panel (best) | ~98% ARC-1, 100% ARC-2 (task solvable by ≥2 people) | Not a model target. |
| Average human | ~64% ARC-1, ~60% ARC-2 | A meaningful “human-like” bar. |
| Prize line | **85%** on private | Grand-prize threshold. Unlocked by nobody on ARC-AGI-2 as of 2026. |

### Current landscape (approximate, 2025–2026)

| System | Size | ARC-AGI-1 | ARC-AGI-2 | Notes |
|---|---|---|---|---|
| TRM (paper) | 7M | 45% (public/test as reported) | 8% | Recursive 2-layer net |
| HRM (paper / verified) | 27M | 41% / **32% semi-private** | 5% / **2% semi-private** | Outer loop + TTT-like training |
| CompressARC | 76K | ~20–34% eval | ~4% | Per-puzzle MDL, no pretraining |
| TinyLM + TTT | ~20M | 21.7% eval (paper) | — | Decoder-only, large train/eval gap |
| NVARC 2025 | ensemble | — | **24.03% private** | Synthetic data + TTT + TRM |
| ARChitects 2025 | larger LM | — | 16.53% private | 2D diffusion LM + refinement |
| Inkling Small (xhigh) | not tiny | **84.0%** | **40.1% semi-private** | Open-weight SOTA, expensive TTC |
| Average human | — | 64% | 60% | |
| Prize target | — | 85% (old) | **85%** | |

TRM/HRM numbers on public eval are **not** the same as semi-private. Always re-score on a held-out split we never train on.

### Project scoreboard (use this, nothing fuzzier)

| Gate | ARC-AGI-1 public eval | ARC-AGI-2 public eval | Meaning |
|---|---|---|---|
| **G0 Reproduce** | ≥ 40% | ≥ 6% | TRM baseline reproduced |
| **G1 Competitive tiny** | ≥ 50% | ≥ 10% | Better than vanilla TRM |
| **G2 2025-Kaggle class** | ≥ 58% | ≥ 18% | In NVARC/BARC territory |
| **G3 Human-average** | ≥ 64% | ≥ 25% | First “this is real” moment |
| **G4 Stretch** | ≥ 75% | ≥ 40% | Matches Inkling Small on ARC-2, with 100× fewer params |
| **Moonshot** | ≥ 85% | ≥ 85% | Prize. Do not plan the calendar around this. |

General-reasoning gates live in §8. They are **separate**. A model that memorizes ARC augmentations and cannot add 27+15 has failed half the project.

---

## 2. Why 7M–15M is the right band (for now)

**Below ~7M.** TRM already sits here. Going smaller (CompressARC at 76K) works only as per-puzzle optimization, not as a reusable reasoner.

**7M.** The published TRM-Att number. This is Phase 1. If we cannot reproduce it, nothing wider will save us.

**~14–15M.** Same 2-layer recursion, wider hidden size (`spark-15`). Only after G0.

**Above 15M.** Parked. Extra width overfits ARC’s tiny real set unless the data engine is huge, and it lengthens every probe.

**Working sizes**

| Name | Params (core) | Role |
|---|---|---|
| `spark-7` | ~7M | Phase 1 TRM clone. Default. |
| `spark-15` | ~14M | Width probe, still 2 layers. |
| `spark-debug` | ~0.5M | CPU overfit / unit tests only |

Do not start a 20M+ or 100M LM. `spark-7` first.

---

## 3. Hard constraints (read before designing)

1. **ARC is few-shot abstraction, not next-token trivia.** Every evaluation task is a new rule. Pretraining on the public train set is allowed as a prior, not as the answer.
2. **Exact grid match.** One wrong cell is a full miss. pass@2 (two attempts per test input).
3. **No internet at Kaggle eval.** Everything must ship in a notebook. Weights, DSL interpreter, augmentations, TTT.
4. **Compute cap.** 2025 was 4×L4, 12 hours, ~240 tasks. 2026 limits TBD; design as if the 2025 cap still exists.
5. **HRM’s lesson:** architecture < outer loop < data augmentation < per-task adaptation. Do not romanticize hierarchy.
6. **TRM’s lesson:** a 2-layer net with recursion can beat a deep net on these tasks because **effective depth** comes from looping, not stacking.
7. **BARC’s lesson:** induction and transduction are complementary. Build both.
8. **General reasoning is parked.** Phase 1–2 is the ARC transducer only. A 100M LM is not in this cycle.

---

## 4. System design

Three modules, one submission.

```
                    ┌─────────────────────────────┐
   task demos  ──►  │  A. spark-7 / spark-15      │ ──► grid
                    │  recursive latent + answer  │
                    └─────────────────────────────┘

Modules B (DSL) and C (TTT / voting) are parked until G0.
```

### Module A — Transductive recursive reasoner (`spark-7` / `spark-15`)

This is the TRM-class model. Official TRM trains on **single input→output grids** (30×30 packed), not a packed few-shot prompt. Task identity is a short puzzle embedding. That is Phase 1.

**Core idea (from TRM):**

- Keep two states: answer `y` and latent `z`.
- For up to `N_sup` outer steps:
  - Recurse `n` times: `z ← net(x, y, z)`
  - Then `y ← net(y, z)`
  - Supervise `y` against the true output every outer step (deep supervision)
  - Halt early if a learned stop head fires (training only; eval uses max steps)

**Architecture defaults**

| | `spark-7` | `spark-15` |
|---|---|---|
| Core parameters | ~7M | ~14M |
| Layers in the shared net | 2 | 2 |
| Hidden size | 512 | 768 |
| Heads | 8 | 12 |
| Recursion `n` (`L_cycles`) | 6 | 6 |
| Outer cycles `T` (`H_cycles`) | 3 | 3 |
| Outer steps `N_sup` | 16 | 16 |
| Grid encoding | 30×30, PAD=0, EOS=1, colors 2–11 | same |
| Attention | full on 900 cells | same |
| Norm / MLP | RMSNorm, SwiGLU, no bias | same |
| Position | 1D RoPE (Phase 2: 2D) | same |
| Equivariance | color perm + D4 as data aug | same |

Config files: `configs/spark7.yaml`, `configs/spark15.yaml`.

**Inductive biases to add after G0 (Phase 2, one at a time)**

1. **2D RoPE** — `configs/probes/rope_2d.yaml`
2. **2 vs 4 layers** — `configs/probes/layers4.yaml`
3. **Inner `n` = 2, 6, 12** — `n2.yaml`, `n12.yaml`
4. **MLP-mixer vs attention** — `mlp_mixer.yaml`
5. **Halt head vs fixed 16** — `no_halt.yaml`
6. **Puzzle-identity embedding on/off** — `no_puzzle_emb.yaml`

Color permutation is already the default training aug (Phase 1). Object tokens are **not** in this cycle.

**What not to do**

- Do not replace recursion with 24 transformer layers. Depth belongs in the loop.
- Do not train it as a chatbot. No BPE on ARC grids.
- Do not backprop through all 16 outer steps if VRAM dies; TRM-style detach between outer steps is fine. Prefer full BPTT through the inner `n` steps (TRM’s improvement over HRM’s 1-step IFT hack).

### Module B / C — parked

DSL induction, `spark-100`, TTT, and majority-vote ensembles are **out of scope** until Phase 1 G0 passes. The original notes remain below as a later-cycle sketch; do not implement them now.

Transduction fails on tasks that need an explicit rule (“count the unique colors, draw that many”). Induction fails on messy visual tasks. We want both.

**DSL (v1, keep it tiny)**

A functional language over grids, not unrestricted Python:

- Grid ops: `crop`, `pad`, `rotate`, `flip`, `transpose`, `tile`, `scale`, `overlay`
- Object ops: `cc` (connected components, 4- and 8-conn), `bbox`, `recolor`, `move`, `gravity`
- Logic: `filter`, `map`, `where`, `argmax`, `count`, `unique`
- Numbers: `+ − × // %`, comparisons
- Control: `if`, bounded `repeat` (not Turing-complete loops)

Interpreter must be:

- Deterministic
- Time-bounded (ms per program)
- Shape-safe (bad programs return `FAIL`, not crash)

**Search**

1. `spark-100` proposes programs in the DSL (or a Python subset that compiles to the DSL).
2. Execute on all demo pairs. Keep programs that fit every demo exactly.
3. Rank survivors by: demo fit, description length (MDL), and transducer agreement.
4. Apply the best 1–2 programs to the test input.

If no program fits, fall back to Module A.

**`spark-100` LM sketch**

| | `spark-100` |
|---|---|
| Parameters | 90–100M |
| Layers | 12 |
| Hidden | 768 |
| Heads | 12 (GQA 12q/4kv is optional) |
| Vocab | 16k–32k SentencePiece + DSL tokens |
| Context | 2048 (4096 later if TTT needs it) |
| Position | RoPE |
| Norm | RMSNorm + QK-norm |
| MLP | SwiGLU, tied embeddings |

This model is also the **general reasoner**. Same weights, two SFT mixtures: DSL programs and math/logic CoT.

### Module C — Test-time refinement (this is where most points come from)

For each eval task, in order of cheap → expensive:

1. **Dihedral + color augmentations** of the demos. Run Module A on each. Invert the transform. Majority vote.
2. **Test-time training (TTT):** few-shot gradient steps on the task’s demo pairs (and their augmentations) starting from the pretrained checkpoint. Reset weights after the task.
3. **Program search** (Module B) in parallel, budgeted.
4. **Cross-check:** if A and B agree, emit that as attempt 1. Use the next-best distinct grid as attempt 2.
5. **Halt / budget manager:** stop when remaining wall-clock per remaining tasks is tight. Never spend 12 hours on the first 20 puzzles.

Start with (1). Add (2) only after G1. Add (3) after the DSL interpreter is trustworthy.

---

## 5. Data plan

Data quality is the project. Architecture is the multiplier.

### 5.1 Official ARC (tiny, sacred)

| Set | Size | Use |
|---|---|---|
| ARC-AGI-1 training | 400 tasks | Priors + generators. Never treat as the test. |
| ARC-AGI-1 public eval | 400 | Dev score. Freeze a 100-task **never-train** slice as `eval-holdout-1`. |
| ARC-AGI-2 public training | (imported 400 from ARC-1 + extras as released) | Same as above. |
| ARC-AGI-2 public eval | as released | `eval-holdout-2`. |
| Semi-private / private | 120 + 120 | Do not touch. Only ARC Prize / Kaggle. |
| ConceptARC | systematic abstractions | Diagnostic, not a trophy. |
| Mini-ARC / 1D-ARC | smaller grids | Cheap unit tests for the transducer. |

**Contamination rule:** `eval-holdout-*` is hashed and excluded from every generator, mix, and TTT debug run. If a number is not on holdout, it is not a number.

### 5.2 Existing synthetic sources (use immediately)

| Source | What it is | How we use it |
|---|---|---|
| **RE-ARC** (Hodel) | Procedural generators for the 400 ARC-1 train tasks | Unlimited extra demos per known task; TTT fuel |
| **BARC seeds + remixes** | 162 human programs → ~400k new tasks with solutions | Induction SFT + transduction |
| **NVARC synthetic** | 103k puzzles + 3.2M augmented | Pretrain transducer |
| **Sudoku-Extreme, Maze-Hard** | TRM’s other puzzles | Recursion sanity, not ARC score |
| **ARC-Potpourri** | Mixed 400k | Transduction pretrain |

### 5.3 Our generator (must-build)

Hand-written generators beat LLM-remixed ones on **coverage of core knowledge**. Chollet’s priors, each with 10–30 parameterized families:

1. **Objectness** — connected blobs, occlusion, same-shape different-color, extract-the-unique-object
2. **Geometry / topology** — symmetry, rotation, holes, enclosure, path-connectedness, flood-fill
3. **Numbers / counting** — count objects, draw N, sort by size, modulo coloring
4. **Goal-directed motion** — gravity, bounce, “move until wall”, align to marker
5. **Copy / align / pattern completion** — tiling, fractal-ish repetition, inpainting from symmetry
6. **Relational** — same as, next to, inside, above, matching pairs
7. **Color logic** — remap, majority color, palette from a legend
8. **Composition** — apply two sampled rules in sequence (this is how ARC-2 hurts)

Each generator emits:

```text
task_id, rule_id, difficulty, grid_size, n_demos, demos[], test[], dsl_program?, nl_description?
```

**Targets**

| Milestone | Unique tasks | Notes |
|---|---|---|
| D0 | 50k | One family per prior, end-to-end pipeline |
| D1 | 500k | All priors, difficulty mix |
| D2 | 2M+ | Compositions of 2–3 rules (ARC-2 style) |
| D3 | 5M augmented views | Cheap D4 + color perms on D2 |

Filter garbage:

- Output equals input (unless the rule is identity)
- Degenerate (single color, empty)
- Inconsistent demos (generator bug)
- Test leak (test input seen in demos)
- Too easy (unique object, trivial crop) beyond a capped fraction

### 5.4 General-reasoning curriculum (`spark-100` only)

Small LMs reason only if the text is **simple, dense, and verifiable**.

| Bucket | Examples | Share (stage 1 → 3) |
|---|---|---|
| Clean web/edu | FineWeb-Edu (high-score slice), SimpleWiki | 50% → 15% |
| Synthetic textbooks | TinyStories-like, Phi-style “teach X” | 20% → 15% |
| Math | TinyMath, GSM8K, integer arithmetic, Countdown | 10% → 25% |
| Code | Tiny algorithms, DSL programs, Python snippets that execute | 10% → 25% |
| Logic | ProntoQA, ProofWriter, grid-logic in text, chess-mate-in-1 style | 5% → 10% |
| Instruction / CoT | Distilled traces from a 7B–32B teacher on **verifiable** problems | 5% → 10% |

**Teacher distillation rule:** only keep traces whose final answer checks with a verifier (calculator, interpreter, unit test). Unverified CoT will teach the 100M model to ramble.

---

## 6. Phases

Calendar is ~**4–8 weeks** for G0 + Phase 2 probes. Later phases stay on the page but are not this cycle.

```
Week   1  2  3  4  5  6  7  8
P0 eval + repo ████
P1 reproduce TRM  ████████
P2 arch probes          ████████
```

### Phase 0 — Foundations (weeks 1–2)

**Build**

- Repo layout in §10
- ARC JSON loaders, visualizer (grid → PNG), pass@2 scorer
- Frozen `eval-holdout-1` / `eval-holdout-2` splits
- Logging: wandb or plain JSONL (loss, exact-match, time/task)
- One dummy `spark-7` forward pass on a packed grid
- `python scripts/count_params.py` prints core params in band

**Gate G0a:** scoring a random baseline and an “copy the last demo’s output” baseline on holdout. Numbers recorded. If the scorer is wrong, everything after this is fiction.

### Phase 1 — Reproduce TRM (weeks 2–6)

**Build**

- Faithful TRM: 2-layer net, `y`/`z` states, inner recursion, deep supervision, halt head
- Official-style packing of ARC tasks
- Training-time augmentations: rotations, flips, color permutations (start at **300**, not 1000; HRM analysis says 300 is enough)
- Public RE-ARC extra demos

**Train:** `spark-7` (`configs/spark7.yaml`). Use `spark-debug` only for the CPU overfit test. Do **not** train `spark-15` until G0.

**How to run**

```bash
python scripts/download_arc.py
python -m src.train.train_arc --arch configs/spark_debug.yaml --synthetic --max-steps 200 --device cpu --out-dir artifacts/overfit
python -m src.train.train_arc --arch configs/spark7.yaml --train configs/train_arc.yaml --out-dir artifacts/spark7
```

**Gate G0:** ≥ 40% ARC-AGI-1 public eval **or** within 5pp of the public TRM checkpoint we can re-run. If we cannot reproduce, **stop scaling**. Debug packing, aug, supervision, and eval format.

**Kill criterion:** after two honest reproduction attempts, if holdout is < 20% while loss is tiny, we are leaking or scoring wrong.

### Phase 2 — Architecture probes (weeks 5–9)

Change **one** thing per run, **7M only** (use `spark-15` only if a probe clearly needs width):

| Probe | Config | Hypothesis |
|---|---|---|
| 2D RoPE vs 1D | `configs/probes/rope_2d.yaml` | Spatial tasks jump |
| 2 vs 4 layers | `configs/probes/layers4.yaml` | TRM says 2 generalizes better |
| Inner `n` = 2, 6, 12 | `n2.yaml`, `n12.yaml` | Diminishing returns vs time |
| MLP-mixer vs attention | `mlp_mixer.yaml` | Attention should win on 30×30 (~14M core) |
| Halt head vs fixed 16 | `no_halt.yaml` | Training speed; maybe worse accuracy |
| Puzzle embedding on/off | `no_puzzle_emb.yaml` | Identity conditioning is doing a lot |

```bash
python -m src.train.train_arc --arch configs/probes/rope_2d.yaml --out-dir artifacts/probe_rope2d
```

**Gate:** at least one probe is **+3pp** on holdout vs Phase 1 `spark-7`. Freeze that as `spark-7-v2`.

### Phase 3+ — parked

Data engine, TTT, induction, `spark-100`, and Kaggle packaging wait until Phase 1–2 gates pass. The old phase text is kept as a backlog, not a schedule.

**Build** the generator in §5.3. Ship D0 by week 4, D1 by week 8, D2 by week 14.

Also ingest RE-ARC, BARC, NVARC datasets behind a single `TaskDataset` interface.

**Diagnostics (not leaderboard):**

- Per-prior accuracy (object, count, symmetry, …)
- Difficulty buckets
- Composition depth 1 vs 2 vs 3
- Grid-size buckets (≤10, 11–20, 21–30)

**Gate D1:** 500k unique tasks, < 2% filter-fail, every prior has ≥ 10k examples.

### Phase 4 — Train the ARC specialist (weeks 8–14)

**Recipe for `spark-20-v2` then `spark-50`**

1. Pretrain on D1/D2 + RE-ARC + NVARC mixed, packed tasks, deep supervision.
2. Mid-train on harder compositions + official 400.
3. Short decay on official train + RE-ARC generators only (narrower distribution).

**Optimization (starting point, not dogma)**

- AdamW, β=(0.9, 0.95), wd=0.05
- LR 3e-4 → cosine to 1e-5, warmup 2–3%
- Batch: as many tasks as VRAM allows (aim ≥ 64 task-instances with aug)
- Precision: bf16
- Gradient clip 1.0
- EMA weights for eval

**Gate G1:** ≥ 50% ARC-AGI-1 holdout, ≥ 10% ARC-AGI-2 holdout, **single-pass** (no 1000-vote, no TTT). If G1 only holds with 1000-vote, the model is not actually better — the ensemble is.

### Phase 5 — Test-time compute (weeks 13–17)

Implement, in order:

1. Majority vote over K ∈ {8, 32, 128} augs
2. TTT: k-step Adam on demo augs, small LR, early-stop on demo exact-match
3. Budget allocator (hard time cap per task)

**Measure the TTC curve:** accuracy vs K vs wall-clock. TRM analyses saw ~+11pp from 1000-vote. We want **most of that at K≤32**.

**Gate G2 (partial):** +8pp over G1 from TTC alone on holdout, still inside a 12h / 240-task budget on one 24 GB GPU (scale later to 4×L4).

### Phase 6 — Induction (weeks 15–21)

1. Freeze DSL v1 + interpreter + unit tests (100 programs, including known ARC-1 train solutions).
2. SFT `spark-100` to emit DSL from demos (BARC seeds + our generators that have ground-truth programs).
3. At test time: sample N programs, execute, keep consistent ones, MDL-rank.
4. Ensemble with Module A (see §4).

**Gate:** induction-only solves a **disjoint** slice of holdout vs transduction-only. Union ≥ +5pp over A+TTC. If the union is not bigger, the DSL is too weak or the search is not finding programs.

### Phase 7 — General reasoner (weeks 6–20, parallel)

`spark-100` training ladder:

| Stage | Tokens (order of magnitude) | Data | Success looks like |
|---|---|---|---|
| 7a tokenizer | — | 10–50M mixed text + DSL | Encode/decode roundtrip on grids-as-text and GSM8K |
| 7b pretrain | 2–10B | FineWeb-Edu + synthetic textbooks + code | Loss ↓, not collapse; can complete simple English |
| 7c midtrain | 0.5–2B | Math, code, logic, ARC-as-text | Arithmetic exact-match ↑ |
| 7d SFT | 50–200M | Verified CoT + DSL programs | Reliable format: reason then answer |
| 7e RL (optional) | small | GRPO on verifiable math/DSL | Only if 7d format is stable |

Chinchilla is ~20 tokens/param (2B tokens at 100M). **Overtrain** past that; inference-cheap models should be data-heavy. 5–10B tokens is the real target if money allows.

**Gate (general):** see §8.2. If ARC is green and this is red, we still have a specialist. Keep going, but do not claim “reasoning model.”

### Phase 8 — System integration (weeks 20–24)

- One `solve(task) -> (grid_a, grid_b)` entry point
- Module C orchestration
- Kaggle notebook: no internet, cached weights, deterministic seeds
- Dry run: 50 tasks timed
- Failure taxonomy: wrong size, off-by-one object, color swap, identity, timeout

**Gate G2 full:** ARC-AGI-1 holdout ≥ 58% **and** ARC-AGI-2 holdout ≥ 18% in the timed setting.

### Phase 9 — Evidence (weeks 22–28)

- Ablations that would convince a skeptic: no-aug, no-TTT, no-induction, 7M vs 20M vs 50M, inner n, outer N
- Semi-private verification path (open source + ARC Prize process)
- Writeup: what failed, not just what scored

---

## 7. Training details worth freezing early

### Packing an ARC task into the transducer

For each task:

1. Sample a color permutation and a D4 transform (train only; at test, enumerate).
2. Apply to all demos and the test input (not the test output at eval).
3. Encode each grid as cells with `(row, col, color)` plus a separator token between pairs.
4. Targets: the output grid of the “query” pair (a held-out demo during train, the test at eval).
5. Auxiliary losses (optional, add only if they help holdout):
   - Demo reconstruction
   - Halt / continue
   - Object-count head
   - Output-shape head (H, W) — **do this**; wrong shape is a free way to score 0

### Exact-match training

Cross-entropy on cells is fine. Add a **whole-grid** bonus (or extra weight on the last outer step) so the model is not content with 99% cells correct.

### Evaluation protocol (non-negotiable)

- Same code path for every reported number
- pass@2 as in ARC Prize (two attempts)
- Report **single-pass** and **TTC** separately
- Report wall-clock and $ / task when possible
- Never fit hyperparameters on `eval-holdout`

---

## 8. Evaluation suite

### 8.1 ARC and grid

| Benchmark | Why |
|---|---|
| ARC-AGI-1 holdout | Main north star (tiny models) |
| ARC-AGI-2 holdout | True hardness |
| ConceptARC | Abstraction categories |
| Mini-ARC / 1D-ARC | Cheap regression |
| Sudoku-Extreme, Maze-Hard | Recursion regression (TRM paper tasks) |

### 8.2 General reasoning (size-adjusted, not vs GPT)

A 100M model will not win GSM8K against 7B. Score against **other small models** (SmolLM2-135M, GPT-2-124M, TinyStories-scale) and against **ourselves over time**.

| Benchmark | G1 (honest) | G2 (strong for 100M) | Stretch |
|---|---|---|---|
| Integer arithmetic (4-digit ±, 2-digit ×) | 80% | 95% | 99% |
| GSM8K (flexible match, CoT) | 8% | 20% | 35% |
| HumanEval / tiny Python (pass@5) | 5% | 12% | 20% |
| ProntoQA / simple deduction | 40% | 70% | 85% |
| Execute-our-DSL roundtrip | 70% | 90% | 95% |
| Instruction following (internal 100 prompts) | format-ok 80% | 95% | — |

If GSM8K is 0% after 7d SFT, the tokenizer/curriculum is broken. Fix that before GRPO.

### 8.3 Process metrics

- Exact-match vs cell-accuracy (cell-accuracy can look great while exact-match is dead)
- Shape error rate
- Time per task vs accuracy (Pareto)
- TTT steps until demo fit
- % tasks where A and B agree
- Collapse-to-identity rate

---

## 9. Compute and cost

Rough, so we do not plan a 70B run by accident.

| Work | Hardware | Time | Ballpark $ |
|---|---|---|---|
| P1 TRM 7M reproduce | 1× 12–24 GB | 2–5 days | $0–80 |
| P2 probes (6 runs) | 1× 12–24 GB | 2 weeks | $100–400 |

**FLOPs sanity:** Chinchilla 6ND. 100M params × 5B tokens × 6 ≈ 3e18 FLOP. On an H100 (~1e15 FLOP/s peak, ~40% MFU) that is on the order of **1–2 GPU-days**, plus overhead. The ARC recursive trainer is less efficient (small batches, 2D, outer loops) — budget **10×** that wall-clock for Module A.

**Priority if money is tight:** `spark-7` + 300 augs. Skip `spark-15` until G0.

---

## 10. Repo layout (target)

```text
reasoning-model/
  PLAN.md
  README.md
  configs/
    spark7.yaml
    spark15.yaml
    spark_debug.yaml
    train_arc.yaml
    probes/            # Phase 2, one change each
  src/
    arc/               # io, encode, augment, metrics, split, dataset
    spark/             # TRM: config, layers, recursive, losses, ema
    train/train_arc.py
  tests/
  scripts/             # download_arc, count_params, viz_task
  artifacts/
  data/
```

**Rule:** a new idea is a config in `configs/probes/` + a test, not a fork of `recursive.py`.

---

## 11. Risks and how we will notice

| Risk | Symptom | Response |
|---|---|---|
| Overfitting public eval | Holdout << public, or ConceptARC dead | Frozen holdout; stop peeking; more generators, less official replay |
| Architecture cosplay | Fancy hierarchy, same score as 2-layer TRM | Ablate to TRM every time |
| TTC theater | +10pp only at K=1000, blows Kaggle time | Pareto curve; cap K |
| Identity collapse | Outputs copy the test input | Explicit identity tasks in mix **and** a penalty; check TRM identity papers |
| DSL too weak | Search never finds a program | Add primitives from failed-task autopsy, not from imagination |
| DSL too strong / slow | Interpreter timeouts | Bound steps; profile; subset primitives |
| LM eats the project | 3 weeks on tokenizer, 0 ARC runs | Module A is the critical path until G1 |
| Unverified CoT | Fluent nonsense, 0% GSM8K | Verifier-only distillation |
| “We’ll TTT at the end” | Pretrain is garbage, TTT cannot save it | G1 is **single-pass** |
| Prize hallucination | Calendar built around 85% ARC-2 | Gates G0–G3 only |

---

## 12. Decision log

Record outcomes here. Do not keep decisions in chat.

| Date | Decision | Why | Result |
|---|---|---|---|
| 2026-09-11 | Dual system: recursive transducer + tiny LM/DSL, 20–100M | Matches what actually scores on ARC | **Superseded** |
| 2026-09-11 | **Lock to 7–15M. Phase 1–2 only.** | Shorter train/learn loop; TRM is already 7M | active |
| 2026-09-11 | Reproduce TRM before any original architecture | If we cannot hit ~40% ARC-1, we cannot beat it | — |
| 2026-09-11 | G1 measured single-pass | Avoid TTC-inflated claims | — |
| 2026-09-11 | 85% ARC-AGI-2 is a moonshot, not Phase 1 | SOTA small is ~8% | — |

---

## 13. First two weeks — concrete checklist

1. `pip install -r requirements.txt` and `python scripts/download_arc.py`.
2. `pytest -q` — encode roundtrip, dihedral inverse, param counts, synthetic overfit.
3. `python scripts/count_params.py` — `spark-7` core in 6–9M, `spark-15` in 12–16M.
4. Freeze holdout via the first train run (`data/holdout_eval_v1.json`).
5. `python scripts/viz_task.py` on a handful of training tasks.
6. CPU overfit: `--arch configs/spark_debug.yaml --synthetic`.
7. Overnight `spark-7` on official train (+ eval demos, not holdout, not test outputs).
8. Score holdout exact-match. Update this decision log.
9. Do **not** start `spark-15` or any LM.
10. First Phase 2 probe: `rope_2d` unless G0 failed.

When the overnight number exists, pick the next probe from the table in Phase 2.

---

## 14. References (starting set)

- Chollet, *On the Measure of Intelligence* (2019) — why ARC exists
- ARC Prize 2025 technical report — refinement loops, NVARC 24%, TRM paper award
- Jolicoeur-Martineau, *Less is More: Recursive Reasoning with Tiny Networks* (TRM)
- Wang et al., Hierarchical Reasoning Model (HRM)
- ARC Prize, *The Hidden Drivers of HRM’s Performance on ARC-AGI*
- Li, Ellis et al., *Combining Induction and Transduction* (BARC)
- Hodel, RE-ARC
- Liao & Gu, CompressARC / *ARC-AGI Without Pretraining*
- Akyürek et al., test-time training on ARC
- NVARC 2025 open-source writeup
- Hoffmann et al., Chinchilla
- Gunasekar et al., Phi-1 (textbooks are all you need)
- Eldan & Li, TinyStories

Code to actually read, not just cite:

- https://github.com/SamsungSAILMontreal/TinyRecursiveModels
- https://github.com/sapientinc/HRM
- https://github.com/michaelhodel/re-arc
- https://github.com/xu3kev/BARC
- https://github.com/1ytic/NVARC
- https://github.com/fchollet/ARC-AGI

---

## 15. Bottom line

**Stay at 7–15M.** TRM already reports 45%/8% at 7M. Phase 1 is reproduce that. Phase 2 is one-knob probes inside the same band.

**Do not** train a 100M GPT, a DSL, or TTT until G0 is a real number on the frozen holdout.

Next action: `pytest -q`, then the synthetic overfit, then an overnight `spark-7` run.
