# BeliefNet

*Sound, cheap, auditable detection of emergent joint inconsistency in
language-model assertions.*

> BeliefNet extracts a typed constraint graph from a set of statements and
> decides joint satisfiability with a **sound exact checker** that returns a
> minimal unsat-core certificate and a minimal repair. An optional six-parameter
> differentiable equilibrium core is a trainable variant.

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue)](pyproject.toml)
[![Tests](https://img.shields.io/badge/tests-124%20passing-brightgreen)](tests/)

## Emergent joint inconsistency (real LLM + SciFact + FEVER results)

BeliefNet studies **emergent joint inconsistency** — contradictions with *no
pairwise witness* (e.g. an ordering cycle `A<B<C<A`) — with a **sound exact
checker** that returns proofs. All numbers below are real (live Azure GPT-4o /
GPT-5; real SciFact + FEVER; a Claude Haiku proxy reviewer):

- **Separation theorem** — pairwise judges provably miss emergent (`k>=3`)
  cases; an all-pairs judge drops AUROC `1.0 -> 0.5`.
- **A frontier reasoning model's blind spot (headline)** — on *numeric* emergent
  inconsistency (many-term global sums), **GPT-5 collapses AUROC `1.0 -> 0.233`
  (below chance)** as the term count grows, while the sound checker stays
  `1.000`. GPT-4o is at chance throughout.
- **GPT-4o fails** as hop-count grows (AUROC `0.976 -> 0.587`); **GPT-5 holds**
  `1.000` on *ordering* cycles even at 13-hop in 50 paraphrased statements
  (which is why the numeric result above isolates the real failure).
- **Cost + auditability** — the checker decides in **0.92 ms** vs GPT-5's
  5,691 ms (**~6193×**) at equal accuracy, and returns a **certificate + repair**.
- **Self-fix** — routing GPT-4o's local extractions through the checker lifts
  AUROC **+0.40** at hop 7 (decompose-and-verify).
- **Real data (SciFact + FEVER)** — LLM judges strong (SciFact GPT-5 0.989 /
  GPT-4o 0.960; FEVER ~0.98); the base schema is chance (honest limitation), and
  a polarity-triple extension restores certified detection (SciFact 0.84, FEVER
  0.89 with GPT-5) — the pattern replicates across two real corpora.
- **Auditability (proxy human study)** — giving a Claude Haiku reviewer the
  checker's certificate lifts verification accuracy from **0.50 (chance) to
  1.000** on hard depth-40 numeric cases, where a fluent LLM rationale gives no
  help — auditability measured, not asserted.

Full paper draft: [`paper.md`](paper.md). Protocol + all results:
[`results/experiments.md`](results/experiments.md) (machine-readable aggregate in
[`results/details_results.md`](results/details_results.md)).

## Optional trainable variant (equilibrium core)

A six-parameter differentiable `StructuredEquilibriumCore` (1 hinge margin,
4 readout weights, 1 bias) is an optional trainable reasoner. It ties the exact
checker on separability (exp5: exact = energy = 1.000) and, trained on the
synthetic benchmark, generalizes to **0.832 AUROC on held-out paraphrase** while
the best text baseline is at chance (0.500) — evidence the energy layer is
trainable and shift-robust. It is justified by that trainability, not by beating
the sound checker. Details in [`paper.md`](paper.md) §8.

## Pipeline

```
statements
  -> BeliefParser (perception)    text -> typed constraints (rule or LLM, per-statement)
  -> graph builder                typed constraints: anchor / before / greater / sum_eq
  -> exact checker (sound)        is_satisfiable(constraints)
       + certificates             minimal unsat core + minimal repair
     [or equilibrium core]        optional differentiable residual score
  -> decision (+ certificate)
```

The exact checker decides the order fragment by cycle detection and the linear
fragment (`anchor` / `sum_eq`) by exact Gaussian elimination over the rationals,
so a positive verdict is a *proof*, not a score. Method details (including the
optional equilibrium energy) are in [`paper.md`](paper.md).

## Install

```bash
git clone <repo-url> beliefnet && cd beliefnet
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
```

Targets local TITAN RTX (Turing, sm_75): fp16 compute, SDPA attention, 4-bit
NF4 quantisation. The 0.5B backbone runs on CPU as well for inspection.

## Quickstart

```bash
# unit tests (no GPU / model download)
python -m pytest -q

# headline experiments (real LLM calls; needs Azure creds in .env; set PYTHONPATH=src)
python scripts/run_numeric_stress.py --depths 5 12 20 28 36 --n-per-cell 15 --tag full --models gpt4o gpt5
python scripts/run_failure_curve.py --models gpt4o gpt5
python scripts/run_scifact.py --models gpt4o gpt5
python scripts/run_fever.py --models gpt4o gpt5 --n-per-class 40
python scripts/build_report.py            # refresh results/details_results.md

# optional: train the differentiable equilibrium variant on the synthetic benchmark
python scripts/train_synthetic.py \
    --model-config configs/model/paraphrase_robust.yaml \
    --data-config  configs/data/synthetic_augmented.yaml \
    --train-config configs/train/synthetic_equilibrium.yaml \
    --eval-config  configs/eval/in_dist.yaml
```

## Layout

```
src/beliefnet/
  encoders/      frozen LLM loader (FrozenBackbone, pool_hidden)
  perception/    BeliefParser family: type heads, span heads, numeric extractors
  graph/         typed constraint schema, graph builder, batch packer
  equilibrium/   energy functions, unrolled-GD solver, residual readout
  models/        StructuredEquilibriumCore + StatementEnergyBaseline (v1, archived)
  data/          synthetic generators + SciFact loader + FEVER loader skeleton
  eval/          AUROC / F1 / ECE metrics + matched-AUROC harness
  utils/         seeding, YAML config loader

configs/         model/data/train/eval YAMLs (no Hydra dependency)
scripts/         experiment runners (run_*.py) + build_report.py + train_synthetic.py
tests/           124 unit tests (no GPU required)
results/         per-experiment result.json + experiments.md + details_results.md
reports/synthetic/legacy_steps/step{02..12}/   legacy EEN trainer scripts (archived)
```

## Status

- **Emergent-inconsistency experiments**: complete and real — numeric-stress
  headline, ordering failure curve, separation theorem, cost, self-fix,
  certificates, online monitor, and the proxy human study (all in
  [`results/experiments.md`](results/experiments.md)).
- **Real data**: SciFact + FEVER zero-shot done (LLM judges ~0.97–0.99; the
  polarity-extension checker is certified).
- **Optional equilibrium variant**: trained on the synthetic benchmark; ties the
  exact checker and reaches 0.832 AUROC on held-out paraphrase.

## Versions and renames

This repository was originally developed under the working name **EEN**
(Epistemic Equilibrium Networks); public BeliefNet component names replace the
internal EEN-vN module names throughout.

## Citation

Working title and abstract are in preparation. For now please cite the
repository directly.
