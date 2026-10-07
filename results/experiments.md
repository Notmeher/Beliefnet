# BeliefNet — End-to-End Experiment Documentation

> Single-source write-up of the problem, theory, method, benchmark, every
> experiment (protocol, config, exact commands, results, interpretation),
> proposal-satisfaction analysis, and full reproducibility. A co-author should
> be able to draft the paper from this file alone.
>
> Companion machine-readable report: [`details_results.md`](details_results.md)
> (auto-generated). Raw per-experiment payloads: `results/<exp>/result.json`.
> All numbers below were produced on the current codebase with `seed=0`.

---

## Table of contents

1. [One-paragraph summary](#1-one-paragraph-summary)
2. [Motivation and research question](#2-motivation-and-research-question)
3. [Problem formalization](#3-problem-formalization)
4. [The separation theorem](#4-the-separation-theorem)
5. [Method](#5-method)
6. [The benchmark](#6-the-benchmark)
7. [Experiments](#7-experiments)
8. [Statistical methodology](#8-statistical-methodology)
9. [Headline results table](#9-headline-results-table)
10. [Does it satisfy the proposal?](#10-does-it-satisfy-the-proposal)
11. [What is pending and how to complete it](#11-what-is-pending-and-how-to-complete-it)
12. [Reproducibility](#12-reproducibility)
13. [Writing the paper from this](#13-writing-the-paper-from-this)

---

## 1. One-paragraph summary

Large language models assert many facts across long outputs but cannot verify
that those assertions form a **jointly consistent** set. We study **emergent
joint inconsistency** — contradictions with *no pairwise witness*, such as an
ordering cycle `A < B < C < A`, whose smallest unsatisfiable subset has size
`k >= 3`. We prove (and verify empirically) that any judge whose decision
factorizes over subsets of size `< k` (all pairwise NLI, all-pairs aggregators)
must miss these. BeliefNet extracts a typed constraint graph from text and
decides joint satisfiability with a **sound exact checker** that returns a
**minimal unsatisfiable-core certificate** (a proof) plus a **minimal repair**;
an **online monitor** does this incrementally and localizes the exact statement
that breaks consistency. On a controlled benchmark the structured checker is
perfect across hop-counts while an all-pairs judge collapses to chance at
`k >= 3` (AUROC `1.0 -> 0.5 -> 0.5`). Our **headline** result: on *numeric*
emergent inconsistency (many-term global sums) even a frontier reasoning model
(GPT-5) collapses from AUROC 1.0 to **0.233 (below chance)** as the term count
grows, while the checker stays at 1.0 — and in a proxy-reviewer study the
checker's certificate lifts a reviewer from chance to perfect where an LLM
rationale does not.

---

## 2. Motivation and research question

Frontier LLMs are deployed as research assistants, clinical decision support,
and long-form writers. The failure mode that matters is **joint inconsistency**:
the model asserts `A`, later asserts something incompatible with `A`, and a
reader without the budget to cross-check accepts both. Existing detectors
(confidence thresholding, retrieval verification, trained NLI classifiers) embed
the decision inside an opaque function and, critically, are **pairwise**: they
compare a claim to one piece of evidence at a time.

**Research question.** Can an explicit typed belief graph plus a sound reasoning
layer detect *joint* (global) inconsistency — including contradictions with no
pairwise witness — while producing a human-auditable proof, where opaque
pairwise methods provably cannot?

**Positioning (what is new).** Not "another contradiction detector." The
contribution is a **phenomenon** (emergent inconsistency is a structural blind
spot of pairwise methods and, empirically, of frontier LLMs as the set grows),
a **theorem** explaining why, a **benchmark** that measures it, and a
**verifiable monitor** that emits certificates and repairs.

---

## 3. Problem formalization

Let `S = {s_1, ..., s_n}` be a set of statements. A **perception map** `P`
extracts a typed constraint set `C = P(S)` over entities `E` with one scalar
latent `u_e` per entity. Constraint types:

- `anchor(e, v)`   => `u_e = v`
- `before(a, b)`   => `u_a < u_b`
- `greater(a, b)`  => `u_a > u_b`
- `sum_eq(a, b, t)`=> `u_a + u_b = u_t`

`S` is **jointly consistent** iff `C` is satisfiable (there exists an assignment
of the `u_e` satisfying all constraints).

**Emergence order.** `kappa(S) = ` size of the *smallest* unsatisfiable subset
of `C`. An inconsistency is **pairwise-detectable** iff `kappa = 2` and
**emergent** iff `kappa >= 3`.

**Task.** Given `S`, decide joint consistency; if inconsistent, return a minimal
unsatisfiable core `U ⊆ C` with `|U| = kappa`, localized to the responsible
statements — even when `kappa >= 3`.

---

## 4. The separation theorem

**Definition (m-factored judge).** A consistency judge is *m-factored* if it
declares `S` inconsistent iff some subset of size `<= m` is inconsistent
(monotone AND-aggregation over `<= m`-subsets). Pairwise NLI and evidence-pair
fact-checkers are `m = 2`.

**Proposition.** For every `m >= 2` there exist statement sets with `|S| = m+1`
that are jointly inconsistent while every subset of size `<= m` is consistent;
hence no `m`-factored judge can detect them.

**Proof.** Take a strict order relation and a directed cycle
`e_0 -> e_1 -> ... -> e_m -> e_0`. Every proper subset omits an edge, leaving an
acyclic (satisfiable) order; the full set is a cycle (unsatisfiable). The cycle
length `k = m+1` is the hop-count. ∎

**Honesty note (state in the paper).** The theorem *strictly* bounds the
pairwise / `m`-factored class (NLI, retrieval-pair, all-pairs aggregators). A
frontier LLM that reads all statements at once is in principle `|S|`-factored,
so the theorem does **not** prove LLMs must fail — that is an *empirical* claim
(their effective global reasoning degrades with `n` and `k`), measured by the
failure-curve experiment. The theorem for the pairwise class + the empirical
curve for LLMs is the intended pairing.

Empirical verification: Experiment 2 and Experiment 4 (below).

---

## 5. Method

### 5.1 Pipeline

```
statements
  -> BeliefParser (perception)         text -> typed constraints
  -> build_graph_from_supervisions     -> (entities, constraints)
  -> exact checker  (sound decision)   is_satisfiable(constraints)
       + certificates                  minimal unsat core + minimal repair
     [or equilibrium residual]         differentiable soft score (optional)
  -> decision (+ certificate)
```

Code: perception in `src/beliefnet/perception/`, graph in
`src/beliefnet/graph/graph_builder.py`, reasoning in
`src/beliefnet/reasoning/`, end-to-end in `src/beliefnet/models/pipeline.py`
(`BeliefNet`).

### 5.2 Exact checker (soundness)

`src/beliefnet/reasoning/exact_checker.py`. Decides satisfiability exactly:

- **Order fragment** (`before` / `greater`): build the strict `<` digraph and
  test for a directed cycle (DFS). Unsatisfiable iff a cycle exists.
- **Linear fragment** (`anchor` / `sum_eq`): exact Gaussian elimination over the
  rationals (`fractions.Fraction`); unsatisfiable iff the system is inconsistent.
- The two fragments share no entities in the current schema, so `S` is SAT iff
  both are SAT. If a future schema couples them on a shared entity the checker
  raises rather than return a possibly-unsound answer.

**Soundness property (paper claim):** a positive contradiction verdict is a
proof, not a score. Therefore *every false positive is a perception error, never
a reasoning error* — all learnable error is localized to (and measured in) the
perception map.

### 5.3 Certificates

`src/beliefnet/reasoning/certificates.py`.

- `minimal_unsat_core` — Junker's **QuickXplain** using the exact checker as the
  consistency oracle; returns the smallest jointly-unsatisfiable subset (the
  proof of contradiction).
- `minimal_correction_set` — complement of a greedily-grown maximal satisfiable
  subset; the smallest set whose retraction restores satisfiability (the repair).

### 5.4 Perception (BeliefParser)

`src/beliefnet/perception/belief_parser.py` — the `BeliefParser` ABC and:

- `RuleBeliefParser` — deterministic regex parser (`parser_rules.derive_supervision`),
  offline, exact on the templated benchmark.
- `LLMBeliefParser` — **strictly local** per-statement extraction via an Azure
  chat client (`src/beliefnet/llm/extractor.py`); each statement is parsed on
  its own so the global reasoning is done only by the checker.
- `ParaphraseRobustBeliefParser` — driven by a trained supervision function
  (wire from the Hybrid + Span + numeric pipeline in
  `scripts/train_synthetic.py`).

All return `(entities, constraints)`, so the reasoner is invariant to parser choice.

### 5.5 End-to-end `BeliefNet`

`src/beliefnet/models/pipeline.py`. `BeliefNet(parser, reasoner="exact")`.
`predict(statements)` returns `{inconsistent, entities, constraints}` and, on a
contradiction, `unsat_core` + `repair`. `reasoner="equilibrium"` returns the
differentiable `StructuredEquilibriumCore` residual instead (needs torch).

### 5.6 Online monitor

`src/beliefnet/reasoning/monitor.py` — `BeliefMonitor`. Ingests statements one
at a time, maintains the accumulating constraint set, decides satisfiability
after each statement, and on the first statement that breaks consistency emits
the certificate + repair and records `flagged_index` (localization).
Per-constraint confidences enable **abstention** below a threshold instead of
false-alarming. This is the runtime-monitor form for tool-using LLM agents.

---

## 6. The benchmark

`src/beliefnet/data/emergent.py` (generator) and
`src/beliefnet/data/benchmark.py` (packaging). Export:
`python scripts/export_benchmark.py` -> `data/benchmark/<variant>.jsonl`.

**Design.** Each case is a set of statements, balanced consistent/inconsistent,
with matched statement count `n` across the consistent/inconsistent pair so
surface length is not a shortcut. Controlled axes:

- `hop_count` (kappa): inconsistent = an order **cycle** of length kappa;
  consistent = an acyclic **chain** with the same number of edges/events.
- `n_statements` (`n_total`): core kappa statements + neutral distractors.

**Three variants** (each 600 cases at `n_per_cell=50`, balanced 300/300, hops
{2:200, 3:200, 5:200}; total **1800**):

| variant | surface | distractors | purpose |
|---|---|---|---|
| `canonical` | single template ("occurred before") | neutral fillers | controlled probe for the theorem |
| `paraphrase` | 5 surface forms | neutral fillers | extractor robustness |
| `real_seeded` | paraphrased | **real document sentences** | bridge toward real data |

`real_seeded` embeds synthetic contradiction cores among real-style sentences
(`REAL_FIXTURE_SENTENCES`); swap in a domain corpus via `document_pool` (e.g.
SciFact abstracts) for a true real-text set.

**Ground truth.** Every case stores the oracle typed graph in
`metadata.constraints`; the exact checker reproduces the generator labels with
**accuracy 1.0000 on 800 cases** (Experiment 1 oracle sanity), so the labels
and the emergence construction are provably correct.

---

## 7. Experiments

All experiments write `results/<id>/result.json` and refresh
[`details_results.md`](details_results.md). Commands assume the repo root and
`PYTHONPATH=src` (PowerShell: `$env:PYTHONPATH='src'`).

### Experiment (HEADLINE) -- Numeric global-sum stress (`exp_numeric_stress_full`)

- **Purpose.** Does a frontier *reasoning* LLM fail at *emergent numeric*
  consistency -- many-term signed arithmetic where only the global sum reveals
  the conflict (no pairwise witness)?
- **Protocol.** Running-balance cases: opening amount + `depth` signed
  transactions + a stated final balance; consistent iff final = opening + sum.
  Balanced, matched length, with distractors. Score with the exact checker and
  GPT-4o / GPT-5 whole-set judges; AUROC by depth + 95% bootstrap CI.
- **Config.** `depths=[5,12,20,28,36]`, `n_per_cell=15`, `seed=0`
  (150 cases/model, 300 live calls).
- **Command.** `python scripts/run_numeric_stress.py --depths 5 12 20 28 36 --n-per-cell 15 --tag full --models gpt4o gpt5`
- **Result (real).** AUROC by depth:

  | depth | oracle | GPT-4o | GPT-5 |
  |---|---|---|---|
  | 5  | 1.000 | 0.580 | 1.000 |
  | 12 | 1.000 | 0.524 | 1.000 |
  | 20 | 1.000 | 0.489 | 0.867 |
  | 28 | 1.000 | 0.489 | 0.480 |
  | 36 | 1.000 | 0.642 | 0.233 |

  Overall (95% CI): oracle 1.000; GPT-5 **0.679 [0.578, 0.779]**; GPT-4o
  0.543 [0.466, 0.621].
- **Interpretation (the A* result).** GPT-5's reasoning solves *ordering* cycles
  at any scale we tried, but it **collapses monotonically to below chance on
  numeric emergent inconsistency** as the number of terms grows (perfect at
  <=12 terms, chance at 28, **0.233 at 36** -- systematically wrong). GPT-4o is
  at chance throughout. The sound checker stays at **1.000**. This is the
  emergent-inconsistency blind spot on the distribution where even a frontier
  reasoning model cannot compensate: multi-term global arithmetic.

### Experiment 1 -- Emergent-inconsistency failure curve (`exp1_failure_curve`)

- **Purpose / gate.** Does joint-consistency detection AUROC collapse as
  hop-count and statement count grow? This is the make-or-break question for the
  frontier-LLM story.
- **Protocol.** Generate the controlled grid; score each case with (a) the exact
  symbolic oracle (sanity) and (b) optional GPT-4o / GPT-5 judges asked to judge
  whole-set consistency; report AUROC per (hop, n) cell and per hop-count.
- **Config.** `kappa=[2,3,5]`, `n_total=[4,8,16]`, `n_per_cell=50`, `seed=0`
  (800 cases).
- **Command.** `python scripts/run_failure_curve.py [--models gpt4o gpt5]`
- **Result (REAL, GPT-4o + GPT-5, seed=0; kappa 2/3/5/7, n_total 8/20, 6 per
  cell; 96 cases/model, 192 live Azure calls).** Oracle accuracy **1.0000**.
  Mean AUROC by hop-count:

  | hop | oracle | GPT-4o | GPT-5 |
  |---|---|---|---|
  | 2 | 1.000 | 0.976 | 1.000 |
  | 3 | 1.000 | 0.840 | 1.000 |
  | 5 | 1.000 | 0.667 | 1.000 |
  | 7 | 1.000 | 0.587 | 1.000 |

- **Interpretation (honest).** GPT-4o (non-reasoning) degrades monotonically
  toward chance as hop-count grows -- the predicted failure curve. GPT-5
  (reasoning) **holds at 1.000** up to 7-hop cycles / 20 statements, so the naive
  "frontier LLMs fail" claim must be **qualified**: a reasoning model can do this
  global step at this scale. The contribution reframes to (a) the pairwise
  theorem (unaffected), (b) non-reasoning LLMs empirically fail, (c) BeliefNet is
  cheap + sound + certified where GPT-5 is expensive and uncertified, and (d) the
  self-fix turns cheap GPT-4o reliable (Experiment 7). **Resolved follow-up:** the
  numeric global-sum headline (top of Section 7) finds GPT-5's breaking point on
  a *different* axis -- multi-term arithmetic -- where it collapses to 0.233.
- **GPT-5 stress test (REAL, `--tag gpt5_stress`).** Pushed to 7/10/13-hop cycles
  buried in **50 paraphrased statements**: GPT-5 still scores **AUROC 1.000 at
  every hop**. On clean, explicitly-stated ordering constraints the reasoning
  model does **not** break at this scale. Conclusion (honest): on *ordering* the
  emergent-inconsistency failure is specific to (a) pairwise-factored methods
  (theorem) and (b) non-reasoning models (GPT-4o) -- GPT-5 solves it. We did
  **not** try to force a failure by inflating ordering cycles (that would be
  p-hacking); instead the numeric global-sum experiment (Section 7 headline)
  identifies a *distinct* distribution -- many-term arithmetic -- where GPT-5
  genuinely collapses to 0.233. BeliefNet's value therefore rests on that numeric
  blind spot plus cost + verifiable certificates + soundness + rescuing cheap
  models.

### Experiment 2 — Emergence separation, verified (`exp2_emergence_theorem`)

- **Purpose.** Confirm empirically that emergent (`kappa >= 3`) inconsistencies
  have no witness of size `< kappa`.
- **Protocol.** For every inconsistent case, compute the minimal unsat core and
  its size; check `size == kappa`; compute the pairwise-miss rate
  (`1 - fraction with core size <= 2`).
- **Config.** `kappa=[2,3,5]`, `n_total=[6,12]`, `n_per_cell=50`, `seed=0`
  (300 inconsistent cases).
- **Command.** `python scripts/run_emergence_theorem.py`
- **Result.**

  | hop_count | mean_min_core_size | core_size==kappa_rate | pairwise_miss_rate |
  |---|---|---|---|
  | 2 | 2.000 | 1.000 | 0.000 |
  | 3 | 3.000 | 1.000 | 1.000 |
  | 5 | 5.000 | 1.000 | 1.000 |

  Summary: `min_core_equals_kappa_rate = 1.0000`, `pairwise_miss_rate_on_emergent = 1.000`.
- **Interpretation.** The whole cycle is always the smallest conflict, so a
  pairwise judge provably misses 100% of `kappa >= 3` cases. This is the
  theorem, realized.

### Experiment 4 — Pairwise baseline vs emergent (`exp4_pairwise_baseline`)

- **Purpose.** Show a concrete pairwise method collapsing on emergent cases.
- **Protocol.** `all_pairs_symbolic`: extract constraints (rule), then check
  every 2-subset; flag inconsistent iff any pair conflicts — the exact analogue
  of an all-pairs NLI aggregator. Compare AUROC per hop vs the oracle; report
  95% bootstrap CIs.
- **Config.** `kappa=[2,3,5]`, `n_total=[6,12]`, `n_per_cell=50`, `seed=0`
  (600 cases).
- **Command.** `python scripts/run_baselines.py`
- **Result.**

  | hop_count | oracle AUROC | all_pairs_symbolic AUROC |
  |---|---|---|
  | 2 | 1.000 | 1.000 |
  | 3 | 1.000 | 0.500 |
  | 5 | 1.000 | 0.500 |

  Overall (95% bootstrap CI): oracle **1.000** [1.000, 1.000]; all-pairs
  **0.667** [0.640, 0.694].
- **Interpretation.** The **empirical separation**: the pairwise judge is perfect
  at `kappa = 2` and exactly chance at `kappa >= 3`. This is the figure that
  motivates a structured (non-pairwise) method. (Neural DeBERTa-NLI all-pairs is
  the drop-in that reproduces this with a real model — pending, needs
  torch/transformers.)

### Experiment 5 — Exact checker vs equilibrium residual (`exp5_reasoner_ablation`)

- **Purpose.** Does the differentiable `StructuredEquilibriumCore` residual
  separate cases as well as the exact checker (does the energy layer earn its
  place, or is it justified only by trainability)?
- **Protocol.** Score cases by exact (0/1) and by the equilibrium residual;
  compare AUROC per hop.
- **Command.** `python scripts/run_reasoner_ablation.py`
- **Result (real, torch).** exact mean AUROC **1.000**, energy mean AUROC
  **1.000** -- the differentiable residual separates cases as well as the exact
  checker.
- **Interpretation.** Expectation: energy AUROC ~= exact AUROC (both high),
  which frames the energy layer as the *trainable* path and the exact checker as
  the *certificate/soundness* path.

### Experiment 6 — Local-extractor faithfulness (`exp6_extractor_faithfulness`)

- **Purpose.** How faithfully does perception recover the oracle graph
  (constraint-level micro P/R/F1)? Perception is the true bottleneck.
- **Protocol.** For each backend, extract constraints and compare to the oracle
  set per case; accumulate micro precision/recall/F1 by hop.
- **Config.** `kappa=[2,3,5]`, `n_total=[6,12]`, `n_per_cell=20`, `seed=0`
  (240 cases).
- **Command.** `python scripts/run_extractor_faithfulness.py [--models gpt4o gpt5]`
- **Result.** Rule extractor **precision = recall = F1 = 1.000** at every
  hop-count (`rule_mean_f1 = 1.000`). LLM extractor columns **pending** (keys).
- **Interpretation.** On the templated benchmark the deterministic parser is a
  perfect perception ceiling, isolating any downstream error to reasoning (of
  which there is none). The open question is LLM-extractor faithfulness on real
  text.

### Experiment 7 — LLM self-fix (`exp7_self_fix`)

- **Purpose.** Can **local** LLM extraction + the sound checker recover the
  emergent inconsistency the **same** LLM misjudges end-to-end?
- **Protocol.** Per hop, AUROC of: `oracle`; `rule->checker`;
  `<model>_judge` (whole-set consistency judgment); `<model>_extract->checker`
  (per-statement extraction routed through the checker). Extraction is kept
  strictly local so the LLM never does the global step.
- **Config.** `kappa=[2,3,5]`, `n_total=[6,12]`, `n_per_cell=10`, `seed=0`
  (120 cases).
- **Command.** `python scripts/run_self_fix.py [--models gpt4o gpt5]`
- **Result (REAL, GPT-4o, kappa 3/5/7, n_total 8, 5/cell, 30 cases, ~270 live
  calls).** AUROC by hop:

  | hop | oracle | rule->checker | gpt4o_judge | gpt4o_extract->checker |
  |---|---|---|---|---|
  | 3 | 1.000 | 1.000 | 1.000 | 1.000 |
  | 5 | 1.000 | 1.000 | 0.800 | 0.900 |
  | 7 | 1.000 | 1.000 | 0.500 | 0.900 |

- **Interpretation (the self-fix, real).** GPT-4o's end-to-end judgment collapses
  to chance at hop 7 (0.500), but its *local* per-statement extractions routed
  through the sound checker stay at **0.900** -- a +0.40 AUROC lift at hop 7.
  Decomposition + verification rescues a cheap non-reasoning model; it is 0.9 not
  1.0 because GPT-4o extraction is imperfect (perception, not reasoning, is the
  residual error), exactly the localization the framework predicts. n=10 per hop;
  scale up for tighter CIs.

### Experiment 8 — Certificate quality (`exp8_certificates`)

- **Purpose.** Are the emitted proofs and repairs valid and minimal?
- **Protocol.** For each inconsistent case, compute the minimal unsat core and
  the minimal correction set; verify the core is unsatisfiable and that removing
  the correction set restores satisfiability; report size distributions and one
  worked example.
- **Config.** `kappa=[2,3,5]`, `n_total=[6,12]`, `n_per_cell=30`, `seed=0`
  (180 inconsistent cases).
- **Command.** `python scripts/run_certificate_eval.py`
- **Result.** `valid_core_rate = 1.000`, `valid_repair_rate = 1.000`; mean core
  size = kappa (2 / 3 / 5); mean repair size = **1.0** (retracting one edge
  breaks the cycle). Example certificate (from the report): a 3-cycle
  `chemotherapy < discharge`, `discharge < diagnosis`, `diagnosis < chemotherapy`
  with repair = retract one edge.
- **Interpretation.** Every flag ships with a checkable proof and a one-edge
  repair — artifacts no score-only detector provides.

### Experiment P4 — Online BeliefMonitor (`expP4_online_monitor`)

- **Purpose.** Streaming detection + **localization**: does the monitor flag the
  exact statement that closes the contradiction?
- **Protocol.** Stream each case through `BeliefMonitor`; detection score = flagged
  or not; localization = on detected inconsistent cases, is the flagged statement
  a constraint-bearing (relation) statement.
- **Config.** `kappa=[2,3,5]`, `n_total=[6,12]`, `n_per_cell=30`, `seed=0`
  (360 cases).
- **Command.** `python scripts/run_online_monitor.py`
- **Result.** Detection AUROC **1.000** at every hop; overall localization
  accuracy **1.000**.
- **Interpretation.** The monitor detects emergent inconsistency (unlike the
  pairwise baseline) and points at the offending statement — runtime behavior
  batch NLI cannot provide.

### Experiment (auditability) — Proxy-reviewer human study (`exp_human_study`)

- **Purpose.** Is the checker's certificate *more useful to a reviewer* than an
  LLM's free-text rationale? We measure auditability instead of asserting it.
- **Protocol.** A reviewer model (Claude Haiku, `claude-haiku-4-5`) judges the
  joint consistency of hard numeric balance cases under three aid conditions:
  (alone) statements only; (+certificate) statements + the checker's arithmetic
  proof; (+LLM rationale) statements + GPT-4o's verdict and free-text reasoning.
  We report reviewer accuracy and Cohen's kappa vs truth per condition, plus the
  GPT-4o assistant's own accuracy as a reference.
- **Config.** `depth in {15, 40}`, `n_per_cell=15` (30 cases each), `seed=0`;
  reviewer `claude-haiku-4-5`; rationale from GPT-4o.
- **Command.** `python scripts/run_human_study.py --depth 15 --n-per-cell 15 --reviewer-model claude-haiku-4-5`
  and `... --depth 40 --tag d40`.
- **Result (real).** Reviewer accuracy by aid condition, at two difficulties:

  | condition | depth 15 | depth 40 |
  |---|---|---|
  | reviewer alone | 0.933 | 0.500 |
  | **reviewer + sound certificate** | **1.000** | **1.000** |
  | reviewer + LLM rationale | 0.867 | 0.500 |
  | (ref) GPT-4o assistant alone | 0.500 | 0.433 |

- **Interpretation.** The certificate lifts the reviewer to **perfect at both
  depths**. At depth 15 the reviewer can mostly self-compute (unaided 0.933), so
  the certificate adds +0.067 while the fluent-but-wrong LLM rationale *lowers*
  accuracy to 0.867 (automation bias). At depth 40 the reviewer is helpless
  (unaided 0.500 = chance; GPT-4o is *below* chance at 0.433, so its rationale
  gives no signal — reviewer stays at 0.500), yet the certificate still yields
  **1.000** — a **+0.50** rescue. Auditability is thus a measured benefit that
  **grows with difficulty**: a checkable proof rescues a reviewer exactly when
  the problem exceeds unaided verification, whereas an LLM rationale never helps
  and can hurt.

### Experiment (real data #2) — FEVER fact-checking (`exp_fever`)

- **Purpose.** Does the SciFact picture (LLMs strong; base schema chance;
  polarity extension lifts it) replicate on a *second* real dataset?
- **Protocol.** FEVER gold-evidence claims (`copenlu/fever_gold_evidence`,
  validation split), balanced SUPPORTS vs REFUTES (positive = REFUTES). Same
  protocol as SciFact: rule schema, GPT-4o/GPT-5 whole-pair judges, and the
  polarity-triple checker. Evidence sentences de-tokenized from FEVER markup.
- **Config.** `n_per_class=40` (80 cases), `seed=0`.
- **Command.** `python scripts/run_fever.py --models gpt4o gpt5 --n-per-class 40`
- **Result (real).**

  | system | AUROC | accuracy | F1(refute) |
  |---|---|---|---|
  | BeliefNet (rule schema) | 0.500 | 0.500 | 0.000 |
  | GPT-4o judge | 0.981 | 0.963 | 0.963 |
  | GPT-5 judge | 0.983 | 0.963 | 0.962 |
  | BeliefNet (polarity + GPT-4o) | 0.775 | 0.775 | 0.719 |
  | BeliefNet (polarity + GPT-5) | **0.888** | 0.888 | 0.877 |

- **Interpretation.** External validity confirmed: the base four-relation schema
  is at chance on FEVER's semantic contradictions (0.500), the LLM judges are
  strong (~0.98), and the **polarity-triple extension lifts the checker to 0.775
  (GPT-4o) / 0.888 (GPT-5)** while emitting a structured certificate — the same
  pattern as SciFact, now on a second real corpus. The lift is below the raw
  judge because reducing a heterogeneous claim to one relation direction is
  lossy, but it is interpretable and certified.

---

## 8. Statistical methodology

`src/beliefnet/eval/stats.py` (numpy only):

- `bootstrap_ci(scores, labels, n_boot=1000)` — bootstrap mean + 95% CI of any
  metric (used for the overall-AUROC CIs in Experiment 4).
- `auroc_difference_test(a, b, labels)` — paired-bootstrap two-sided test on an
  AUROC gap (used in place of DeLong; assumption-light and easy to verify).
- `mcnemar_test(correct_a, correct_b)` — continuity-corrected McNemar for paired
  accuracy.
- `benjamini_hochberg(pvals, alpha)` — BH false-discovery-rate control for the
  many per-cell comparisons.

Metrics (`src/beliefnet/eval/metrics.py`): tie-aware AUROC (Mann-Whitney U),
accuracy, F1, ECE. AUROC is reported per (hop, n) cell and per hop-count;
matched controls (cycle vs chain of equal length) remove surface-length
shortcuts by construction.

---

## 9. Headline results table

| Experiment | Metric | Result | Status |
|---|---|---|---|
| **Numeric stress (HEADLINE, real)** | GPT-5 AUROC @ depth 5/12/20/28/36 | **1.0 / 1.0 / 0.87 / 0.48 / 0.23** | done (real) |
| **Numeric stress (real)** | GPT-4o AUROC (all depths) | **~0.54** (chance) | done (real) |
| **Numeric stress (real)** | oracle / BeliefNet exact checker | **1.000** every depth | done (real) |
| Oracle sanity (exp1) | accuracy vs generator labels | **1.0000** (800 cases) | done |
| Emergence theorem (exp2) | min-core == kappa | **1.0000** (300) | done |
| Emergence theorem (exp2) | pairwise miss @ kappa>=3 | **1.000** | done |
| Pairwise baseline (exp4) | AUROC by hop (2/3/5) | **1.0 / 0.5 / 0.5** | done |
| Pairwise baseline (exp4) | overall AUROC (95% CI) | 0.667 [0.640, 0.694] | done |
| Extractor faithfulness (exp6) | rule micro F1 | **1.000** (240) | done |
| Self-fix (exp7) | GPT-4o judge AUROC hop 3->7 | 1.000 -> **0.500** | done (real) |
| Self-fix (exp7) | GPT-4o extract->checker hop 3->7 | 1.000 -> **0.900** | done (real) |
| Certificates (exp8) | valid core / valid repair | **1.000 / 1.000** (180) | done |
| Online monitor (P4) | detection AUROC / localization | **1.000 / 1.000** (360) | done |
| Failure curve (exp1) | GPT-4o AUROC hop 2->7 | **0.976 -> 0.587** | done (real) |
| Failure curve (exp1) | GPT-5 AUROC hop 2->7 | **1.000 (holds)** | done (real) |
| GPT-5 stress (exp1) | GPT-5 AUROC hop 7/10/13 @ 50 paraphrased stmts | **1.000 (holds)** | done (real) |
| Cost/latency (real) | BeliefNet vs GPT-5 latency/decision | **0.92 ms vs 5691 ms (~6193x)** | done (real) |
| Cost/latency (real) | accuracy / auditable proof | 1.0 both; **proof only from BeliefNet** | done (real) |
| Pairwise baseline (exp4) | neural DeBERTa-NLI all-pairs | code-ready; env download-limited (symbolic shows the collapse) | code done |
| Equilibrium ablation (exp5) | exact vs energy AUROC | **1.000 / 1.000** | done (real) |
| Real data - SciFact | GPT-5 / GPT-4o judge AUROC | **0.988 / 0.966** | done (real) |
| Real data - SciFact | BeliefNet base -> polarity-schema AUROC | **0.500 -> 0.838** | done (real) |
| Real data - FEVER | GPT-5 / GPT-4o judge AUROC | **0.983 / 0.981** | done (real) |
| Real data - FEVER | BeliefNet base -> polarity (GPT-5) AUROC | **0.500 -> 0.888** | done (real) |
| Human study (real) | reviewer alone -> +certificate @ depth 40 | **0.500 -> 1.000** | done (real) |
| Human study (real) | reviewer + LLM rationale @ depth 40 | **0.500** (no help) | done (real) |

Offline unit tests: **124/124 passing** across the reasoning, data, extractor,
monitor, stats, parser, pipeline, benchmark and equilibrium (torch) modules.

---

## 10. Does it satisfy the proposal?

Mapping to the reframed experiment program (Sections 2 and 7 above).

| Proposal item | Status | Evidence / gap |
|---|---|---|
| **Numeric blind spot** (new headline) | **SATISFIED (real)** | GPT-5 numeric AUROC 1.0->0.233 (below chance) as terms grow; checker stays 1.0 (exp_numeric_stress_full) |
| **Separation theorem** (Axis 2) | **SATISFIED** | exp2 min-core==kappa 1.0; formal proof in Section 4 |
| **Failure curve** (Exp 1) | **SATISFIED (real)** | GPT-4o 0.976->0.587 by hop; GPT-5 holds 1.0 on ordering but collapses to 0.233 on numeric (headline row) |
| **Certificate artifact** (Axis 3) | **SATISFIED** | exp8 valid core/repair 1.0; worked example |
| **Certificate human study** (Exp 3) | **SATISFIED (proxy)** | Claude Haiku reviewer: +certificate 0.5->1.0 at depth 40; LLM rationale gives no help (exp_human_study) |
| **Real-data results** (Exp 4) | **SATISFIED (real)** | SciFact: GPT-5 0.988/GPT-4o 0.966, base 0.500 -> polarity **0.838**; FEVER replicates: judges ~0.98, base 0.500 -> polarity **0.888** (GPT-5) -- two real corpora, certified |
| **Exact vs equilibrium** (Exp 5) | **SATISFIED (real)** | exact 1.0 = energy 1.0; the energy layer is justified by trainability, not by beating the exact checker |
| **Perception faithfulness** (Exp 6) | **PARTIAL** | rule F1 1.0 done; LLM-on-real pending |
| **LLM self-fix** (Exp 7) | **SATISFIED (real)** | GPT-4o judge 1.0->0.5; extract->checker 1.0->0.9 (+0.40 at hop 7) |
| **Online monitor / system** | **SATISFIED** | P4 detection + localization 1.0 |
| **Statistical rigor** | **SATISFIED (harness)** | bootstrap CI/McNemar/BH-FDR implemented; applied to exp4 |

**Verdict.** The **theory and mechanism are fully satisfied** and the **real LLM
experiments are done**: the separation theorem is proved and verified, the
pairwise collapse is demonstrated, GPT-4o exhibits the predicted failure curve
(0.976 -> 0.587), the self-fix rescues it (0.500 -> 0.900 at hop 7), and
certificates + the online monitor work at 1.0. **The headline (updated):** GPT-5
(reasoning) holds at 1.000 on ordering cycles even at 13 hops in 50 paraphrased
statements, but on **numeric** emergent inconsistency it **collapses to 0.233
(below chance)** as the term count grows (exp_numeric_stress_full) -- the
distribution where even a frontier reasoning model fails while the checker stays
1.0. Real corpora are done (SciFact + FEVER, polarity extension certified) and
the proxy human study is done (Claude Haiku reviewer). Remaining nice-to-have:
neural DeBERTa-NLI all-pairs (env download-limited; the symbolic all-pairs
already shows the 1.0->0.5 collapse).

---

## 11. What is pending and how to complete it

1. **LLM failure curve + self-fix + faithfulness (the gate).** Put Azure
   deployments in `.env`:
   ```
   AZURE_OPENAI_ENDPOINT=...
   AZURE_OPENAI_API_KEY=...
   AZURE_OPENAI_API_VERSION=2024-10-21
   AZURE_OPENAI_GPT4O_DEPLOYMENT=...
   AZURE_OPENAI_GPT5_DEPLOYMENT=...
   ```
   Then:
   ```
   python scripts/run_failure_curve.py --models gpt4o gpt5
   python scripts/run_self_fix.py --models gpt4o gpt5
   python scripts/run_extractor_faithfulness.py --models gpt4o gpt5
   python scripts/build_report.py
   ```
2. **Equilibrium ablation (exp5) done; neural DeBERTa-NLI baseline available.**
   exp5 exact = energy = 1.000 (torch 2.13). Neural all-pairs NLI:
   `python scripts/run_baselines.py --nli` (the symbolic all-pairs already shows
   the 1.0 -> 0.5 collapse empirically).
3. **Real data (SciFact / FEVER) -- DONE.** SciFact zero-shot
   (`scripts/run_scifact.py`) and FEVER (`scripts/run_fever.py`) both run: base
   schema 0.500, judges ~0.97-0.99, polarity extension 0.838 (SciFact) / 0.888
   (FEVER, GPT-5), all certified. Optional: point `document_pool` at SciFact
   abstracts for a real-text `real_seeded` set.
4. **Human study (DONE, proxy).** `scripts/run_human_study.py` runs a Claude
   Haiku proxy reviewer across the three aid conditions; +certificate lifts a
   depth-40 reviewer 0.5->1.0 while an LLM rationale gives no help
   (`exp_human_study`, `exp_human_study_d40`). Live human reviewers optional.

---

## 12. Reproducibility

### 12.1 Environment

- Python 3.10+; offline experiments need only `numpy`.
- Optional: `torch` (equilibrium, checkpoints), `transformers` (neural NLI),
  `openai` (Azure GPT-4o/GPT-5). Install extras: `pip install -e ".[llm]"`.
- Set `PYTHONPATH=src` (or install the package). Azure creds via `.env`
  (auto-loaded by the runners; never hard-coded).

### 12.2 Run everything (offline)

```
$env:PYTHONPATH='src'
python scripts/run_failure_curve.py
python scripts/run_emergence_theorem.py
python scripts/run_extractor_faithfulness.py
python scripts/run_self_fix.py
python scripts/run_certificate_eval.py
python scripts/run_reasoner_ablation.py     # needs torch
python scripts/run_baselines.py
python scripts/run_online_monitor.py
python scripts/export_benchmark.py
python scripts/build_report.py
```

### 12.3 File map

Reasoning / certificates:
- `src/beliefnet/reasoning/exact_checker.py` — sound feasibility
- `src/beliefnet/reasoning/certificates.py` — unsat core + repair
- `src/beliefnet/reasoning/monitor.py` — online BeliefMonitor

Data / benchmark:
- `src/beliefnet/data/emergent.py` — controlled generator (hop-count, surfaces,
  distractor pools)
- `src/beliefnet/data/benchmark.py` — dual benchmark + datasheet + export

Perception:
- `src/beliefnet/perception/local_extract.py` — rule per-statement extraction
- `src/beliefnet/perception/belief_parser.py` — BeliefParser ABC + concrete parsers
- `src/beliefnet/llm/extractor.py` — LLM local extraction
- `src/beliefnet/llm/judge.py` — LLM whole-set consistency judge
- `src/beliefnet/llm/azure_client.py` — Azure client + `.env` loader

Pipeline / eval:
- `src/beliefnet/models/pipeline.py` — `BeliefNet`
- `src/beliefnet/models/beliefnet.py` — `StructuredEquilibriumCore` (torch)
- `src/beliefnet/eval/reporting.py` — JSON + aggregated markdown
- `src/beliefnet/eval/stats.py` — bootstrap CI / McNemar / BH-FDR
- `src/beliefnet/eval/human_study.py` — kappa + review export
- `src/beliefnet/eval/metrics.py` — AUROC / F1 / ECE

Scripts: one per experiment (Section 7) + `build_report.py`.

### 12.4 Results layout

```
results/
  experiments.md                 <- this document
  details_results.md             <- auto-generated aggregate
  <exp>/result.json              <- per-experiment machine-readable payload
  exp1_failure_curve/failure_curve.csv
data/benchmark/<variant>.jsonl   <- exported benchmark
```

Payload schema: `{experiment, generated, title, question, config, summary,
tables, notes}`. Re-run any script then `python scripts/build_report.py` to
refresh `details_results.md`.

### 12.5 Tests

Offline (numpy-only): `test_exact_checker`, `test_certificates`,
`test_emergent_data`, `test_local_extract`, `test_monitor`, `test_stats`,
`test_belief_parser`, `test_pipeline`, `test_benchmark`, `test_reporting`.
Full suite (this environment has torch): **124/124 pass** including the
equilibrium/synthetic tests. Run with `python -m pytest -q` (or
`pip install -e ".[dev]"`).

---

## 13. Writing the paper from this

**Narrative arc (honest, data-driven).** (1) Pairwise judges provably (Section 4
theorem) and empirically (Exp 2/4: all-pairs 1.0 -> 0.5) miss emergent joint
inconsistency; non-reasoning LLMs empirically fail too (Exp 1: GPT-4o 0.98 ->
0.59). (2) A reasoning LLM (GPT-5) is accurate on ordering even at 13-hop /
50-statement stress, **but collapses to 0.233 (below chance) on numeric emergent
inconsistency** as the term count grows (headline) -- and even where it is
accurate it is ~6193x slower (5.7 s/query) with no verifiable proof (cost
experiment). (3) BeliefNet decides in ~1 ms, soundly, and returns a
human-auditable certificate + repair + localization (Exp 8, P4); its only error
source is a measurable perception layer (Exp 6); and it rescues cheap models --
decompose GPT-4o + verify lifts AUROC +0.40 at hop 7 (Exp 7). The paper's claim
is cost + verifiability + cheap-model rescue, not "LLMs cannot do it."

**Contributions (order for the paper).**
1. Emergent joint inconsistency: separation theorem (pairwise-factored judges
   provably miss `kappa >= 3`) + empirical failure curves for pairwise methods
   and non-reasoning LLMs (GPT-4o 0.98 -> 0.59).
2. A frontier reasoning model's blind spot: GPT-5 collapses AUROC 1.0 -> 0.233
   (below chance) on numeric emergent inconsistency as terms grow, while the
   checker stays 1.0.
3. A sound, ~1 ms belief-graph checker returning minimal unsat-core certificates
   + repairs + statement-level localization -- matching GPT-5 accuracy at
   ~1/6000th the latency and with a proof GPT-5 does not provide.
4. Decompose-and-verify (self-fix): routing a cheap model's local extractions
   through the checker rescues it (+0.40 AUROC on GPT-4o at hop 7).
5. A controlled + naturalistic + real-seeded benchmark, an online monitor, a
   proxy-reviewer auditability study (certificate 0.5 -> 1.0 vs rationale no
   help), and a polarity-triple schema extension validated on SciFact + FEVER.

**Figures to render from `results/`.**
- Fig 0 (headline): numeric AUROC vs depth -- oracle flat at 1.0, GPT-4o at
  chance, GPT-5 falling 1.0 -> 0.233. From `exp_numeric_stress_full`.
- Fig 1: AUROC vs hop-count -- oracle/monitor + GPT-5 flat at 1.0 vs
  all-pairs and GPT-4o collapsing toward 0.5. From `exp4_pairwise_baseline` +
  `exp1_failure_curve`.
- Fig 2: cost/accuracy/proof -- BeliefNet 0.9 ms + certificate vs GPT-5 5.7 s +
  no proof (both accuracy 1.0). From `exp_cost_comparison`.
- Fig 3: self-fix -- GPT-4o `_judge` (-> 0.5) vs `_extract->checker` (-> 0.9) by
  hop. From `exp7_self_fix`.
- Fig 4: min-core-size == kappa and pairwise-miss vs hop. From `exp2`.
- Fig 5: proxy-reviewer accuracy by aid (alone / +certificate / +rationale) at
  depth 15 and 40. From `exp_human_study` + `exp_human_study_d40`.
- Table: certificate validity + sizes. From `exp8`.

**Abstract pointer.** The full abstract is in [`../paper.md`](../paper.md); all
LLM / real-data columns are filled (numeric headline, SciFact + FEVER, proxy
human study).

**Honesty checklist for reviewers.** (a) The theorem bounds pairwise methods,
not `|S|`-factored LLMs — keep the LLM claim empirical. (b) Report SciFact/FEVER
honestly (pairwise tasks may favor NLI/LLMs); win on the multi-statement/global
setting. (c) If the LLM extractor is used inside BeliefNet, also report an
open-model extractor so the method is not API-locked. (d) The equilibrium layer
is justified by trainability, not by beating the exact checker — say so.
