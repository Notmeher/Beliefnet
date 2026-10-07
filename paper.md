# BeliefNet: Sound, Cheap, Auditable Detection of Emergent Joint Inconsistency in Language-Model Assertions

> End-to-end paper draft. All quantitative claims are backed by real runs in
> [`results/`](results/) (per-experiment `result.json`, aggregated in
> [`results/details_results.md`](results/details_results.md), full protocol in
> [`results/experiments.md`](results/experiments.md)). Live LLM numbers are from
> Azure GPT-4o and GPT-5; the SciFact numbers are from the real SciFact dev split.

---

## Abstract

Language models assert many facts across long outputs but cannot certify that
those assertions form a *jointly consistent* set. We study **emergent joint
inconsistency**: contradictions with no pairwise witness, whose smallest
unsatisfiable subset has size `k >= 3` (e.g. an ordering cycle `A<B<C<A`). We
prove a **separation theorem**: any consistency judge that factorizes over
subsets of size `< k` — all pairwise NLI and all-pairs aggregators — must miss
them, and verify it empirically (an all-pairs judge drops from AUROC 1.0 at
`k=2` to 0.5 at `k>=3`). Empirically, a non-reasoning LLM (GPT-4o) also
collapses toward chance as the hop-count grows (0.976 → 0.587), while a
reasoning LLM (GPT-5) remains accurate on *ordering* cycles even at 13 hops in
50 statements. Crucially, however, on **numeric emergent inconsistency** --
many-term signed arithmetic where only the global sum reveals the conflict --
GPT-5 itself collapses **monotonically to below chance** (AUROC 1.0 at `<=12`
terms → **0.233 at 36 terms**; overall 0.68, 95% CI [0.58, 0.78]) and GPT-4o is
near chance throughout, while the exact checker stays at 1.0. We
contribute **BeliefNet**, a belief-graph checker that (i) decides joint
satisfiability *soundly and exactly* in ~1 ms — **~6000× faster than a GPT-5
call at equal accuracy** — while returning a **minimal unsatisfiable-core
certificate** and a **minimal repair** that no LLM verdict provides; (ii)
**rescues cheap models**: routing GPT-4o's local extractions through the checker
lifts AUROC by +0.40 at hop 7 (decompose-and-verify); and (iii) runs as an
**online monitor** that localizes the exact statement that breaks consistency.
In a **proxy-reviewer study**, giving a reviewer the checker's certificate lifts
verification accuracy to 1.0, whereas a fluent GPT-4o rationale drags it *below*
the unaided baseline — auditability measured, not asserted.
On real biomedical claims (SciFact) the base schema is too narrow (chance), but
a lightweight **polarity-triple extension** driven by an LLM extractor lifts
AUROC from 0.50 to 0.84 while emitting a structured certificate (below the 0.99
raw judge — the direction reduction is lossy). BeliefNet reframes
consistency verification from an accuracy race into a question of **cost,
soundness, and auditability**.

---

## 1. Introduction

Frontier LLMs are deployed as research assistants, clinical decision support,
and long-form writers. A failure mode that matters in all three is **joint
inconsistency**: the model asserts `A`, later asserts something incompatible
with `A`, and a reader without the budget to cross-check accepts both.
Mainstream detectors — confidence thresholding, retrieval verification, trained
NLI classifiers — share two properties: their decision is an opaque function,
and it is **pairwise** (a claim compared to one piece of evidence at a time).

We ask: can an explicit typed belief graph plus a *sound* reasoning layer detect
*global* joint inconsistency, including contradictions with no pairwise witness,
while producing a human-auditable proof — where pairwise methods provably
cannot, and at a cost far below a reasoning-LLM call?

Our contributions:
1. **Emergent joint inconsistency** as a problem, with a separation theorem
   (Section 4) and empirical failure curves for pairwise methods and
   non-reasoning LLMs (Section 7).
2. **A frontier reasoning model's blind spot.** On *numeric* emergent
   inconsistency (many-term global sums) **GPT-5 collapses from AUROC 1.000 to
   0.233 (below chance) as the term count grows** (Section 7.0), while a sound
   checker stays at 1.000 — the first distribution we find where reasoning does
   not compensate.
3. **BeliefNet**, a sound ~1 ms belief-graph checker returning unsat-core
   certificates + repairs + statement-level localization, matching GPT-5
   accuracy at ~1/6000th the latency with a proof GPT-5 does not give.
4. **Decompose-and-verify (self-fix)**: routing a cheap model's local
   extractions through the checker rescues it (+0.40 AUROC on GPT-4o).
5. A controlled + naturalistic + real-seeded **benchmark**, an **online
   monitor**, and a **proxy-reviewer (auditability) study**; plus a
   **polarity-triple schema extension** that makes the checker non-trivial on
   real biomedical contradiction (SciFact) and real fact-checking (FEVER).

**On honesty.** GPT-5's reasoning solves clean synthetic *ordering* emergent
inconsistency at every scale we tried (up to 13-hop cycles in 50 statements); we
report this plainly and do not manufacture a failure there. Instead we *locate*
the real failure: on *numeric* global-sum consistency the same model degrades to
below chance (Section 7.0). The structured checker delivers accuracy where the
LLM fails, and elsewhere **cost, soundness, auditability, and the ability to
make cheap models reliable** — all of which the data support directly.

---

## 2. Related work

- **Deep equilibrium / unrolled optimization** (Bai et al., 2019; SPEN,
  Belanger & McCallum, 2016): differentiable fixed-point layers. Our optional
  equilibrium core is one such layer; our contribution is the *typed,
  parameter-free* energy, the *exact* checker, and the *certificate* — not the
  fixed-point machinery.
- **Neuro-symbolic reasoning** (SATNet; Scallop; DeepProbLog): differentiable
  constraint solving for a label. We instead *extract the minimal unsat core as
  an explanation* and target *emergent inconsistency in LLM assertions*.
- **Fact verification / NLI** (FEVER; SciFact, Wadden et al., 2020; VitaminC):
  predominantly *pairwise* claim-vs-evidence. We show the pairwise framing has a
  provable blind spot on *global* consistency.
- **LLM self-consistency** (SelfCheckGPT): sampling-based agreement. We give a
  *structured, verifiable* judgment with a proof.

---

## 3. Problem: emergent joint inconsistency

Let `S = {s_1, ..., s_n}` be a set of statements. A perception map `P` extracts
a typed constraint set `C = P(S)` over entities `E` with one scalar latent `u_e`
each. Constraint types: `anchor(e,v): u_e=v`; `before(a,b): u_a<u_b`;
`greater(a,b): u_a>u_b`; `sum_eq(a,b,t): u_a+u_b=u_t`. `S` is **jointly
consistent** iff `C` is satisfiable.

**Emergence order** `kappa(S)` is the size of the smallest unsatisfiable subset
of `C`. An inconsistency is *pairwise-detectable* iff `kappa=2` and **emergent**
iff `kappa>=3`. The task: decide consistency and, if inconsistent, return a
minimal unsat core of size `kappa` localized to the responsible statements —
even when `kappa>=3`.

---

## 4. Theory: the separation theorem

**Definition.** A judge is *m-factored* if it declares `S` inconsistent iff some
subset of size `<= m` is inconsistent. Pairwise NLI and evidence-pair checkers
are `m=2`.

**Theorem.** For every `m>=2` there exist statement sets with `|S|=m+1` that are
jointly inconsistent while every subset of size `<= m` is consistent; hence no
`m`-factored judge detects them.

**Proof.** A strict-order cycle `e_0 -> ... -> e_m -> e_0`: every proper subset
omits an edge (acyclic, satisfiable); the full set is a cycle (unsatisfiable).
The cycle length `k=m+1` is the hop-count. ∎

**Scope (honest).** The theorem bounds the pairwise / `m`-factored class. A
frontier LLM that reads all statements at once is in principle `|S|`-factored,
so the theorem does not prove LLMs must fail; that is an empirical question
(Section 7). Empirically we verify the theorem for pairwise methods
(min-unsat-core size equals `kappa` for 100% of inconsistent cases; the
all-pairs judge misses 100% of `kappa>=3` cases).

---

## 5. Method: BeliefNet

**Pipeline.** `statements -> BeliefParser (perception) -> typed constraint graph
-> exact checker (+ certificates) -> decision`.

**Exact checker (soundness).** The order fragment is unsatisfiable iff its `<`
digraph has a directed cycle (DFS); the linear fragment is checked by exact
Gaussian elimination over the rationals. A positive contradiction verdict is
therefore a *proof*, not a score, and **every false positive is a perception
error, never a reasoning error** — localizing all learnable error to a
measurable module.

**Certificates.** `minimal_unsat_core` (QuickXplain over the exact checker)
returns the smallest jointly-unsatisfiable subset; `minimal_correction_set`
returns the smallest retraction that restores satisfiability (a repair).

**Perception.** A `BeliefParser` interface with a deterministic rule parser, an
LLM per-statement extractor (strictly local, so global reasoning is done only by
the checker), and a trained-composite variant.

**Online monitor.** `BeliefMonitor` ingests statements incrementally, flags the
first statement that breaks consistency (localization), and abstains under low
perception confidence.

**Schema extension (polarity triples).** For semantic contradiction (e.g.
biomedical claims), an LLM extracts the *direction* of a claim and, separately,
the direction the evidence establishes for the same relation; the checker flags
opposite polarity via conflicting anchors — reusing the exact checker unchanged
and yielding a structured certificate.

---

## 6. Benchmark

A controlled generator produces balanced consistent/inconsistent cases with
matched length, parameterized by hop-count `kappa` (inconsistent = a length-`k`
order cycle; consistent = an acyclic chain with equal edge count) and statement
count. Three variants (1800 cases total): `canonical` (single template),
`paraphrase` (five surface forms), and `real_seeded` (synthetic cores embedded
among real document sentences, e.g. SciFact abstracts). The exact checker
reproduces the generator labels with accuracy 1.0000 on 800 cases, certifying
the labels and the emergence construction.

---

## 7. Experiments and results

All numbers are real (seed 0). LLM numbers are live Azure GPT-4o / GPT-5 calls.

### 7.0 Headline: numeric emergent inconsistency breaks a frontier reasoning model

We stress a distribution where inconsistency is *emergent and numeric*:
running-balance cases (opening amount + `depth` signed transactions + a stated
final balance), consistent iff the final equals the opening plus the sum. No
pair of statements is a witness; only the global sum decides. AUROC by depth
(150 cases/model, 300 live calls):

| depth (terms) | oracle / BeliefNet | GPT-4o | GPT-5 |
|---|---|---|---|
| 5  | 1.000 | 0.580 | 1.000 |
| 12 | 1.000 | 0.524 | 1.000 |
| 20 | 1.000 | 0.489 | 0.867 |
| 28 | 1.000 | 0.489 | 0.480 |
| 36 | 1.000 | 0.642 | **0.233** |

Overall (95% CI): oracle **1.000**; GPT-5 **0.679 [0.578, 0.779]**; GPT-4o
0.543 [0.466, 0.621]. **GPT-5's reasoning solves ordering cycles at any scale
(§7.2) yet collapses monotonically to below chance on numeric emergent
inconsistency as the term count grows** — perfect at <=12 terms, chance at 28,
and systematically wrong (0.233) at 36. GPT-4o is at chance throughout. The
sound checker stays at 1.000 and returns an arithmetic certificate. This is the
emergent-inconsistency blind spot on the distribution where even a frontier
reasoning model cannot compensate.

### 7.1 The separation, empirically (Exp 2, Exp 4)

Minimal unsat-core size equals `kappa` for **100%** of inconsistent cases; the
pairwise-miss rate on emergent (`kappa>=3`) cases is **1.000**. A concrete
all-pairs judge attains AUROC **1.0 / 0.5 / 0.5** at hop-count 2 / 3 / 5
(overall 0.667, 95% CI [0.640, 0.694]) — perfect at `k=2`, exactly chance for
`k>=3`.

### 7.2 LLM failure curve (Exp 1)

Mean AUROC by hop-count (96 cases/model, 192 live calls):

| hop | oracle | GPT-4o | GPT-5 |
|---|---|---|---|
| 2 | 1.000 | 0.976 | 1.000 |
| 3 | 1.000 | 0.840 | 1.000 |
| 5 | 1.000 | 0.667 | 1.000 |
| 7 | 1.000 | 0.587 | 1.000 |

GPT-4o (non-reasoning) collapses toward chance as hop-count grows. **GPT-5
(reasoning) holds at 1.000**, and continues to hold at **13-hop cycles buried in
50 paraphrased statements** (stress run). We report this honestly: on clean,
explicit *ordering* constraints the reasoning model does not break at the scales
we tested — which is exactly why the *numeric* headline (§7.0), where the same
model collapses to 0.233, isolates the real blind spot.

### 7.3 Cost, soundness, auditability (Exp: cost comparison)

Same cases, per-decision latency and accuracy (real):

| system | latency/decision | accuracy | certificate |
|---|---|---|---|
| **BeliefNet (exact)** | **0.92 ms** | **1.000** | **yes** (unsat core + repair) |
| GPT-4o judge | 2,267 ms | 0.722 | no |
| GPT-5 judge | 5,691 ms | 1.000 | no |

BeliefNet matches GPT-5 accuracy at **~6,193× lower latency** and returns a
proof; GPT-5 returns an opaque verdict in 5.7 s.

### 7.4 Decompose-and-verify: the self-fix (Exp 7)

AUROC by hop-count for GPT-4o (real, ~270 live calls):

| hop | GPT-4o judge (end-to-end) | GPT-4o extract→checker |
|---|---|---|
| 3 | 1.000 | 1.000 |
| 5 | 0.800 | 0.900 |
| 7 | **0.500** | **0.900** |

Routing GPT-4o's *local* extractions through the checker lifts AUROC by **+0.40
at hop 7**, recovering reliability a cheap model lacks end-to-end. It is 0.9 not
1.0 because GPT-4o's extraction is imperfect — the residual error is
*perception*, exactly where the framework localizes it.

### 7.5 Certificates and online monitoring (Exp 8, P4)

Every flagged contradiction yields a valid minimal unsat core and a valid
minimal repair (**both 1.000** over 180 cases; core size = `kappa`, repair size
= 1). Streamed through the online monitor, detection AUROC and statement-level
localization accuracy are **1.000** at every hop-count.

### 7.6 Real data: SciFact and FEVER

On the real SciFact dev split (CONTRADICT vs SUPPORT):

| system | AUROC | accuracy | F1(contradict) |
|---|---|---|---|
| GPT-5 judge | 0.988 | 0.950 | 0.950 |
| GPT-4o judge | 0.966 | 0.938 | 0.937 |
| BeliefNet (base schema) | 0.500 | 0.500 | 0.000 |
| BeliefNet (polarity + GPT-4o extractor) | 0.838 | 0.838 | 0.812 |
| BeliefNet (polarity + GPT-5 extractor) | 0.825 | 0.825 | 0.800 |

The base `before/greater/anchor/sum_eq` schema does not cover biomedical
semantic contradiction and scores at chance — an honest schema-breadth
limitation localized to perception. The **polarity-triple extension** (LLM
extracts claim/evidence direction independently; checker compares) lifts AUROC
from 0.500 to **0.838** while emitting a structured certificate — still below
the 0.99 raw judge because reducing a claim to a single direction is lossy, but
interpretable and certified, and it demonstrates that schema extension closes
the gap on real text.

**Replication on FEVER.** The same picture holds on a second real corpus (FEVER
gold-evidence, 80 balanced SUPPORTS/REFUTES): the base schema is at chance
(AUROC 0.500), the LLM judges are strong (GPT-4o 0.981, GPT-5 0.983), and the
polarity extension lifts the checker to **0.775 (GPT-4o) / 0.888 (GPT-5)** with
a structured certificate — external validity for the schema-extension result.

### 7.7 Auditability: a proxy-reviewer study

Is the certificate *usable*? A reviewer model (Claude Haiku) judges hard numeric
cases under three aids; accuracy by condition and difficulty (30 cases each,
seed 0):

| condition | depth 15 | depth 40 |
|---|---|---|
| reviewer alone | 0.933 | 0.500 |
| **reviewer + sound certificate** | **1.000** | **1.000** |
| reviewer + GPT-4o rationale | 0.867 | 0.500 |
| (ref) GPT-4o assistant alone | 0.500 | 0.433 |

The certificate takes the reviewer to **perfect at both depths**. At depth 40
the unaided reviewer is at chance and GPT-4o is *below* chance (its rationale
carries no signal), yet the certificate still yields 1.000 — a **+0.50** rescue
that grows with difficulty. A fluent-but-wrong LLM rationale can even *lower*
reviewer accuracy (0.933 → 0.867 at depth 15; automation bias). Auditability is
a measured benefit, not a claim.

---

## 8. Discussion and limitations

- **Where LLMs fail, and where they don't.** GPT-5 solves clean synthetic
  *ordering* inconsistency and excels at pairwise SciFact, but **collapses to
  below chance on numeric emergent inconsistency** (§7.0) — a measured frontier
  blind spot. Beyond accuracy, the contributions are cost, soundness,
  auditability, and cheap-model rescue.
- **Schema breadth is the real limitation.** The base four-relation schema is a
  deliberate, checkable fragment; real semantic contradiction needs extensions
  (polarity triples here; more families are future work). The framework makes
  the gap *measurable and localized to perception*.
- **LLM-as-extractor reproducibility.** When an LLM is the extractor, results
  depend on a proprietary API; an open-model extractor should be reported so the
  method is not API-locked.
- **Equilibrium layer (a trainable variant).** The optional differentiable
  6-parameter core ties the exact checker on separability (exp5: exact = energy
  = 1.000) and, trained on the synthetic ladder, generalizes to **0.832 AUROC on
  held-out paraphrase** — evidence the energy layer is trainable and shift-robust.
  It is justified by that trainability, not by beating the exact checker (which
  remains the sound, certificate-emitting path).

---

## 9. Conclusion

Emergent joint inconsistency provably defeats pairwise methods and empirically
defeats non-reasoning LLMs — and, on numeric global-sum conflicts, defeats a
frontier reasoning model too (GPT-5 falls to 0.233, below chance, as the term
count grows). A sound checker decides all of it in ~1 ms at 1.000, ~6000×
cheaper than a GPT-5 call, certifies and repairs its verdicts, localizes
contradictions online, rescues cheap models via decompose-and-verify (+0.40),
and — in a proxy-reviewer study — its certificate rescues a reviewer from chance
to perfect where an LLM rationale cannot. This reframes consistency verification
around cost, soundness, and auditability, and pinpoints a concrete distribution
where even frontier reasoning needs it.

---

## References (to complete)

- D. Wadden et al. Fact or Fiction: Verifying Scientific Claims. EMNLP 2020.
- S. Bai, J. Z. Kolter, V. Koltun. Deep Equilibrium Models. NeurIPS 2019.
- D. Belanger, A. McCallum. Structured Prediction Energy Networks. ICML 2016.
- P.-W. Wang et al. SATNet. ICML 2019.
- J. Thorne et al. FEVER. NAACL 2018.
- P. Manakul et al. SelfCheckGPT. EMNLP 2023.
- U. Junker. QuickXplain. AAAI 2004.

*(Full BibTeX and additional neuro-symbolic / consistency citations to be added.)*
