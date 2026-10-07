"""Evaluate a trained BeliefNet synthetic pipeline from saved checkpoints.

Loads ``core.pt``, ``perception.pt``, ``span.pt`` from the experiment
directory written by ``scripts/train_synthetic.py`` and re-runs the eval
loop on the configured test sets. The training script does NOT currently
serialise checkpoints; until it does, this script prints a clear message
and exits 1. The wiring is here so checkpoint saving can be flipped on in
one place when needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from beliefnet.data.features import encode_texts
from beliefnet.data.synthetic.case_schema import MultiStmtConfig
from beliefnet.data.synthetic.multi_statement import generate
from beliefnet.data.synthetic.ood import OodConfig, generate_ood
from beliefnet.data.synthetic.templates import (
    AUGMENTED_TEMPLATES,
    CANONICAL_TEMPLATES,
    HELD_OUT_TEMPLATES,
)
from beliefnet.encoders.frozen_backbone import FrozenBackbone, FrozenBackboneConfig
from beliefnet.graph.constraint_types import STATEMENT_TYPES
from beliefnet.models.beliefnet import StructuredEquilibriumCore
from beliefnet.perception.relation_prototypes import RELATION_PROTOTYPES
from beliefnet.perception.span_heads import SpanGroundedParser
from beliefnet.perception.type_heads import HybridRelationParser
from beliefnet.utils.configs import apply_overrides, load_yaml
from beliefnet.utils.seed import set_seed

# Reuse the training script's helpers so the inference path is identical.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_synthetic import (  # noqa: E402
    TokenBank,
    encode_pooled,
    evaluate,
    split_cases,
)

_TEMPLATE_SETS = {
    "canonical": CANONICAL_TEMPLATES,
    "augmented": AUGMENTED_TEMPLATES,
    "held_out": HELD_OUT_TEMPLATES,
}

_REQUIRED_CHECKPOINTS = ("core.pt", "perception.pt", "span.pt")


def _generate_split(data_cfg, key, base_seed):
    spec = data_cfg[key]
    templates = _TEMPLATE_SETS[spec["templates"]]
    if key == "base":
        return generate(MultiStmtConfig(
            n_per_subtype=data_cfg["n_per_subtype"],
            seed=base_seed,
            ratios=tuple(data_cfg["ratios"]),
        ))
    return generate_ood(OodConfig(
        n_per_subtype=data_cfg["n_per_subtype"],
        seed=base_seed + spec.get("seed_offset", 0),
        ratios=tuple(data_cfg["ratios"]),
        templates=templates,
        id_prefix=spec["id_prefix"],
    ))


def run(model_cfg, data_cfg, eval_cfg, ckpt_dir, out_dir, seed):
    ckpt_dir = Path(ckpt_dir)
    missing = [f for f in _REQUIRED_CHECKPOINTS if not (ckpt_dir / f).exists()]
    if missing:
        print(f"[eval_synthetic] missing checkpoints in {ckpt_dir}: {missing}")
        print("  (scripts/train_synthetic.py does not yet save checkpoints; "
              "until it does, this evaluator cannot run standalone.)")
        return 1

    set_seed(seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    backbone = FrozenBackbone(FrozenBackboneConfig(**model_cfg["backbone"]))

    base_cases = _generate_split(data_cfg, "base", seed)
    held_out_cases = _generate_split(data_cfg, "held_out", seed)
    base_splits = split_cases(base_cases)
    paraphrase_test = [c for c in held_out_cases if c.split == "test"]

    all_stmts = (
        [s for c in base_cases for s in c.statements]
        + [s for c in held_out_cases for s in c.statements]
    )
    stmt_to_emb: dict[str, torch.Tensor] = {}
    encode_pooled(stmt_to_emb, all_stmts, backbone)
    H = next(iter(stmt_to_emb.values())).shape[0]
    token_bank = TokenBank()
    token_bank.add(all_stmts, backbone, batch=32)

    proto_embs = {r: encode_texts(backbone, RELATION_PROTOTYPES[r]).cpu()
                  for r in STATEMENT_TYPES}

    core = StructuredEquilibriumCore(**model_cfg["equilibrium_core"]).to(device)
    core.load_state_dict(torch.load(ckpt_dir / "core.pt", map_location=device))
    core.eval()

    pcfg = model_cfg["parser"]
    perc = HybridRelationParser(H, proto_embs, **{k: pcfg[k] for k in
        ("mlp_hidden", "dropout", "temperature_init", "trainable_prototypes")}).to(device)
    perc.load_state_dict(torch.load(ckpt_dir / "perception.pt", map_location=device))
    perc.eval()

    scfg = model_cfg["span_head"]
    tagger = SpanGroundedParser(hidden_size=H,
                                mlp_hidden=scfg["mlp_hidden"],
                                dropout=scfg["dropout"]).to(device)
    tagger.load_state_dict(torch.load(ckpt_dir / "span.pt", map_location=device))
    tagger.eval()

    sources = {"base": base_splits, "held_out": {"test": paraphrase_test}}
    test_sets = {spec["name"]: sources[spec["source"]][spec["split"]]
                 for spec in eval_cfg["test_sets"]}

    summary = {}
    for t_name, t_cases in test_sets.items():
        rep = evaluate(perc, core, t_cases, stmt_to_emb, device,
                       tagger=tagger, token_bank=token_bank)
        summary[t_name] = {
            "n": rep["n_test"],
            "type_accuracy": rep["type_accuracy"],
            "overall_auroc": rep["overall_auroc"],
            "accuracy": rep["accuracy"],
            "matched_auroc": rep["matched_auroc"],
            "relation_metrics": rep["relation_metrics"],
            "span_f1": rep["span_f1"],
        }
        print(f"  {t_name:25s}  type_acc={rep['type_accuracy']:.3f}  "
              f"AUROC={rep['overall_auroc']:.3f}")

    with open(out_dir / "eval_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"[eval_synthetic] wrote {out_dir}/eval_summary.json")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="configs/model/paraphrase_robust.yaml")
    ap.add_argument("--data-config", default="configs/data/synthetic_augmented.yaml")
    ap.add_argument("--eval-config", default="configs/eval/in_dist.yaml")
    ap.add_argument("--ckpt-dir", required=True,
                    help="directory holding core.pt, perception.pt, span.pt")
    ap.add_argument("--out", default="experiments/beliefnet_synthetic_eval")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--override", nargs="*", default=[])
    args = ap.parse_args()

    model_cfg = apply_overrides(load_yaml(args.model_config), args.override)
    data_cfg = apply_overrides(load_yaml(args.data_config), args.override)
    eval_cfg = apply_overrides(load_yaml(args.eval_config), args.override)
    sys.exit(run(model_cfg, data_cfg, eval_cfg, args.ckpt_dir, args.out, args.seed))


if __name__ == "__main__":
    main()
