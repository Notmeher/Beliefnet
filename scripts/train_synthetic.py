"""Train the BeliefNet synthetic pipeline (paraphrase-robust composite).

Config-driven entry point that replaces the per-version ``train_een_v*.py``
scripts. Targets ONLY the final pipeline (``HybridRelationParser`` +
``SpanGroundedParser`` + regex numeric extractor + ``StructuredEquilibriumCore``),
mirroring the EEN-v9 setup that closed the synthetic ladder.

Default usage:

    PYTHONPATH=src python scripts/train_synthetic.py \\
        --model-config configs/model/paraphrase_robust.yaml \\
        --data-config  configs/data/synthetic_augmented.yaml \\
        --train-config configs/train/synthetic_equilibrium.yaml \\
        --eval-config  configs/eval/in_dist.yaml
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from beliefnet.data.features import encode_texts
from beliefnet.data.synthetic.case_schema import Case, MultiStmtConfig
from beliefnet.data.synthetic.multi_statement import generate
from beliefnet.data.synthetic.ood import OodConfig, generate_ood
from beliefnet.data.synthetic.templates import (
    AUGMENTED_TEMPLATES,
    CANONICAL_TEMPLATES,
    HELD_OUT_TEMPLATES,
)
from beliefnet.data.synthetic.vocabulary import (
    DIAGNOSES,
    DRUGS,
    NAMES,
    ORDER_EVENTS,
    SYMPTOMS,
)
from beliefnet.encoders.frozen_backbone import FrozenBackbone, FrozenBackboneConfig
from beliefnet.eval.metrics import auroc
from beliefnet.eval.structured_eval import evaluate_from_scores
from beliefnet.graph.constraint_types import STATEMENT_TYPES
from beliefnet.graph.graph_builder import build_graph_from_supervisions
from beliefnet.graph.graph_utils import pack_graphs
from beliefnet.models.beliefnet import StructuredEquilibriumCore
from beliefnet.perception.numeric_extractors import hybrid_value
from beliefnet.perception.parser_rules import (
    decode_bio_spans,
    derive_bio_labels,
    derive_supervision,
    predicted_supervision,
)
from beliefnet.perception.relation_prototypes import RELATION_PROTOTYPES
from beliefnet.perception.span_heads import (
    SpanGroundedParser,
    extract_categorical,
    extract_relation_args,
)
from beliefnet.perception.type_heads import (
    HybridRelationParser,
    perception_loss,
    supervised_contrastive_loss,
)
from beliefnet.utils.configs import apply_overrides, load_yaml
from beliefnet.utils.seed import set_seed

NUMERIC_TYPES = ("arm_count", "total_count", "direct_enrolled")

_TEMPLATE_SETS = {
    "canonical": CANONICAL_TEMPLATES,
    "augmented": AUGMENTED_TEMPLATES,
    "held_out": HELD_OUT_TEMPLATES,
}


# ============================================================ data helpers
def split_cases(cases):
    return {s: [c for c in cases if c.split == s] for s in ("train", "val", "test")}


def encode_pooled(stmt_to_emb, texts, backbone, batch=64):
    new = sorted({t for t in texts if t not in stmt_to_emb})
    if not new:
        return
    embs = encode_texts(backbone, new, batch).cpu()
    for t, e in zip(new, embs):
        stmt_to_emb[t] = e


class TokenBank:
    def __init__(self):
        self.embs: dict[str, torch.Tensor] = {}
        self.offsets: dict[str, list[tuple[int, int]]] = {}
        self.bio: dict[str, list[int]] = {}

    @torch.no_grad()
    def add(self, texts, backbone, batch=32):
        new = sorted({t for t in texts if t not in self.embs})
        if not new:
            return
        for i in range(0, len(new), batch):
            chunk = new[i:i + batch]
            enc = backbone.tokenizer(
                chunk, return_tensors="pt", padding=True, truncation=True,
                max_length=backbone.config.max_length,
                return_offsets_mapping=True,
            )
            mask = enc["attention_mask"]
            offsets = enc["offset_mapping"]
            ids = enc["input_ids"].to(backbone.device)
            attn = mask.to(backbone.device)
            out = backbone.model(input_ids=ids, attention_mask=attn,
                                 output_hidden_states=True)
            hidden = out.hidden_states[backbone.config.layer].float().cpu()
            for j, t in enumerate(chunk):
                length = int(mask[j].sum())
                self.embs[t] = hidden[j, :length].clone()
                self.offsets[t] = [
                    (int(offsets[j, k, 0]), int(offsets[j, k, 1])) for k in range(length)
                ]
                sup = derive_supervision(t)
                self.bio[t] = derive_bio_labels(t, sup, self.offsets[t])

    def gather(self, texts, device):
        embs = [self.embs[t] for t in texts]
        T = max(e.shape[0] for e in embs)
        B, H = len(texts), embs[0].shape[1]
        out_e = torch.zeros(B, T, H, device=device)
        out_m = torch.zeros(B, T, dtype=torch.bool, device=device)
        out_l = torch.full((B, T), -100, dtype=torch.long, device=device)
        for i, t in enumerate(texts):
            e = embs[i]
            bio = self.bio[t]
            out_e[i, :e.shape[0]] = e.to(device)
            out_m[i, :e.shape[0]] = True
            out_l[i, :len(bio)] = torch.tensor(bio, dtype=torch.long, device=device)
        return out_e, out_m, out_l


def make_perc_dataset(cases, stmt_to_emb):
    from beliefnet.graph.constraint_types import supervision_to_targets
    embs, sups, texts = [], [], []
    targets = {k: [] for k in [
        "type", "order_event1", "order_event2", "name1", "name2",
        "symptom", "diagnosis", "drug", "arm", "value", "value_mask",
    ]}
    for case in cases:
        for s in case.statements:
            sup = derive_supervision(s)
            sups.append(sup)
            texts.append(s)
            tgt = supervision_to_targets(sup)
            embs.append(stmt_to_emb[s])
            for k in targets:
                targets[k].append(tgt[k])
    emb_t = torch.stack(embs).float()
    out = {}
    for k, vs in targets.items():
        dtype = torch.float if k in ("value", "value_mask") else torch.long
        out[k] = torch.tensor(vs, dtype=dtype)
    return emb_t, out, sups, texts


# ============================================================ training loops
def train_equilibrium(model, specs, labels, epochs, lr):
    pos = labels["train"].sum()
    neg = labels["train"].numel() - pos
    pos_w = neg / pos.clamp_min(1.0)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_w)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best, best_state = -1.0, {k: v.clone() for k, v in model.state_dict().items()}
    for _ in range(epochs):
        model.train()
        opt.zero_grad()
        loss_fn(model(specs["train"]), labels["train"]).backward()
        opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.sigmoid(model(specs["val"])).cpu().numpy()
        va_auc = auroc(va, labels["val"].cpu().numpy())
        if va_auc > best:
            best, best_state = va_auc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best


def train_perception(perc, tr_emb, tr_tgt, va_emb, va_tgt, epochs, lr, batch,
                     contrastive_w=0.1, aux_w=0.3):
    opt = torch.optim.Adam(perc.parameters(), lr=lr, weight_decay=1e-4)
    n = tr_emb.shape[0]
    best, best_state = -1.0, {k: v.clone() for k, v in perc.state_dict().items()}
    for _ in range(epochs):
        perc.train()
        perm = torch.randperm(n, device=tr_emb.device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            out = perc(tr_emb[idx], return_hidden=True, return_gate=True)
            hidden = out.pop("_hidden")
            a_logits = out.pop("_a_logits")
            b_logits = out.pop("_b_logits")
            _ = out.pop("_gate")
            tgt = {k: v[idx] for k, v in tr_tgt.items()}
            ce_loss, _ = perception_loss(out, tgt)
            aux = F.cross_entropy(a_logits, tgt["type"]) + F.cross_entropy(b_logits, tgt["type"])
            con_loss = supervised_contrastive_loss(hidden, tgt["type"])
            loss = ce_loss + aux_w * aux + contrastive_w * con_loss
            loss.backward()
            opt.step()
        perc.eval()
        with torch.no_grad():
            vl = perc(va_emb)
            va = float((vl["type"].argmax(-1) == va_tgt["type"]).float().mean())
        if va > best:
            best, best_state = va, {k: v.clone() for k, v in perc.state_dict().items()}
    perc.load_state_dict(best_state)
    return best


def train_span_tagger(tagger, texts, token_bank, epochs, lr, batch, device,
                      val_texts):
    opt = torch.optim.Adam(tagger.parameters(), lr=lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
    n = len(texts)
    best, best_state = -1.0, {k: v.clone() for k, v in tagger.state_dict().items()}
    for _ in range(epochs):
        tagger.train()
        perm = torch.randperm(n).tolist()
        for s in range(0, n, batch):
            chunk = [texts[i] for i in perm[s:s + batch]]
            embs, mask, labels = token_bank.gather(chunk, device)
            opt.zero_grad()
            logits = tagger(embs)
            loss = loss_fn(logits.view(-1, logits.size(-1)), labels.view(-1))
            loss.backward()
            opt.step()
        tagger.eval()
        with torch.no_grad():
            embs, mask, labels = token_bank.gather(val_texts, device)
            logits = tagger(embs)
            preds = logits.argmax(-1)
            valid = labels != -100
            acc = float(((preds == labels) & valid).sum().float() / valid.sum().clamp_min(1.0))
        if acc > best:
            best, best_state = acc, {k: v.clone() for k, v in tagger.state_dict().items()}
    tagger.load_state_dict(best_state)
    return best


# ============================================================ inference glue
def hybrid_supervision_with_spans(text, neural_sup, tag_pred, offsets,
                                  event_vocab, name_vocab):
    t = neural_sup.get("type")
    if t == "before":
        e1, e2 = extract_relation_args(text, tag_pred, offsets,
                                       "before", event_vocab, NAMES)
        if e1 and e2:
            return {**neural_sup, "entity1": e1, "entity2": e2}
    elif t == "greater":
        e1, e2 = extract_relation_args(text, tag_pred, offsets,
                                       "greater", ORDER_EVENTS, name_vocab)
        if e1 and e2:
            return {**neural_sup, "entity1": e1, "entity2": e2}
    elif t in ("presence_pos", "presence_neg"):
        s = extract_categorical(text, tag_pred, offsets, "SYMPTOM", SYMPTOMS)
        if s:
            return {**neural_sup, "symptom": s}
    elif t == "diagnosis":
        d = extract_categorical(text, tag_pred, offsets, "DIAGNOSIS", DIAGNOSES)
        if d:
            return {**neural_sup, "diagnosis": d}
    elif t == "drug_active":
        d = extract_categorical(text, tag_pred, offsets, "DRUG", DRUGS)
        if d:
            return {**neural_sup, "drug": d}
    return neural_sup


def predict_tags_for(texts, tagger, token_bank, device, batch=64):
    out: dict[str, list[int]] = {}
    for i in range(0, len(texts), batch):
        chunk = texts[i:i + batch]
        embs, mask, _ = token_bank.gather(chunk, device)
        with torch.no_grad():
            logits = tagger(embs)
            preds = logits.argmax(-1).cpu().numpy()
        for j, t in enumerate(chunk):
            length = sum(1 for m in mask[j].cpu().numpy() if m)
            out[t] = [int(x) for x in preds[j, :length]]
    return out


# ============================================================ evaluation
def evaluate(perc, core, test_cases, stmt_to_emb, device,
             tagger, token_bank, event_vocab=None, name_vocab=None):
    test_emb, test_tgt, test_truth, test_texts = make_perc_dataset(test_cases, stmt_to_emb)
    test_emb = test_emb.to(device)

    perc.eval()
    with torch.no_grad():
        decoded = perc.decode(test_emb)
    type_true = test_tgt["type"].numpy()
    type_pred = np.array([d["type"] for d in decoded])
    type_acc = float((type_pred == type_true).mean())

    R = len(STATEMENT_TYPES)
    confusion = np.zeros((R, R), dtype=int)
    for t, p in zip(type_true, type_pred):
        confusion[int(t), int(p)] += 1

    ev = event_vocab or ORDER_EVENTS
    nv = name_vocab or NAMES
    tag_preds = predict_tags_for(test_texts, tagger, token_bank, device)
    final_sups = []
    for text, d in zip(test_texts, decoded):
        sup = predicted_supervision(d)
        tags = tag_preds.get(text, [])
        offs = token_bank.offsets.get(text, [])
        sup = hybrid_supervision_with_spans(text, sup, tags, offs, ev, nv)
        if sup.get("type") in NUMERIC_TYPES:
            sup = {**sup, "value": hybrid_value(text,
                                                neural_value=float(d.get("value", 0.0)))}
        final_sups.append(sup)

    n_rel = n_arg_correct = n_order_correct = 0
    for true_sup, pred_sup in zip(test_truth, final_sups):
        if true_sup.get("type") in ("before", "greater"):
            n_rel += 1
            e1_ok = (true_sup.get("entity1") == pred_sup.get("entity1"))
            e2_ok = (true_sup.get("entity2") == pred_sup.get("entity2"))
            if e1_ok and e2_ok:
                n_arg_correct += 1
                n_order_correct += 1
    rel_metrics = {
        "n_relation_statements": n_rel,
        "both_entities_accuracy": round(n_arg_correct / max(n_rel, 1), 4),
        "argument_order_accuracy": round(n_order_correct / max(n_rel, 1), 4),
    }

    tp = defaultdict(int)
    fp = defaultdict(int)
    fn = defaultdict(int)
    for text in test_texts:
        true_tags = token_bank.bio.get(text, [])
        pred_tags = tag_preds.get(text, [])
        offs = token_bank.offsets.get(text, [])
        true_spans = decode_bio_spans(text, true_tags, offs)
        pred_spans = decode_bio_spans(text, pred_tags, offs)
        true_set = {(s["type"], s["text"].lower()) for s in true_spans}
        pred_set = {(s["type"], s["text"].lower()) for s in pred_spans}
        for k in true_set & pred_set:
            tp[k[0]] += 1
        for k in pred_set - true_set:
            fp[k[0]] += 1
        for k in true_set - pred_set:
            fn[k[0]] += 1
    span_f1 = {}
    for ent in ("EVENT", "NAME", "SYMPTOM", "DIAGNOSIS", "DRUG"):
        p = tp[ent] / max(tp[ent] + fp[ent], 1)
        r = tp[ent] / max(tp[ent] + fn[ent], 1)
        f = 2 * p * r / max(p + r, 1e-9)
        span_f1[ent] = {"precision": round(p, 4), "recall": round(r, 4),
                        "f1": round(f, 4), "tp": tp[ent], "fp": fp[ent], "fn": fn[ent]}
    ttp, tfp, tfn = sum(tp.values()), sum(fp.values()), sum(fn.values())
    mp = ttp / max(ttp + tfp, 1)
    mr = ttp / max(ttp + tfn, 1)
    span_f1["micro"] = {"precision": round(mp, 4), "recall": round(mr, 4),
                        "f1": round(2 * mp * mr / max(mp + mr, 1e-9), 4)}

    offset = 0
    pred_cases = []
    for case in test_cases:
        n = len(case.statements)
        ents, constraints = build_graph_from_supervisions(final_sups[offset:offset + n])
        offset += n
        pred_cases.append(Case(
            id=case.id, statements=case.statements, label=case.label,
            conflict_kind=case.conflict_kind, conflict_subtype=case.conflict_subtype,
            conflict_indices=case.conflict_indices,
            metadata={"n_statements": n, "entities": ents, "constraints": constraints},
            split=case.split,
        ))
    pred_graph = pack_graphs(pred_cases).to(device)
    with torch.no_grad():
        scores = torch.sigmoid(core(pred_graph)).cpu().numpy()
        res = core.per_case_residuals(pred_graph)["total"].detach().cpu().numpy()
    metrics = evaluate_from_scores(scores, test_cases)

    fail = []
    for true_t, pred_t, sup_true, sup_pred, text in zip(
            type_true.tolist(), type_pred.tolist(),
            test_truth, final_sups, test_texts):
        type_ok = pred_t == true_t
        ent_ok = True
        if sup_true.get("type") in ("before", "greater"):
            ent_ok = (sup_true.get("entity1") == sup_pred.get("entity1") and
                      sup_true.get("entity2") == sup_pred.get("entity2"))
        if not type_ok or not ent_ok:
            fail.append({
                "text": text,
                "true_type": STATEMENT_TYPES[int(true_t)],
                "pred_type": STATEMENT_TYPES[int(pred_t)],
                "true_sup": sup_true, "pred_sup": sup_pred,
                "type_correct": type_ok, "entities_correct": ent_ok,
            })

    return {
        "type_accuracy": round(type_acc, 4),
        "overall_auroc": metrics["overall_auroc"],
        "accuracy": metrics["accuracy"],
        "matched_auroc": metrics["matched_auroc"],
        "flag_rate_by_subtype": metrics["flag_rate_by_subtype"],
        "confusion": confusion.tolist(),
        "relation_metrics": rel_metrics,
        "span_f1": span_f1,
        "residuals": res.tolist(),
        "subtypes": [c.conflict_subtype for c in test_cases],
        "labels": [c.label for c in test_cases],
        "failure_examples": fail[:30],
        "n_test": len(test_cases),
    }


# ============================================================ pipeline build
def _build_backbone(model_cfg):
    bb = FrozenBackboneConfig(**model_cfg["backbone"])
    return FrozenBackbone(bb)


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


def run(model_cfg, data_cfg, train_cfg, eval_cfg, out_dir):
    set_seed(train_cfg["seed"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    backbone = _build_backbone(model_cfg)

    print("[data] generating canonical / augmented / held-out paraphrase sets")
    base_cases = _generate_split(data_cfg, "base", train_cfg["seed"])
    aug_cases = _generate_split(data_cfg, "augmented", train_cfg["seed"])
    held_out_cases = _generate_split(data_cfg, "held_out", train_cfg["seed"])
    base_splits = split_cases(base_cases)
    aug_splits = split_cases(aug_cases)
    paraphrase_test = [c for c in held_out_cases if c.split == "test"]
    print(f"  base train={len(base_splits['train'])} aug train={len(aug_splits['train'])}"
          f" paraphrase test={len(paraphrase_test)}")

    all_stmts = (
        [s for c in base_cases for s in c.statements]
        + [s for c in aug_cases for s in c.statements]
        + [s for c in held_out_cases for s in c.statements]
    )
    stmt_to_emb: dict[str, torch.Tensor] = {}
    encode_pooled(stmt_to_emb, all_stmts, backbone)
    H = next(iter(stmt_to_emb.values())).shape[0]
    print(f"  pooled cache: {len(stmt_to_emb)} unique statements, dim={H}")

    print("[token] encoding token-level Qwen states + offsets + BIO labels")
    token_bank = TokenBank()
    token_bank.add(all_stmts, backbone, batch=32)

    print("[proto] encoding relation prototypes")
    proto_embs: dict[str, torch.Tensor] = {}
    for r in STATEMENT_TYPES:
        proto_embs[r] = encode_texts(backbone, RELATION_PROTOTYPES[r]).cpu()

    print("[core] training and freezing StructuredEquilibriumCore")
    eq_cfg = model_cfg["equilibrium_core"]
    specs = {s: pack_graphs(base_splits[s]).to(device) for s in base_splits}
    labels = {s: torch.tensor(
        [1.0 if c.label == "inconsistent" else 0.0 for c in base_splits[s]], device=device)
        for s in base_splits}
    core = StructuredEquilibriumCore(**eq_cfg).to(device)
    core_val = train_equilibrium(core, specs, labels,
                                 train_cfg["equilibrium"]["epochs"],
                                 train_cfg["equilibrium"]["lr"])
    core.requires_grad_(False)
    core.eval()
    print(f"  StructuredEquilibriumCore frozen, val AUROC = {core_val:.4f}")

    print("[perception] training HybridRelationParser on AUGMENTED")
    tr_emb, tr_tgt, _, _ = make_perc_dataset(aug_splits["train"], stmt_to_emb)
    va_emb, va_tgt, _, _ = make_perc_dataset(aug_splits["val"], stmt_to_emb)
    tr_emb = tr_emb.to(device)
    tr_tgt = {k: v.to(device) for k, v in tr_tgt.items()}
    va_emb = va_emb.to(device)
    va_tgt = {k: v.to(device) for k, v in va_tgt.items()}
    pcfg = model_cfg["parser"]
    perc = HybridRelationParser(H, proto_embs,
                                mlp_hidden=pcfg["mlp_hidden"],
                                dropout=pcfg["dropout"],
                                temperature_init=pcfg["temperature_init"],
                                trainable_prototypes=pcfg["trainable_prototypes"]).to(device)
    set_seed(train_cfg["seed"])
    val_acc = train_perception(perc, tr_emb, tr_tgt, va_emb, va_tgt,
                               train_cfg["perception"]["epochs"],
                               train_cfg["perception"]["lr"],
                               train_cfg["perception"]["batch_size"],
                               contrastive_w=train_cfg["perception"]["contrastive_w"],
                               aux_w=train_cfg["perception"]["aux_w"])
    print(f"  HybridRelationParser val type_acc = {val_acc:.4f}")

    print("[span] training SpanGroundedParser on AUGMENTED")
    aug_train_texts = list({s for c in aug_splits["train"] for s in c.statements})
    aug_val_texts = list({s for c in aug_splits["val"] for s in c.statements})
    scfg = model_cfg["span_head"]
    tagger = SpanGroundedParser(hidden_size=H,
                                mlp_hidden=scfg["mlp_hidden"],
                                dropout=scfg["dropout"]).to(device)
    set_seed(train_cfg["seed"])
    span_val_acc = train_span_tagger(tagger, aug_train_texts, token_bank,
                                     train_cfg["span"]["epochs"],
                                     train_cfg["span"]["lr"],
                                     train_cfg["span"]["batch_size"],
                                     device, aug_val_texts)
    print(f"  SpanGroundedParser val per-token accuracy = {span_val_acc:.4f}")

    sources = {"base": base_splits, "held_out": {"test": paraphrase_test}}
    test_sets = {}
    for spec in eval_cfg["test_sets"]:
        test_sets[spec["name"]] = sources[spec["source"]][spec["split"]]

    print("\n[eval]")
    results, summary = {}, {}
    for t_name, t_cases in test_sets.items():
        rep = evaluate(perc, core, t_cases, stmt_to_emb, device,
                       tagger=tagger, token_bank=token_bank)
        results[t_name] = rep
        summary[t_name] = {
            "n": rep["n_test"],
            "type_accuracy": rep["type_accuracy"],
            "overall_auroc": rep["overall_auroc"],
            "accuracy": rep["accuracy"],
            "matched_auroc": rep["matched_auroc"],
            "relation_metrics": rep["relation_metrics"],
            "span_f1": rep["span_f1"],
        }
        rel = rep["relation_metrics"]
        print(f"  {t_name:25s}  type_acc={rep['type_accuracy']:.3f}  "
              f"AUROC={rep['overall_auroc']:.3f}  "
              f"ent_acc={rel['both_entities_accuracy']:.3f}")
        print(f"    matched: {rep['matched_auroc']}")
        print(f"    span_f1 micro: {rep['span_f1']['micro']}")

    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    with open(out_dir / "full.json", "w") as fh:
        keep = ["type_accuracy", "overall_auroc", "accuracy", "matched_auroc",
                "flag_rate_by_subtype", "confusion", "relation_metrics", "span_f1"]
        json.dump({k: {kk: v[kk] for kk in keep if kk in v}
                   for k, v in results.items()}, fh, indent=2)
    with open(out_dir / "failure_cases.json", "w") as fh:
        json.dump({k: results[k]["failure_examples"] for k in results}, fh, indent=2)
    print(f"\n[train_synthetic] wrote {out_dir}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-config", default="configs/model/paraphrase_robust.yaml")
    ap.add_argument("--data-config", default="configs/data/synthetic_augmented.yaml")
    ap.add_argument("--train-config", default="configs/train/synthetic_equilibrium.yaml")
    ap.add_argument("--eval-config", default="configs/eval/in_dist.yaml")
    ap.add_argument("--out", default=None,
                    help="override train.out_dir")
    ap.add_argument("--override", nargs="*", default=[],
                    help="dot.key=value overrides applied across all configs")
    args = ap.parse_args()

    model_cfg = apply_overrides(load_yaml(args.model_config), args.override)
    data_cfg = apply_overrides(load_yaml(args.data_config), args.override)
    train_cfg = apply_overrides(load_yaml(args.train_config), args.override)
    eval_cfg = apply_overrides(load_yaml(args.eval_config), args.override)

    out_dir = args.out or train_cfg["out_dir"]
    run(model_cfg, data_cfg, train_cfg, eval_cfg, out_dir)


if __name__ == "__main__":
    main()
