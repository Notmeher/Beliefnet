"""Shared harness for the multi-statement consistency task.

Encodes statements once (cached), assembles padded per-case tensors, trains
any binary model with signature ``model(emb, mask) -> logit``, and evaluates
with the matched cycle-vs-chain AUROCs that isolate non-pairwise reasoning.
Used by both the baseline script and the BeliefNet script so the comparison
is apples-to-apples.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from beliefnet.data.features import encode_texts
from beliefnet.data.synthetic.case_schema import Case
from beliefnet.eval.metrics import accuracy, auroc, binary_f1


def encode_statement_bank(backbone, cases: list[Case], cache_path, batch: int = 64):
    uniq = sorted({s for c in cases for s in c.statements})
    cache_path = Path(cache_path)
    if cache_path.exists():
        data = np.load(cache_path, allow_pickle=True)
        if list(data["stmts"]) == uniq:
            return {s: i for i, s in enumerate(uniq)}, torch.from_numpy(data["emb"])
    emb = encode_texts(backbone, uniq, batch)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache_path, stmts=np.array(uniq, dtype=object), emb=emb.numpy())
    return {s: i for i, s in enumerate(uniq)}, emb


def assemble(cases: list[Case], lookup, emb_bank, max_k: int):
    n, h = len(cases), emb_bank.shape[1]
    E = torch.zeros(n, max_k, h)
    M = torch.zeros(n, max_k, dtype=torch.bool)
    y = torch.zeros(n)
    for i, c in enumerate(cases):
        for j, s in enumerate(c.statements):
            E[i, j] = emb_bank[lookup[s]]
            M[i, j] = True
        y[i] = 1.0 if c.label == "inconsistent" else 0.0
    return E, M, y


def train_binary(model, E_tr, M_tr, y_tr, E_va, M_va, y_va,
                 epochs: int, lr: float, batch: int, device: str) -> float:
    model.to(device)
    pos = float(y_tr.sum())
    neg = float(len(y_tr) - pos)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([neg / max(pos, 1.0)], device=device))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)

    n = len(y_tr)
    best_state, best_auc = {k: t.clone() for k, t in model.state_dict().items()}, -1.0
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(n, device=device)
        for s in range(0, n, batch):
            idx = perm[s:s + batch]
            opt.zero_grad()
            loss_fn(model(E_tr[idx], M_tr[idx]), y_tr[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            va = torch.sigmoid(model(E_va, M_va)).cpu().numpy()
        va_auc = auroc(va, y_va.cpu().numpy())
        if va_auc > best_auc:
            best_auc = va_auc
            best_state = {k: t.clone() for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return best_auc


def evaluate_cases(model, E_te, M_te, cases: list[Case], device: str) -> dict:
    model.eval()
    with torch.no_grad():
        scores = torch.sigmoid(model(E_te.to(device), M_te.to(device))).cpu().numpy()
    y = np.array([1 if c.label == "inconsistent" else 0 for c in cases])
    pred = (scores >= 0.5).astype(int)

    def matched(pos_pred, neg_pred):
        pos = np.array([scores[i] for i, c in enumerate(cases) if pos_pred(c)])
        neg = np.array([scores[i] for i, c in enumerate(cases) if neg_pred(c)])
        if len(pos) == 0 or len(neg) == 0:
            return None
        s = np.concatenate([pos, neg])
        lab = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
        return round(auroc(s, lab), 4)

    flag_rate = {}
    for sub in sorted({c.conflict_subtype for c in cases}):
        idx = [i for i, c in enumerate(cases) if c.conflict_subtype == sub]
        flag_rate[sub] = {"label": cases[idx[0]].label, "n": len(idx),
                          "flagged_inconsistent": round(float(pred[idx].mean()), 4)}

    return {
        "overall_auroc": round(auroc(scores, y), 4),
        "accuracy": round(accuracy(y, pred), 4),
        "f1_inconsistent": round(binary_f1(y, pred), 4),
        "matched_auroc": {
            "temporal_cycle_vs_chain": matched(
                lambda c: c.conflict_subtype == "temporal_cycle",
                lambda c: c.conflict_subtype == "temporal_chain"),
            "comparative_cycle_vs_chain": matched(
                lambda c: c.conflict_subtype == "comparative_cycle",
                lambda c: c.conflict_subtype == "comparative_chain"),
            "numeric_bad_vs_good_sum": matched(
                lambda c: c.conflict_subtype == "numeric_sum" and c.label == "inconsistent",
                lambda c: c.conflict_subtype == "numeric_sum" and c.label == "consistent"),
            "pairwise_vs_independent": matched(
                lambda c: c.conflict_kind == "pairwise",
                lambda c: c.conflict_subtype == "independent"),
        },
        "flag_rate_by_subtype": flag_rate,
    }


def print_eval(name: str, ev: dict) -> None:
    print(f"\n=== {name}  (params={ev.get('params', '?')}, val_auroc={ev.get('val_auroc_best', '?')}) ===")
    print(f"  overall: AUROC={ev['overall_auroc']:.3f}  acc={ev['accuracy']:.3f}  F1={ev['f1_inconsistent']:.3f}")
    print("  matched AUROC (1.0=detects, 0.5=blind):")
    for k, v in ev["matched_auroc"].items():
        print(f"    {k:30s} {('n/a' if v is None else f'{v:.3f}')}")
    print("  flag rate by subtype (fraction called inconsistent):")
    for sub, st in ev["flag_rate_by_subtype"].items():
        print(f"    {sub:20s} [{st['label']:12s} n={st['n']:3d}] {st['flagged_inconsistent']:.3f}")
