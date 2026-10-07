"""Real-data #2 via HuggingFace: FEVER (claim + gold evidence), SUPPORTS vs
REFUTES, using the same protocol as SciFact (judge + polarity extension).

    PYTHONPATH=src python scripts/run_fever.py --models gpt4o gpt5
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np

from beliefnet.eval.metrics import auroc, binary_f1
from beliefnet.eval.reporting import build_details_md, save_result
from beliefnet.llm.azure_client import client_for, load_dotenv, model_available
from beliefnet.models.pipeline import BeliefNet
from beliefnet.perception.belief_parser import RuleBeliefParser
from beliefnet.reasoning.exact_checker import is_satisfiable


def _metrics(name: str, scores: np.ndarray, labels: np.ndarray) -> dict:
    pred = (scores >= 0.5).astype(int)
    return {"system": name,
            "auroc": round(float(auroc(scores, labels)), 4),
            "accuracy": round(float((pred == labels).mean()), 4),
            "f1_refute": round(float(binary_f1(labels, pred)), 4)}


def _clean_evidence(ev) -> str:
    """Extract readable sentence text from FEVER gold evidence.

    Evidence is ``[page, sent_id, sentence]`` (or a list of such triples). We
    keep the sentence(s) and de-tokenize FEVER's bracket/colon placeholders.
    """
    def detok(s: str) -> str:
        for a, b in (("-LRB-", "("), ("-RRB-", ")"), ("-LSB-", "["),
                     ("-RSB-", "]"), ("-COLON-", ":")):
            s = s.replace(a, b)
        return s.strip()

    sents = []
    if isinstance(ev, list):
        if ev and isinstance(ev[0], (list, tuple)):
            for e in ev:
                if e and isinstance(e[-1], str):
                    sents.append(e[-1])
        elif ev and isinstance(ev[-1], str):
            sents.append(ev[-1])
        else:
            sents.append(" ".join(str(x) for x in ev))
    elif ev:
        sents.append(str(ev))
    return detok(" ".join(sents))


def _load(name: str, split: str):
    """Download the split's JSONL directly over certifi-verified HTTPS.

    We bypass ``datasets``/``huggingface_hub`` because its httpx transport does
    not honor the CA-bundle env vars on this machine (SSL verify failure),
    whereas urllib with certifi's bundle connects cleanly.
    """
    import json
    import ssl
    import urllib.request

    import certifi

    os.environ.setdefault("HF_TOKEN", os.environ.get("Hf_api_key", ""))
    token = os.environ.get("HF_TOKEN", "")
    alias = {"validation": ["valid", "validation", "dev", "val"],
             "val": ["valid", "val", "validation", "dev"],
             "dev": ["valid", "dev", "validation"],
             "train": ["train"], "test": ["test"]}
    candidates = alias.get(split, [split])
    ctx = ssl.create_default_context(cafile=certifi.where())
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    text = None
    for sp in candidates:
        url = f"https://huggingface.co/datasets/{name}/resolve/main/{sp}.jsonl"
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, context=ctx, timeout=90) as resp:
                text = resp.read().decode("utf-8")
            break
        except Exception:
            continue
    if text is None:
        raise RuntimeError(f"could not download split '{split}' of {name}")
    rows = [json.loads(ln) for ln in text.splitlines() if ln.strip()]
    cols = list(rows[0].keys())
    claim_col = next(c for c in cols if "claim" in c.lower())
    label_col = next(c for c in cols
                     if c.lower() in ("label", "gold_label", "verdict"))
    ev_col = next((c for c in cols
                   if "evidence" in c.lower() or "context" in c.lower()), None)
    sup, ref = [], []
    for r in rows:
        lab = str(r[label_col]).upper()
        item = {"claim": str(r[claim_col]), "evidence": _clean_evidence(r[ev_col])}
        if "SUPPORT" in lab:
            sup.append(item)
        elif "REFUTE" in lab or "CONTRADICT" in lab:
            ref.append(item)
    return sup, ref


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="copenlu/fever_gold_evidence")
    ap.add_argument("--split", default="validation")
    ap.add_argument("--n-per-class", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--models", nargs="*", default=["gpt4o", "gpt5"])
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    load_dotenv()
    try:
        sup, ref = _load(args.dataset, args.split)
    except Exception as e:
        print(f"[fever] failed to load {args.dataset}: {type(e).__name__}: {e}")
        return 1

    rng = random.Random(args.seed)
    rng.shuffle(sup)
    rng.shuffle(ref)
    k = min(args.n_per_class, len(sup), len(ref))
    items = ([("consistent", x) for x in sup[:k]]
             + [("inconsistent", x) for x in ref[:k]])
    rng.shuffle(items)
    labels = np.array([1 if lab == "inconsistent" else 0 for lab, _ in items])
    print(f"[fever] {len(items)} cases ({k} SUPPORTS + {k} REFUTES) "
          f"from {args.dataset}")

    def stmts(x):
        return [x["claim"]] + ([x["evidence"]] if x["evidence"] else [])

    rows = []
    net = BeliefNet(RuleBeliefParser())
    bn = np.array([1.0 if net.predict(stmts(x))["inconsistent"] else 0.0
                   for _, x in items])
    rows.append(_metrics("BeliefNet (rule schema)", bn, labels))

    from beliefnet.llm.extractor import scifact_polarity_constraints
    from beliefnet.llm.judge import judge_support_contradict
    for m in args.models:
        if not model_available(m):
            print(f"[skip] {m}: Azure env vars not set")
            continue
        client = client_for(m)
        js = np.array([judge_support_contradict(client, x["claim"],
                                                [x["evidence"]])["p_contradict"]
                       for _, x in items])
        rows.append(_metrics(f"{m} judge", js, labels))
        ps = np.zeros(len(items))
        for i, (_, x) in enumerate(items):
            cons = scifact_polarity_constraints(client, x["claim"], [x["evidence"]])
            ps[i] = 0.0 if is_satisfiable(cons) else 1.0
        rows.append(_metrics(f"BeliefNet (polarity + {m})", ps, labels))

    payload = {
        "title": "Real-data #2 -- FEVER (claim + gold evidence)",
        "question": "Does the SciFact picture replicate on a second real dataset "
                    "(FEVER): LLMs strong, base schema chance, polarity lifts it?",
        "config": {"dataset": args.dataset, "split": args.split,
                   "n_per_class": k, "seed": args.seed, "n_cases": len(items)},
        "summary": {"n_cases": len(items)},
        "tables": {"REFUTES vs SUPPORTS (positive=REFUTES)": rows},
        "notes": ["Second real dataset for external validity; same protocol as "
                  "SciFact."],
    }
    save_result(args.out, "exp_fever", payload)
    build_details_md(args.out)
    print("[fever] " + "; ".join(f"{r['system']}={r['auroc']}" for r in rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
