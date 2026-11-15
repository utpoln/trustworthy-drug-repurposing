"""Regenerate every results table of the paper from the run logs.

Reads results/runs/{kge,llm}/ and data/poisoned/, writes
  results/tables/results.json         all numbers (single source for the paper)
  results/tables/*.md                 human-readable tables
  paper/tables/*.tex                  LaTeX table bodies (\\input in the manuscript)

Confidence intervals: 95% percentile bootstrap, 2,000 resamples of test
*diseases* (cluster bootstrap: pairs of one disease are resampled together),
paired across methods. Seed 0.

Usage: python -m experiments.make_results
"""
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.eval.faithfulness import cites_fake, fake_edge_set
from src.eval.ranking import ranking_metrics
from src.eval.rerank import calibration, load_llm, rerank_ranks

ROOT = Path(__file__).resolve().parents[1]
KGE = ROOT / "results" / "runs" / "kge"
LLM = ROOT / "results" / "runs" / "llm"
POIS = ROOT / "data" / "poisoned"
TAB, TEX = ROOT / "results" / "tables", ROOT / "paper" / "tables"
K, B, SEED = 20, 2000, 0

MODELS = [  # (slug, display name), display order
    ("qwen_qwen3.8-27b", "Qwen3.8-27B"),
    ("google_gemma-4-31b", "Gemma-4-31B"),
    ("qwen_qwen3.5-122b", "Qwen3.5-122B"),
    ("openai_gpt-oss-120b", "gpt-oss-120B"),
    ("moonshotai_kimi-k3", "Kimi-K3"),
]
JUDGE = "Nemotron-3-Super-120b"


# ----------------------------------------------------------------- helpers
class Boot:
    """Paired cluster bootstrap over diseases."""

    def __init__(self, diseases: np.ndarray):
        u, inv = np.unique(diseases, return_inverse=True)
        rng = np.random.default_rng(SEED)
        draws = rng.integers(0, len(u), (B, len(u)))
        # weight of each pair in each resample = how often its disease was drawn
        counts = np.stack([np.bincount(d, minlength=len(u)) for d in draws])
        self.w = counts[:, inv].astype(float)

    def ci(self, x: np.ndarray):
        est = (self.w * x).sum(1) / self.w.sum(1)
        return [float(v) for v in np.percentile(est, [2.5, 97.5])]


def fmt_ci(m, ci):
    return f"{m:+.3f} [{ci[0]:+.3f}, {ci[1]:+.3f}]"


def md_table(rows, cols):
    out = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    out += ["| " + " | ".join(str(r[c]) for c in cols) + " |" for r in rows]
    return "\n".join(out) + "\n"


def tex_rows(rows, cols):
    return "\n".join(" & ".join(str(r[c]) for c in cols) + r" \\" for r in rows) + "\n"


def write(name, rows, cols, title):
    (TAB / f"{name}.md").write_text(f"# {title}\n\n" + md_table(rows, cols))
    (TEX / f"{name}.tex").write_text(tex_rows(rows, cols))


# ----------------------------------------------------------------- accuracy + calibration
def accuracy_calibration(res):
    tr = pd.read_csv(KGE / "RotatE_seed42" / "test_ranks.tsv", sep="\t")
    boot = Boot(tr["disease"].to_numpy())
    kge = tr["kge_rank"].to_numpy(float)
    transe = pd.read_csv(KGE / "TransE_seed42" / "test_ranks.tsv", sep="\t") \
        if (KGE / "TransE_seed42" / "test_ranks.tsv").exists() else None

    acc_rows, cal_rows, acc = [], [], {}

    def row(name, ranks, ref=None, ref2=None):
        m = ranking_metrics(ranks)
        r = {"Method": name, "MRR": f"{m['MRR']:.3f}", "H@1": f"{m['Hits@1']:.3f}",
             "H@3": f"{m['Hits@3']:.3f}", "H@10": f"{m['Hits@10']:.3f}", "dKGE": "–", "dNoPaths": "–"}
        out = {"metrics": m}
        for key, base in (("dKGE", ref), ("dNoPaths", ref2)):
            if base is not None:
                d = 1 / ranks - 1 / base
                out[key] = {"mean": float(d.mean()), "ci": boot.ci(d)}
                r[key] = fmt_ci(d.mean(), out[key]["ci"])
        acc_rows.append(r)
        return out

    if transe is not None:
        acc["TransE"] = row("TransE (graph only)", transe["kge_rank"].to_numpy(float))
    acc["RotatE"] = row("RotatE (graph only)", kge)
    for slug, name in MODELS:
        R, cal = {}, {}
        for mode in ("no_paths", "grounded"):
            L = load_llm(LLM / "clean" / mode / f"{slug}.jsonl")
            rr = rerank_ranks(L, tr, K)
            R[mode], R[mode + "_rrf"] = np.array(rr["llm"], float), np.array(rr["rrf"], float)
            cal[mode] = calibration(L)
        acc[slug] = {
            "no_paths": row(f"{name} no paths", R["no_paths"], kge),
            "grounded": row(f"{name} + KG paths", R["grounded"], kge, R["no_paths"]),
            "grounded_rrf": row(f"{name} + KG paths, RRF", R["grounded_rrf"], kge),
        }
        for mode in ("no_paths", "grounded"):
            c = cal[mode]
            cal_rows.append({"Model": name, "Mode": mode.replace("_", " "), "Parse": f"{c['parse_rate']:.3f}",
                             "MeanConf": f"{c['mean_conf']:.3f}", "ECE": f"{c['ECE']:.3f}",
                             "Brier": f"{c['Brier']:.3f}", "AUROC": f"{c['AUROC']:.3f}"})
        acc[slug]["calibration"] = cal
    res["accuracy"] = acc
    res["base_rate_topk"] = cal["grounded"]["base_rate"]
    res["gold_in_topk"] = float(np.mean(kge <= K))
    write("accuracy", acc_rows, ["Method", "MRR", "H@1", "H@3", "H@10", "dKGE", "dNoPaths"],
          f"Accuracy on {len(tr)} held-out indications (re-ranking top-{K}; dMRR with 95% disease-bootstrap CI)")
    write("calibration", cal_rows, ["Model", "Mode", "Parse", "MeanConf", "ECE", "Brier", "AUROC"],
          f"Calibration of LLM confidence over all disease-candidate pairs (base rate {res['base_rate_topk']:.3f})")


# ----------------------------------------------------------------- faithfulness
def faithfulness(res):
    rows, out = [], {}
    for slug, name in MODELS:
        f = LLM / "clean" / "grounded" / f"{slug}.judge_{JUDGE}.jsonl"
        if not f.exists():
            continue
        by = defaultdict(Counter); n_pairs = all_ok = valid = n_st = 0; seen = set()
        recs = [json.loads(l) for l in open(f)]
        for r in reversed(recs):  # last judgement per pair wins
            key = (r["disease"], r["drug"])
            if key in seen or r["llm"]["error"] or not r.get("labels"):
                continue
            seen.add(key); n_pairs += 1; ok = True
            for s in r["statements"]:
                lab = r["labels"].get(s["id"], "MISSING")
                by[s["source"]][lab] += 1; by["all"][lab] += 1; n_st += 1
                ok &= lab == "SUPPORTED"
                if s["source"] == "claim":
                    valid += bool(s["cited"]) and all(
                        c.startswith("P") and c[1:].isdigit() and 1 <= int(c[1:]) <= r["n_paths"] for c in s["cited"])
            all_ok += ok
        share = {src: {k: v / sum(c.values()) for k, v in c.items()} for src, c in by.items()}
        out[slug] = {"pairs": n_pairs, "statements": n_st, "share": share,
                     "pairs_all_supported": all_ok / n_pairs,
                     "claim_citation_valid": valid / max(sum(by["claim"].values()), 1)}
        g = lambda s, k: f"{share.get(s, {}).get(k, 0):.3f}"
        rows.append({"Model": name, "Pairs": n_pairs, "Stmts": n_st,
                     "Claim S": g("claim", "SUPPORTED"), "Claim U": g("claim", "UNSUPPORTED"),
                     "Mech S": g("mechanism", "SUPPORTED"), "Mech P": g("mechanism", "PARTIAL"),
                     "Mech U": g("mechanism", "UNSUPPORTED"), "All S": g("all", "SUPPORTED"),
                     "Pair all-S": f"{all_ok / n_pairs:.3f}",
                     "Cite valid": f"{out[slug]['claim_citation_valid']:.3f}"})
    res["faithfulness"] = {"judge": JUDGE, "models": out,
                           "note": "LLM-judge labels; pending validation against human labels"}
    write("faithfulness", rows, ["Model", "Pairs", "Stmts", "Claim S", "Claim U", "Mech S", "Mech P",
                                 "Mech U", "All S", "Pair all-S", "Cite valid"],
          f"Faithfulness (judge {JUDGE}; S=supported, P=partial, U=unsupported; 100 sampled diseases)")


# ----------------------------------------------------------------- robustness
def decoy_stats(L, decoys):
    out = {}
    for d, g in L.groupby("disease"):
        if d not in decoys:
            continue
        g = g[g["kge_rank_raw"] <= K].copy()
        g["confidence"] = g["confidence"].fillna(0.0)
        g = g.sort_values(["confidence", "kge_rank_raw"], ascending=[False, True]).reset_index(drop=True)
        hit = g.index[g["drug"] == decoys[d]]
        if len(hit):
            out[d] = (int(hit[0]) + 1, float(g.loc[hit[0], "confidence"]))
    return out


def robustness(res):
    tr = pd.read_csv(KGE / "RotatE_seed42" / "test_ranks.tsv", sep="\t")
    rob = {"kge": {}, "llm_targeted": {}, "llm_random": {}}
    # training-time (KGE retrained on poisoned graph)
    kge_rows = []
    for tag, name in (("", "clean"), ("_poison_targeted_n3_seed13", "targeted (n=3)"),
                      ("_poison_random_r0.05_seed13", "random 5%")):
        run = KGE / f"RotatE_seed42{tag}"
        m = json.loads((run / "metrics.json").read_text())["test"]
        dr = json.loads((run / "decoy_ranks.json").read_text()) if (run / "decoy_ranks.json").exists() else None
        rob["kge"][name] = {"test": m, "decoy": dr}
        kge_rows.append({"Graph": name, "MRR": f"{m['MRR']:.3f}", "H@1": f"{m['Hits@1']:.3f}",
                         "H@10": f"{m['Hits@10']:.3f}",
                         "Decoy@1": f"{dr['top1_clean']:.3f} -> {dr['top1_poisoned']:.3f}" if dr else "–",
                         "Decoy med": f"{dr['median_rank_clean']:.0f} -> {dr['median_rank_poisoned']:.0f}" if dr else "–"})
    write("robustness_kge", kge_rows, ["Graph", "MRR", "H@1", "H@10", "Decoy@1", "Decoy med"],
          "Training-time poisoning: RotatE retrained on the poisoned graph")

    # retrieval-time targeted
    dec = pd.read_csv(POIS / "targeted_n3_seed13" / "decoys.tsv", sep="\t")
    D = dict(zip(dec["disease"], dec["decoy_drug"]))
    fake_t = fake_edge_set(POIS / "targeted_n3_seed13" / "fake_edges.tsv")
    fake_r = fake_edge_set(POIS / "random_r0.05_seed13" / "fake_edges.tsv")
    t_rows, r_rows = [], []
    for slug, name in MODELS:
        C = load_llm(LLM / "clean" / "grounded" / f"{slug}.jsonl")
        P = load_llm(LLM / "poison_targeted_n3" / "grounded" / f"{slug}.jsonl")
        c, p = decoy_stats(C, D), decoy_stats(P, D)
        ks = sorted(set(c) & set(p))
        rc, rp = np.array([c[k][0] for k in ks]), np.array([p[k][0] for k in ks])
        cc, cp = np.array([c[k][1] for k in ks]), np.array([p[k][1] for k in ks])
        mc = ranking_metrics(np.array(rerank_ranks(C, tr, K)["llm"], float))
        mp = ranking_metrics(np.array(rerank_ranks(P, tr, K)["llm"], float))
        nf = nt = 0
        for line in open(LLM / "poison_targeted_n3" / "grounded" / f"{slug}.jsonl"):
            r = json.loads(line)
            if D.get(r["disease"]) == r["drug"] and r.get("parse_ok") and not r["llm"]["error"]:
                nt += 1; nf += cites_fake(r, fake_t)
        rob["llm_targeted"][slug] = {
            "n": len(ks), "decoy_top1": [float(np.mean(rc == 1)), float(np.mean(rp == 1))],
            "decoy_median_rank": [float(np.median(rc)), float(np.median(rp))],
            "decoy_conf": [float(cc.mean()), float(cp.mean())], "promoted": float(np.mean(rp < rc)),
            "gold_MRR": [mc["MRR"], mp["MRR"]], "gold_H1": [mc["Hits@1"], mp["Hits@1"]],
            "decoy_expl_cites_fake": nf / max(nt, 1)}
        x = rob["llm_targeted"][slug]
        t_rows.append({"Model": name, "Decoy@1": f"{x['decoy_top1'][0]:.3f} -> {x['decoy_top1'][1]:.3f}",
                       "Decoy med": f"{x['decoy_median_rank'][0]:.0f} -> {x['decoy_median_rank'][1]:.0f}",
                       "Decoy conf": f"{x['decoy_conf'][0]:.2f} -> {x['decoy_conf'][1]:.2f}",
                       "Promoted": f"{x['promoted']:.3f}", "Cites fake": f"{x['decoy_expl_cites_fake']:.3f}",
                       "Gold MRR": f"{x['gold_MRR'][0]:.3f} -> {x['gold_MRR'][1]:.3f}"})

        # retrieval-time random 5% (first 100 diseases): compare on the same diseases
        Pr = load_llm(LLM / "poison_random_r0.05" / "grounded" / f"{slug}.jsonl")
        sub = tr[tr["disease"].isin(set(Pr["disease"]))]
        mc = ranking_metrics(np.array(rerank_ranks(C[C.disease.isin(set(sub.disease))], sub, K)["llm"], float))
        mp = ranking_metrics(np.array(rerank_ranks(Pr, sub, K)["llm"], float))
        nf = nt = nfp = 0
        for line in open(LLM / "poison_random_r0.05" / "grounded" / f"{slug}.jsonl"):
            r = json.loads(line)
            if r.get("parse_ok") and not r["llm"]["error"]:
                nt += 1; nf += cites_fake(r, fake_r)
                nfp += any(frozenset((p["nodes"][i], p["nodes"][i + 1])) in fake_r
                           for p in r["paths"] for i in range(len(p["nodes"]) - 1))
        rob["llm_random"][slug] = {"n_pairs": len(sub), "gold_MRR": [mc["MRR"], mp["MRR"]],
                                   "gold_H1": [mc["Hits@1"], mp["Hits@1"]],
                                   "expl_with_fake_path": nfp / max(nt, 1), "expl_cites_fake": nf / max(nt, 1)}
        y = rob["llm_random"][slug]
        r_rows.append({"Model": name, "Pairs": len(sub),
                       "Gold MRR": f"{y['gold_MRR'][0]:.3f} -> {y['gold_MRR'][1]:.3f}",
                       "Gold H@1": f"{y['gold_H1'][0]:.3f} -> {y['gold_H1'][1]:.3f}",
                       "Fake in paths": f"{y['expl_with_fake_path']:.3f}", "Cites fake": f"{y['expl_cites_fake']:.3f}"})
    res["robustness"] = rob
    write("robustness_llm_targeted", t_rows,
          ["Model", "Decoy@1", "Decoy med", "Decoy conf", "Promoted", "Cites fake", "Gold MRR"],
          "Retrieval-time targeted poisoning (3 fake chains per decoy; clean -> poisoned)")
    write("robustness_llm_random", r_rows, ["Model", "Pairs", "Gold MRR", "Gold H@1", "Fake in paths", "Cites fake"],
          "Retrieval-time random poisoning (5% fake drug-protein / disease-protein edges; 100 diseases)")


def main():
    TAB.mkdir(parents=True, exist_ok=True); TEX.mkdir(parents=True, exist_ok=True)
    res = {"k": K, "bootstrap": {"resamples": B, "unit": "disease", "seed": SEED}}
    accuracy_calibration(res)
    faithfulness(res)
    robustness(res)
    (TAB / "results.json").write_text(json.dumps(res, indent=1, default=float))
    for f in ("accuracy", "calibration", "faithfulness", "robustness_kge",
              "robustness_llm_targeted", "robustness_llm_random"):
        print((TAB / f"{f}.md").read_text())


if __name__ == "__main__":
    main()
