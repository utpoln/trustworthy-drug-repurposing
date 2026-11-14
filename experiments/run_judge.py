"""Judge faithfulness of grounded LLM records (see src/eval/faithfulness.py).

Judges a fixed, seeded sample of diseases (same sample for every model) so
results are comparable. Each judgement is logged with provenance.

Usage:
  python -m experiments.run_judge --llm-file results/runs/llm/clean/grounded/<model>.jsonl
  python -m experiments.run_judge --llm-file ... --judge mistral-large:123b --n-diseases 20  # agreement
"""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from tqdm import tqdm

from src.eval.faithfulness import (JUDGE_MODEL, JUDGE_VERSION, judge_prompt, parse_labels,
                                   statements)
from src.llm.client import chat, get_client, parse_json
from src.provenance.log import JsonlLog, file_md5, now_utc, text_sha

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--llm-file", type=Path, required=True)
    ap.add_argument("--judge", default=JUDGE_MODEL)
    ap.add_argument("--n-diseases", type=int, default=100)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    recs = [json.loads(l) for l in open(args.llm_file)]
    recs = [r for r in recs if r["llm"]["error"] is None and r.get("parse_ok")]
    diseases = sorted({r["disease"] for r in recs})
    # the disease sample depends only on the seed and the full test-disease list
    topk = ROOT / "results" / "runs" / "kge" / "RotatE_seed42" / "test_topk.jsonl"
    all_d = sorted(json.loads(l)["disease"] for l in open(topk))
    rng = np.random.default_rng(args.seed)
    sample = set(rng.choice(all_d, size=min(args.n_diseases, len(all_d)), replace=False).tolist())
    recs = [r for r in recs if r["disease"] in sample]

    judge_slug = args.judge.replace("/", "_").replace(":", "-")
    out = args.llm_file.with_name(args.llm_file.stem + f".judge_{judge_slug}.jsonl")
    log = JsonlLog(out, ("disease", "drug"))
    done = log.done_keys()
    todo = [r for r in recs if (r["disease"], r["drug"]) not in done]
    print(f"judge={args.judge}: {len(todo)} records to judge ({len(done)} done) "
          f"from {len(sample & set(diseases))} diseases -> {out}")
    client = get_client()
    src_md5 = file_md5(str(args.llm_file))

    def work(r):
        stmts = statements(r)
        if not stmts:
            labels, res, system, user = {}, {"content": "", "error": None, "usage": {}, "latency_s": 0}, "", ""
        else:
            system, user = judge_prompt(r, stmts)
            res = chat(client, args.judge, system, user)
            labels = parse_labels(parse_json(res["content"]), stmts)
        log.write({
            "timestamp": now_utc(), "disease": r["disease"], "drug": r["drug"],
            "is_gold": r["is_gold"], "model": r["llm"]["model"], "source_file": str(args.llm_file),
            "source_md5": src_md5, "n_paths": len(r["paths"]),
            "statements": stmts, "labels": labels,
            "judge": {"model": args.judge, "version": JUDGE_VERSION, "prompt_sha": text_sha(system + user),
                      "raw": res["content"], "usage": res.get("usage"), "latency_s": res.get("latency_s"),
                      "error": res["error"]},
            "llm": {"error": res["error"]},
        })
        return res["error"] is None

    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, r) for r in todo]
        fails = sum(not f.result() for f in tqdm(as_completed(futs), total=len(futs)))
    print(f"{fails} failed judge calls (rerun to retry)")


if __name__ == "__main__":
    main()
