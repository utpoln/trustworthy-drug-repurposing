"""Score KGE candidates with an LLM (grounded in KG paths, or no-paths baseline).

For every test disease, the top-k KGE candidates (from test_topk.jsonl) are
sent to the LLM one at a time with up to `max_paths` retrieved paths. Every
call is written to a provenance JSONL record (src/provenance/log.py).
The run resumes: pairs already logged without error are skipped.

Usage:
  python -m experiments.run_llm --model openai/gpt-oss-120b --mode grounded
  python -m experiments.run_llm --model openai/gpt-oss-120b --mode no_paths
  python -m experiments.run_llm --model qwen/qwen3.8-27b --limit-diseases 5   # quick test
  python -m experiments.run_llm --model ... --graph data/poisoned/<x>/train.tsv.gz --tag poison_x
"""
import os
import argparse
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from tqdm import tqdm

from src.llm import prompts
from src.llm.client import BASE_URL, chat, get_client, parse_json
from src.provenance.log import JsonlLog, file_md5, git_commit, now_utc, text_sha
from src.retrieval.paths import RELATIONS, find_paths, load_graph, path_to_dict, path_to_text

ROOT = Path(__file__).resolve().parents[1]
SPLIT = Path(os.environ["KG_SPLIT"]).resolve() if os.environ.get("KG_SPLIT") else ROOT / "data" / "processed" / "primekg_v2.1_seed42"  # override with env KG_SPLIT


def slug(model: str) -> str:
    return model.replace("/", "_").replace(":", "-")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--mode", choices=["grounded", "no_paths"], default="grounded")
    ap.add_argument("--kge-run", type=Path, default=ROOT / "results" / "runs" / "kge" / "RotatE_seed42")
    ap.add_argument("--k", type=int, default=20, help="candidates per disease to score")
    ap.add_argument("--max-paths", type=int, default=10)
    ap.add_argument("--path-weight-exp", type=float, default=0.5)
    ap.add_argument("--graph", type=Path, default=None, help="train graph for retrieval (default: clean split)")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit-diseases", type=int, default=None)
    ap.add_argument("--tag", default="clean")
    ap.add_argument("--split", choices=["test", "valid"], default="test",
                    help="which held-out split's top-k candidates to score (valid: for recalibration)")
    ap.add_argument("--sample-diseases", type=int, default=None,
                    help="score a seeded random sample of diseases (same sampler as run_judge)")
    ap.add_argument("--sample-seed", type=int, default=7)
    ap.add_argument("--prompt", choices=sorted(prompts.GROUNDED), default="v1",
                    help="grounded prompt version (v2: every mechanism sentence cites paths)")
    ap.add_argument("--reuse-from", type=Path, default=None,
                    help="clean-run JSONL: reuse its record when the prompt is identical (saves calls)")
    args = ap.parse_args()
    args.kge_run = args.kge_run.resolve()
    if args.graph is not None:
        args.graph = args.graph.resolve()

    topk_file = args.kge_run / f"{args.split}_topk.jsonl"
    diseases = [json.loads(l) for l in open(topk_file)]
    if args.sample_diseases:
        import numpy as np
        all_d = sorted(r["disease"] for r in diseases)
        rng = np.random.default_rng(args.sample_seed)
        keep = set(rng.choice(all_d, size=min(args.sample_diseases, len(all_d)), replace=False).tolist())
        diseases = [r for r in diseases if r["disease"] in keep]
    if args.limit_diseases:
        diseases = diseases[: args.limit_diseases]

    graph_file = args.graph or SPLIT / "train.tsv.gz"
    g = load_graph(SPLIT, graph_file) if args.mode == "grounded" else None

    out = ROOT / "results" / "runs" / "llm" / args.tag / args.mode / f"{slug(args.model)}.jsonl"
    log = JsonlLog(out, ("disease", "drug"))
    done = log.done_keys()

    static = {
        "data": {
            "version": json.loads((SPLIT / "split_meta.json").read_text())["data_version"],
            "split_dir": str(SPLIT.relative_to(ROOT)),
            "retrieval_graph": str(graph_file.relative_to(ROOT)) if graph_file.is_relative_to(ROOT) else str(graph_file),
            "retrieval_graph_md5": file_md5(str(graph_file)) if args.mode == "grounded" else None,
            "tag": args.tag,
        },
        "candidates": {
            "kge_run": str(args.kge_run.relative_to(ROOT)),
            "kge_model_md5": file_md5(str(args.kge_run / "model.pt")),
            "topk_file_md5": file_md5(str(topk_file)),
        },
        "retrieval": {"max_paths": args.max_paths, "w": args.path_weight_exp, "max_hops": 3,
                      "relations": sorted(RELATIONS)} if args.mode == "grounded" else None,
        "code": {"git_commit": git_commit()},
    }

    jobs = []
    for rec in diseases:
        for c in rec["candidates"][: args.k]:
            if (rec["disease"], c["drug"]) not in done:
                jobs.append((rec, c))
    print(f"{args.model} [{args.mode}/{args.tag}]: {len(jobs)} calls to make "
          f"({len(done)} already done) -> {out}")

    client = get_client()
    reuse = {}
    if args.reuse_from and args.reuse_from.exists():
        for line in open(args.reuse_from):
            r = json.loads(line)
            if r["llm"]["error"] is None:
                reuse[r["prompt"]["sha"]] = r
        print(f"reuse pool: {len(reuse)} clean records")

    def work(rec, c):
        disease, drug = rec["disease_name"], c["drug_name"]
        if args.mode == "grounded":
            paths, counts = find_paths(g, c["drug"], rec["disease"], args.max_paths, args.path_weight_exp)
            pdicts = [path_to_dict(g, p) for p in paths]
            system, user = prompts.grounded(disease, drug, [p["text"] for p in pdicts], args.prompt)
        else:
            pdicts, counts = [], None
            system, user = prompts.no_paths(disease, drug)
        sha = text_sha(system + "\n" + user)
        if sha in reuse:
            old = reuse[sha]
            r = {"content": old["llm"]["raw"], "reasoning": old["llm"]["reasoning"],
                 "finish_reason": old["llm"]["finish_reason"], "usage": old["llm"]["usage"],
                 "latency_s": old["llm"]["latency_s"], "attempts": 0, "error": None}
            reused_from = old["timestamp"]
        else:
            r = chat(client, args.model, system, user)
            reused_from = None
        parsed = parse_json(r["content"])
        log.write({
            "timestamp": now_utc(),
            "disease": rec["disease"], "disease_name": disease,
            "drug": c["drug"], "drug_name": drug,
            "is_gold": c["drug"] in rec["gold_drugs"],
            "kge_rank": c["kge_rank"], "kge_score": c["kge_score"],
            "mode": args.mode,
            **static,
            "paths": pdicts, "path_counts": counts,
            "prompt": {"version": prompts.version(args.mode, args.prompt), "system": system, "user": user,
                       "sha": sha},
            "llm": {"model": args.model, "endpoint": BASE_URL, "temperature": 0.0,
                    "raw": r["content"], "reasoning": r["reasoning"], "finish_reason": r["finish_reason"],
                    "usage": r["usage"], "latency_s": r["latency_s"], "attempts": r["attempts"],
                    "error": r["error"], "reused_from_clean": reused_from},
            "parsed": parsed,
            "parse_ok": parsed is not None and isinstance(parsed.get("confidence"), (int, float)),
        })
        return r["error"] is None

    t0, fails = time.time(), 0
    with ThreadPoolExecutor(args.workers) as ex:
        futs = [ex.submit(work, rec, c) for rec, c in jobs]
        for f in tqdm(as_completed(futs), total=len(futs), desc=slug(args.model)):
            fails += not f.result()
    print(f"done in {(time.time() - t0) / 60:.1f} min, {fails} failed calls (rerun to retry)")


if __name__ == "__main__":
    main()
