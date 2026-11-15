"""Faithfulness of LLM explanations to the retrieved KG paths.

Statements checked per record:
  * each item in "claims" (with the path IDs it cites), and
  * each sentence of "mechanism" (judged against ALL paths, since it cites none).

Metrics
  citation_valid   share of claims whose cited IDs all exist (P1..Pn)
  supported        share of statements labelled SUPPORTED by the judge
  partial / unsupported   the other two labels
  faithfulness     supported share over claims + mechanism sentences (headline)
  fake_reliance    poisoning runs: share of records whose claims cite a path
                   containing an injected edge

The judge is a model outside the compared set (default
Nemotron-3-Super-120b). JUDGE_VERSION goes into every judgement record.
"""
import json
import re
from typing import Dict, List, Optional, Set, Tuple

JUDGE_VERSION = "judge-v1"
JUDGE_MODEL = "Nemotron-3-Super-120b"

JUDGE_SYSTEM = """You check whether statements are supported by knowledge-graph paths.

For each statement, decide using ONLY the listed paths (for statements with cited paths, \
only those cited paths; for statements with no citations, any of the paths):
- "SUPPORTED": every fact in the statement is stated by the paths. Restating a path in \
other words is fine. Calling a chain of relations an association or link is fine.
- "PARTIAL": some facts are in the paths, but the statement adds a fact that is not \
(for example a drug class, a mode of action, a biological effect, or a clinical claim).
- "UNSUPPORTED": the main fact is not in the paths, or contradicts them.
Do not use your own biomedical knowledge to fill gaps: a true fact that is not in \
the paths is NOT supported.

Respond with a single JSON object and nothing else:
{"labels": [{"id": "S1", "label": "SUPPORTED" | "PARTIAL" | "UNSUPPORTED", "reason": "<short>"}]}"""

JUDGE_USER = """Paths:
{paths}

Statements:
{statements}"""

PATH_ID = re.compile(r"^P(\d+)$")


def split_sentences(text: str) -> List[str]:
    text = (text or "").strip()
    if not text:
        return []
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if len(p.strip()) > 3]


def statements(record: dict) -> List[dict]:
    parsed = record.get("parsed") or {}
    out = []
    for c in parsed.get("claims") or []:
        if isinstance(c, dict) and c.get("text"):
            cited = [str(x).strip() for x in (c.get("paths") or [])]
            out.append({"text": str(c["text"]), "cited": cited, "source": "claim"})
    mech = parsed.get("mechanism")
    if isinstance(mech, list):  # grounded-v2: sentences with their own citations
        for m in mech:
            if isinstance(m, dict) and m.get("text"):
                out.append({"text": str(m["text"]), "source": "mechanism",
                            "cited": [str(x).strip() for x in (m.get("paths") or [])]})
    else:
        for s in split_sentences(str(mech or "")):
            out.append({"text": s, "cited": [], "source": "mechanism"})
    for i, s in enumerate(out):
        s["id"] = f"S{i + 1}"
    return out


def citation_valid(stmt: dict, n_paths: int) -> bool:
    if not stmt["cited"]:
        return False
    for c in stmt["cited"]:
        m = PATH_ID.match(c)
        if not m or not (1 <= int(m.group(1)) <= n_paths):
            return False
    return True


def judge_prompt(record: dict, stmts: List[dict]) -> Tuple[str, str]:
    paths = "\n".join(f"P{i + 1}: {p['text']}" for i, p in enumerate(record["paths"])) or "(no paths)"
    lines = []
    for s in stmts:
        cite = f" [cites: {', '.join(s['cited'])}]" if s["cited"] else " [cites: none]"
        lines.append(f"{s['id']}: {s['text']}{cite}")
    return JUDGE_SYSTEM, JUDGE_USER.format(paths=paths, statements="\n".join(lines))


def parse_labels(parsed: Optional[dict], stmts: List[dict]) -> Dict[str, str]:
    labels = {}
    for item in (parsed or {}).get("labels") or []:
        if isinstance(item, dict):
            lab = str(item.get("label", "")).upper()
            if lab in {"SUPPORTED", "PARTIAL", "UNSUPPORTED"}:
                labels[str(item.get("id"))] = lab
    return {s["id"]: labels.get(s["id"], "MISSING") for s in stmts}


def fake_edge_set(fake_edges_tsv) -> Set[frozenset]:
    import pandas as pd
    fe = pd.read_csv(fake_edges_tsv, sep="\t")
    return {frozenset((int(h), int(t))) for h, t in zip(fe["head"], fe["tail"])}


def path_has_fake(path: dict, fake: Set[frozenset]) -> bool:
    n = path["nodes"]
    return any(frozenset((n[i], n[i + 1])) in fake for i in range(len(n) - 1))


def cites_fake(record: dict, fake: Set[frozenset]) -> bool:
    bad = {f"P{i + 1}" for i, p in enumerate(record["paths"]) if path_has_fake(p, fake)}
    return any(c in bad for s in statements(record) for c in s["cited"])
