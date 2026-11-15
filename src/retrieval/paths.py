"""Retrieve 2- and 3-hop KG paths linking a drug to a disease.

The graph is the TRAINING graph of the split (held-out indications and
leaky edges are absent), optionally a poisoned copy of it.

Path ranking follows the degree-weighted path count of Himmelstein et al.
(2017, Hetionet): a path's weight is the product over its intermediate
nodes of deg(v) ** -w (default w = 0.5). Hub nodes (e.g. a protein with
thousands of interactions) make a path less specific, so it ranks lower.

Only mechanism-related relations are used (see RELATIONS). Anatomy
expression edges and most ontology parent-child edges are excluded: they are
very numerous and rarely explain a mechanism.
"""
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = Path(os.environ["KG_SPLIT"]).resolve() if os.environ.get("KG_SPLIT") else ROOT / "data" / "processed" / "primekg_v2.1_seed42"  # override with env KG_SPLIT
RAW_KG = ROOT / "data" / "raw" / "kg.csv"

RELATIONS = {
    "drug_protein": "target",                 # refined per edge (target/enzyme/...)
    "protein_protein": "interacts with",
    "disease_protein": "associated with",
    "pathway_protein": "participates in",
    "bioprocess_protein": "involved in",
    "molfunc_protein": "has function",
    "phenotype_protein": "associated with",
    "drug_effect": "has side effect",
    "disease_phenotype_positive": "has phenotype",
    "disease_disease": "is related to (parent/child)",
    "indication": "is indicated for",
    "off-label use": "is used off-label for",
    "drug_drug": "interacts with drug",
    "exposure_disease": "is linked to",
}
# relations between nodes of the same type: no meaningful direction
SYMMETRIC = {"protein_protein", "drug_drug", "disease_disease"}


@dataclass
class Path_:
    nodes: List[int]               # node_index, drug first, disease last
    relations: List[str]           # display text of each hop
    forward: List[Optional[bool]]  # canonical direction of each hop; None = symmetric
    weight: float
    kind: str = ""                 # "mechanistic" or "associative"

    @property
    def hops(self) -> int:
        return len(self.relations)


@dataclass
class KGGraph:
    indptr: np.ndarray
    nbr: np.ndarray                # neighbour node_index
    rel: np.ndarray                # relation id
    fwd: np.ndarray                # bool: edge stored as (u -> nbr)
    rel_names: List[str]
    degree: np.ndarray
    names: np.ndarray
    types: np.ndarray
    dp_display: Dict[Tuple[int, int], str] = field(default_factory=dict)

    def neighbours(self, u: int):
        a, b = self.indptr[u], self.indptr[u + 1]
        return self.nbr[a:b], self.rel[a:b], self.fwd[a:b]

    def hop_dir(self, r: int, fwd: bool) -> Optional[bool]:
        return None if self.rel_names[r] in SYMMETRIC else fwd

    def hop_text(self, u: int, v: int, r: int, fwd: bool) -> str:
        name = self.rel_names[r]
        if name == "drug_protein":
            key = (u, v) if fwd else (v, u)
            # edges absent from kg.csv (i.e. injected by poisoning) read exactly like real targets
            return self.dp_display.get(key, "target")
        return RELATIONS[name]


def drug_protein_display(raw_kg: Path = RAW_KG) -> Dict[Tuple[int, int], str]:
    """(drug, protein) -> 'target' / 'enzyme' / 'target, enzyme' ... from kg.csv."""
    df = pd.read_csv(raw_kg, usecols=["relation", "display_relation", "x_index", "x_type", "y_index"],
                     low_memory=False)
    df = df[(df["relation"] == "drug_protein") & (df["x_type"] == "drug")]
    g = df.groupby(["x_index", "y_index"])["display_relation"].agg(lambda s: ", ".join(sorted(set(s))))
    return {(int(d), int(p)): v for (d, p), v in g.items()}


def build_graph(train_file: Path, nodes_file: Path, raw_kg: Path = RAW_KG) -> KGGraph:
    tr = pd.read_csv(train_file, sep="\t")
    tr = tr[tr["relation"].isin(RELATIONS)]
    nodes = pd.read_csv(nodes_file, sep="\t", low_memory=False).sort_values("node_index")
    n = int(nodes["node_index"].max()) + 1

    rel_names = sorted(RELATIONS)
    rid = {r: i for i, r in enumerate(rel_names)}
    h, t = tr["head"].to_numpy(), tr["tail"].to_numpy()
    r = tr["relation"].map(rid).to_numpy()
    src = np.concatenate([h, t]); dst = np.concatenate([t, h]); rr = np.concatenate([r, r])
    fwd = np.concatenate([np.ones(len(h), bool), np.zeros(len(h), bool)])
    order = np.argsort(src, kind="stable")
    src, dst, rr, fwd = src[order], dst[order], rr[order], fwd[order]
    indptr = np.zeros(n + 1, dtype=np.int64)
    np.add.at(indptr, src + 1, 1)
    indptr = np.cumsum(indptr)

    names = np.empty(n, dtype=object); types = np.empty(n, dtype=object)
    names[nodes["node_index"].to_numpy()] = nodes["node_name"].astype(str).to_numpy()
    types[nodes["node_index"].to_numpy()] = nodes["node_type"].astype(str).to_numpy()
    return KGGraph(indptr, dst, rr.astype(np.int16), fwd, rel_names, np.diff(indptr),
                   names, types, drug_protein_display(raw_kg))


def load_graph(split_dir: Path = DEFAULT_SPLIT, train_file: Optional[Path] = None) -> KGGraph:
    return build_graph(train_file or split_dir / "train.tsv.gz", split_dir / "nodes.tsv.gz")


def find_paths(g: KGGraph, drug: int, disease: int, max_paths: int = 10,
               w: float = 0.5, max_hops: int = 3) -> Tuple[List[Path_], Dict[str, int]]:
    """Top paths of length 2..max_hops from drug to disease (direct edges excluded)."""
    deg = g.degree.astype(float)
    n1, r1, f1 = g.neighbours(drug)
    m1, rm, fm = g.neighbours(disease)
    keep1 = n1 != disease
    n1, r1, f1 = n1[keep1], r1[keep1], f1[keep1]
    keepm = m1 != drug
    m1, rm, fm = m1[keepm], rm[keepm], fm[keepm]

    found: List[Path_] = []
    counts = {"2hop": 0, "3hop": 0}

    # 2-hop: drug - a - disease
    m_index: Dict[int, List[int]] = {}
    for j, v in enumerate(m1):
        m_index.setdefault(int(v), []).append(j)
    for j, a in enumerate(n1):
        for k in m_index.get(int(a), ()):
            counts["2hop"] += 1
            found.append(Path_(
                [drug, int(a), disease],
                [g.hop_text(drug, int(a), int(r1[j]), bool(f1[j])),
                 g.hop_text(int(a), disease, int(rm[k]), not bool(fm[k]))],
                [g.hop_dir(int(r1[j]), bool(f1[j])), g.hop_dir(int(rm[k]), not bool(fm[k]))],
                deg[a] ** -w,
            ))

    # 3-hop: drug - a - b - disease
    if max_hops >= 3 and len(n1) and len(m1):
        starts = g.indptr[n1]; ends = g.indptr[n1 + 1]
        lens = ends - starts
        total = int(lens.sum())
        if total:
            owner = np.repeat(np.arange(len(n1)), lens)
            idx = np.arange(total) - np.repeat(np.cumsum(lens) - lens, lens) + np.repeat(starts, lens)
            b = g.nbr[idx]; rb = g.rel[idx]; fb = g.fwd[idx]
            hit = np.isin(b, m1) & (b != drug) & (b != disease)
            for o, bb, rbb, fbb in zip(owner[hit], b[hit], rb[hit], fb[hit]):
                a = int(n1[o])
                if a == disease or a == int(bb):
                    continue
                for k in m_index.get(int(bb), ()):
                    counts["3hop"] += 1
                    found.append(Path_(
                        [drug, a, int(bb), disease],
                        [g.hop_text(drug, a, int(r1[o]), bool(f1[o])),
                         g.hop_text(a, int(bb), int(rbb), bool(fbb)),
                         g.hop_text(int(bb), disease, int(rm[k]), not bool(fm[k]))],
                        [g.hop_dir(int(r1[o]), bool(f1[o])), g.hop_dir(int(rbb), bool(fbb)),
                         g.hop_dir(int(rm[k]), not bool(fm[k]))],
                        (deg[a] * deg[bb]) ** -w,
                    ))

    found.sort(key=lambda p: (-p.weight, p.hops))
    # de-duplicate identical node sequences (keep best-weighted relation text)
    seen, uniq = set(), []
    for p in found:
        key = tuple(p.nodes)
        if key not in seen:
            seen.add(key)
            # mechanistic = the drug acts on a protein first (target/enzyme/transporter/carrier)
            p.kind = "mechanistic" if g.types[p.nodes[1]] == "gene/protein" else "associative"
            uniq.append(p)
    # Reserve half the slots for each kind; fill leftovers from the other kind.
    mech = [p for p in uniq if p.kind == "mechanistic"]
    assoc = [p for p in uniq if p.kind == "associative"]
    half = max_paths // 2
    top = mech[:half] + assoc[:half]
    rest = mech[half:] + assoc[half:]
    rest.sort(key=lambda p: (-p.weight, p.hops))
    top += rest[: max_paths - len(top)]
    top.sort(key=lambda p: (-p.weight, p.hops))
    counts["total"] = counts["2hop"] + counts["3hop"]
    counts["mechanistic"] = len(mech)
    counts["associative"] = len(assoc)
    return top, counts


def node_label(g: KGGraph, v: int) -> str:
    name, typ = str(g.names[v]), str(g.types[v])
    return name if name.endswith(f"({typ})") else f"{name} ({typ})"


def path_to_text(g: KGGraph, p: Path_) -> str:
    parts = [node_label(g, p.nodes[0])]
    for i, rel in enumerate(p.relations):
        v = p.nodes[i + 1]
        d = p.forward[i]
        arrow = f" -[{rel}]- " if d is None else (f" -[{rel}]-> " if d else f" <-[{rel}]- ")
        parts.append(arrow + node_label(g, v))
    return "".join(parts)


def path_to_dict(g: KGGraph, p: Path_) -> dict:
    return {
        "nodes": [int(v) for v in p.nodes],
        "node_names": [str(g.names[v]) for v in p.nodes],
        "node_types": [str(g.types[v]) for v in p.nodes],
        "relations": p.relations,
        "forward": p.forward,
        "weight": float(p.weight),
        "kind": p.kind,
        "text": path_to_text(g, p),
    }
