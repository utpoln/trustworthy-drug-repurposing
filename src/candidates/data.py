"""Load the processed PrimeKG split into PyKEEN TriplesFactories."""
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Set, Tuple

import numpy as np
import pandas as pd
import torch
from pykeen.triples import TriplesFactory

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPLIT = ROOT / "data" / "processed" / "primekg_v2.1_seed42"


@dataclass
class SplitData:
    split_dir: Path
    nodes: pd.DataFrame
    train: TriplesFactory
    valid: pd.DataFrame          # head, relation, tail (node_index)
    test: pd.DataFrame
    entity_to_id: Dict[str, int]
    relation_to_id: Dict[str, int]
    drug_ids: np.ndarray         # PyKEEN entity ids of all drug nodes
    known: Dict[int, Set[int]]   # disease node_index -> drug node_indexes (all splits)


def load_split(split_dir: Path = DEFAULT_SPLIT) -> SplitData:
    nodes = pd.read_csv(split_dir / "nodes.tsv.gz", sep="\t", low_memory=False)
    train = pd.read_csv(split_dir / "train.tsv.gz", sep="\t")
    valid = pd.read_csv(split_dir / "valid.tsv", sep="\t")
    test = pd.read_csv(split_dir / "test.tsv", sep="\t")

    entity_to_id = {str(i): k for k, i in enumerate(nodes["node_index"].tolist())}
    relation_to_id = {r: k for k, r in enumerate(sorted(train["relation"].unique()))}

    ent = nodes["node_index"].to_numpy()
    lut = np.full(ent.max() + 1, -1, dtype=np.int64)
    lut[ent] = np.arange(len(ent))
    mapped = torch.as_tensor(np.stack([
        lut[train["head"].to_numpy()],
        train["relation"].map(relation_to_id).to_numpy(),
        lut[train["tail"].to_numpy()],
    ], axis=1), dtype=torch.long)
    tf = TriplesFactory(mapped_triples=mapped, entity_to_id=entity_to_id, relation_to_id=relation_to_id)

    drug_nodes = nodes.loc[nodes["node_type"] == "drug", "node_index"].to_numpy()
    known: Dict[int, Set[int]] = {}
    for df in (train[train["relation"] == "indication"], valid, test):
        for d, s in zip(df["head"], df["tail"]):
            known.setdefault(int(s), set()).add(int(d))

    return SplitData(split_dir, nodes, tf, valid, test, entity_to_id, relation_to_id,
                     lut[drug_nodes], known)


def node_lut(data: SplitData) -> Tuple[np.ndarray, np.ndarray]:
    """(node_index -> entity id) and (entity id -> node_index) arrays."""
    ent = data.nodes["node_index"].to_numpy()
    lut = np.full(ent.max() + 1, -1, dtype=np.int64)
    lut[ent] = np.arange(len(ent))
    return lut, ent
