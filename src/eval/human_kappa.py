"""Agreement between the faithfulness judge and human labels (Cohen's kappa).

Reads results/human_eval/faithfulness_sheet.csv (column "your_label ...", filled by the
annotator) and faithfulness_key.csv (hidden judge labels), joined on `item`.
Reports 3-class kappa, kappa with PARTIAL merged into UNSUPPORTED (binary "fully
supported?"), raw agreement, per-source (claim / mechanism) breakdown and the confusion
matrix. Writes results/tables/human_kappa.{md,json}.

Usage: python -m src.eval.human_kappa
"""
import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import cohen_kappa_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[2]
HE, TAB = ROOT / "results" / "human_eval", ROOT / "results" / "tables"
LABELS = ["SUPPORTED", "PARTIAL", "UNSUPPORTED"]


def stats(h, j):
    hb, jb = (h == "SUPPORTED"), (j == "SUPPORTED")
    return {"n": int(len(h)), "agree": float((h == j).mean()),
            "kappa_3class": float(cohen_kappa_score(h, j, labels=LABELS)),
            "kappa_binary": float(cohen_kappa_score(hb, jb))}


def main():
    sheet = pd.read_csv(HE / "faithfulness_sheet.csv")
    col = next(c for c in sheet.columns if c.startswith("your_label"))
    key = pd.read_csv(HE / "faithfulness_key.csv")
    df = sheet[["item", col]].merge(key, on="item")
    df["human"] = df[col].astype(str).str.strip().str.upper()
    missing = df[~df["human"].isin(LABELS)]
    if len(missing):
        raise SystemExit(f"{len(missing)} items unlabelled or invalid, e.g. items {missing['item'].head(10).tolist()}")
    out = {"overall": stats(df.human, df.judge_label)}
    for src, g in df.groupby("source"):
        out[src] = stats(g.human, g.judge_label)
    cm = confusion_matrix(df.human, df.judge_label, labels=LABELS)
    out["confusion_human_rows_judge_cols"] = {"labels": LABELS, "matrix": cm.tolist()}
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / "human_kappa.json").write_text(json.dumps(out, indent=1))
    lines = ["| Subset | n | Agreement | kappa (3-class) | kappa (supported vs. not) |", "|---|---|---|---|---|"]
    for k, v in out.items():
        if k.startswith("confusion"):
            continue
        lines.append(f"| {k} | {v['n']} | {v['agree']:.2f} | {v['kappa_3class']:.2f} | {v['kappa_binary']:.2f} |")
    lines += ["", "Confusion (rows human, cols judge): " + ", ".join(LABELS)]
    lines += ["    " + " ".join(f"{x:4d}" for x in row) for row in cm]
    (TAB / "human_kappa.md").write_text("# Judge vs. human labels\n\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
