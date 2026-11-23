# Annotation guide: does the sentence match the paths?

Thank you for helping. This takes about 1–2 hours for 126 rows. **No medical or
biology knowledge is needed** — the task is careful reading, not judging whether
something is medically true.

## What you get
One file: `faithfulness_sheet.csv` (open it in Excel or Google Sheets). Each row has:

| Column | What it is |
|---|---|
| `item` | row number |
| `drug`, `disease` | the pair being explained |
| `paths` | facts from a database, numbered P1, P2, … |
| `statement` | one sentence an AI wrote about this pair |
| `cited_paths` | which paths the AI said it used (may be empty) |
| `your_label` | **you fill this** |
| `notes` | optional, a few words |

## Your task
For each row, ask: **does every claim in `statement` appear in `paths`?**

Type exactly one of:

* `SUPPORTED`: every claim in the sentence is stated in the paths.
* `PARTIAL`: some claims are in the paths, but at least one is not
  (an extra drug/gene/disease, an extra mechanism, or a hedge like
  "may suggest pharmacokinetic interference" that no path states).
* `UNSUPPORTED`: the main claim of the sentence is not in the paths.

Check **all** paths, not only the ones in `cited_paths`.
Do **not** use outside knowledge: even if a sentence is medically true, it is
`UNSUPPORTED` when the paths do not say it.

## Tip: treat names as letters
You do not need to know what "CYP2D6" or "beriberi" is. Replace names with A, B, C:

```
P2: A -[enzyme]-> B <-[enzyme]- C -[is indicated for]-> D
Sentence: "A and C share the enzyme B, and C is indicated for D."
```
Both claims are in P2 → `SUPPORTED`.

## The connector words
| In the path | Plain meaning |
|---|---|
| `is indicated for` | is a treatment for |
| `interacts with drug` | is known to interact with |
| `target` | acts on this gene/protein |
| `enzyme` | is processed by this enzyme |
| `transporter` | is carried by this protein |
| `has side effect` | can cause this symptom |
| `has phenotype` | the disease shows this symptom |
| `associated with` | is linked to |
| `is related to (parent/child)` | is a broader/narrower version of |

A direction arrow does not matter much; "A shares enzyme B with C" is supported by
`A -[enzyme]-> B <-[enzyme]- C`.

## Examples
(Both examples are made up for practice; they are not rows of the sheet.)

1. Path: `Drug A -[transporter]-> Gene X -[associated with]-> Disease D`.
   Sentence: "Drug A is transported by Gene X and Gene Y, which are associated with
   Disease D." The Gene X part is there, but no path links Gene Y to Disease D →
   `PARTIAL`, note "Gene Y not linked".
2. Path: `Cyclosporine -[interacts with drug]- Desmopressin -[is indicated for]-> von Willebrand disease`.
   Sentence: "Cyclosporine increases clotting factor VIII levels." Nothing about factor
   VIII → `UNSUPPORTED`, note "factor VIII not in paths".

## Rules
* Work alone; do not discuss rows with others until you are done.
* If unsure between two labels, pick the one you lean to and write "unsure" in `notes`.
* Save as CSV with the same file name and send it back.
