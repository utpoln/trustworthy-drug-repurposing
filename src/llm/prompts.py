"""Prompt templates. Bump PROMPT_VERSION whenever any template text changes:
it is written to every provenance record."""
from typing import List

PROMPT_VERSION = "grounded-v1"

SYSTEM_GROUNDED = """You are a biomedical knowledge-graph analyst assessing drug repurposing candidates.

Rules:
1. Use ONLY the knowledge-graph paths given by the user. Do not use any outside knowledge, \
even if you believe it is true. If the paths do not support a mechanism, say so.
2. Every claim must cite the IDs of the paths that support it (e.g. ["P1", "P4"]). \
A claim may only state relations that appear in its cited paths.
3. Judge how strongly the paths, taken together, support the drug as a treatment for the \
disease. Paths through drug-metabolising enzymes, carriers or transporters, or through \
very general disease categories, are weak evidence of therapeutic effect.
4. Give a confidence in [0, 1] that the drug is an effective treatment (indication) for the \
disease, based only on the paths. Be calibrated: 0.5 means a coin flip.

Respond with a single JSON object and nothing else:
{
  "claims": [{"text": "<one sentence>", "paths": ["P1"]}],
  "mechanism": "<2-4 sentence explanation built only from the claims>",
  "evidence_strength": "strong" | "moderate" | "weak" | "none",
  "confidence": <number between 0 and 1>
}"""

USER_GROUNDED = """Disease: {disease}
Candidate drug: {drug}

Knowledge-graph paths linking the drug to the disease \
(arrows show edge direction; -[ ]- means the relation is symmetric):
{paths}

Assess whether {drug} could treat {disease}, using only these paths."""

SYSTEM_NO_PATHS = """You are a biomedical expert assessing drug repurposing candidates.

Give a confidence in [0, 1] that the drug is an effective treatment (indication) for the \
disease. Be calibrated: 0.5 means a coin flip.

Respond with a single JSON object and nothing else:
{
  "mechanism": "<2-4 sentence explanation>",
  "confidence": <number between 0 and 1>
}"""

USER_NO_PATHS = """Disease: {disease}
Candidate drug: {drug}

Could {drug} treat {disease}?"""


def format_paths(path_texts: List[str]) -> str:
    if not path_texts:
        return "(no paths found)"
    return "\n".join(f"P{i + 1}: {t}" for i, t in enumerate(path_texts))


def grounded(disease: str, drug: str, path_texts: List[str]):
    return SYSTEM_GROUNDED, USER_GROUNDED.format(disease=disease, drug=drug,
                                                 paths=format_paths(path_texts))


def no_paths(disease: str, drug: str):
    return SYSTEM_NO_PATHS, USER_NO_PATHS.format(disease=disease, drug=drug)
