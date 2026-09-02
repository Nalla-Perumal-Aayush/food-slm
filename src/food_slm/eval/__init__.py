"""Evaluation harness (baseline §9, Phases 31-37).

  perplexity.py     IMPLEMENTED - held-out loss / perplexity
  generate.py       IMPLEMENTED - sampling + decoding for qualitative checks
  suites.py         knowledge / procedural / scaling / substitution / regional /
                    reasoning / hallucination / abstention / generalization-gap
  judge.py          AI-as-a-Judge scoring harness
  report.py         consolidated automated + judge + human report
"""
from food_slm.eval.generate import generate, generate_text
from food_slm.eval.perplexity import evaluate_loss, perplexity, perplexity_report

__all__ = [
    "evaluate_loss",
    "perplexity",
    "perplexity_report",
    "generate",
    "generate_text",
]
