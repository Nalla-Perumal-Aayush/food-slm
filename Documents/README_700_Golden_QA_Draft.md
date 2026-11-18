# Golden QA — 700 Candidate Preparation Bank

**STATUS: DRAFT / PENDING SOURCE VERIFICATION — NOT A FINISHED GOLDEN QA SET**

This package contains exactly 700 records using the project's documented JSONL field structure.

The Contributor Reference Guide requires every factual answer to be backed by a real, checkable source and an exact supporting quotation, followed by human review. I have therefore **not fabricated 700 citations or quotations**. Candidates that require factual verification contain explicit source placeholders.

## Task distribution
- recipe_request: 100
- scaling: 80
- ingredient_substitution: 80
- troubleshooting: 80
- technique_explanation: 90
- regional_comparison: 90
- definition: 130
- abstention: 50

## Required conversion to project-ready data
For every factual candidate:
1. Replace the source placeholder with a real source.
2. Copy the exact sentence(s) supporting the answer into `source.quote`.
3. Rewrite `candidate_answer` so every claim is directly supported.
4. Do not add quantities, times, substitutions, causes, or comparisons that the source does not state.
5. Human-review every candidate and retain `status: pending` until the project's review process approves it.

The XLSX is provided for convenient manual verification/editing; the JSONL is the closer representation of the project's target structure.
