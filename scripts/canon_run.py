"""Canonical-store extraction pipeline driver (Documents/mcp_extraction_pipeline.md).

  canon_run.py gaps                      # ranked coverage gaps vs coverage_targets.yaml
  canon_run.py run --sources s.json      # fetch -> license -> extract -> validate -> stage
  canon_run.py review                    # list staged records grouped by verdict
  canon_run.py promote --ids A B --reviewer NAME   # staged -> data/canonical/

sources.json:  [{"url": "https://...", "kind": "wikipedia|recipe_blog|gov_doc|..."}, ...]

Source discovery (WebSearch) + licensing approval is a human / Claude-Code step:
run `gaps`, decide what to search for, vet the URLs, then `run` them.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from food_slm.data.canon_staging import list_staged, promote_staged  # noqa: E402
from food_slm.data.coverage import list_gaps  # noqa: E402
from food_slm.data.forbidden import load_forbidden  # noqa: E402
from food_slm.data.ingest import Fetcher  # noqa: E402
from food_slm.data.llm import Panel, build_provider, load_agents_config  # noqa: E402
from food_slm.data.pipeline import process_source  # noqa: E402
from food_slm.schema import load_store  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
CANON = REPO / "data" / "canonical"
STAGING = REPO / "data" / "interim" / "canon_staging"
EVAL = REPO / "data" / "eval"
TARGETS = REPO / "configs" / "coverage_targets.yaml"
AGENTS = REPO / "configs" / "agents.yaml"


def _providers():
    roles = load_agents_config(AGENTS)
    extract = build_provider(roles["extraction_local"][0])
    panel = Panel([build_provider(s) for s in roles["data_validation"]])
    heavy = build_provider(roles["data_validation_heavy"][0]) if roles.get(
        "data_validation_heavy") else None
    return extract, panel, heavy


def cmd_gaps(args) -> int:
    report, gaps = list_gaps(CANON, TARGETS, limit=args.limit)
    print(json.dumps(report.summary(), indent=2))
    print("\nTop gaps:")
    for g in gaps:
        print(f"  [{g.priority_rank:>2}] {g.kind:<20} {g.key:<28} {g.have}/{g.target}")
    return 0


def cmd_run(args) -> int:
    sources = json.loads(Path(args.sources).read_text(encoding="utf-8"))
    store = load_store(CANON, strict=False)
    forbidden = load_forbidden(EVAL)
    fetcher = Fetcher(REPO / ".cache" / "fetch", min_interval_s=args.min_interval)
    extract_p, panel, heavy = _providers()
    print(f"panel: {[p.spec.name for p in panel.providers]}  extractor: {extract_p.spec.name}"
          f"  heavy: {heavy.spec.name if heavy else '-'}\n")

    all_summaries = []
    for entry in sources:
        url, kind = entry["url"], entry.get("kind", "recipe_blog")
        print(f"--- {url}")
        if args.dry_run:
            print("  (dry-run: skipping fetch)")
            continue
        run = process_source(url, kind, fetcher=fetcher, extract_provider=extract_p, panel=panel,
                             staging_dir=STAGING, store=store, forbidden=forbidden,
                             heavy_provider=heavy)
        s = run.summary()
        all_summaries.append(s)
        print(f"  {json.dumps(s)}")

    print("\n=== run summary ===")
    print(json.dumps(all_summaries, indent=2))
    print(f"\nstaged records now in {STAGING} — review with `canon_run.py review`")
    return 0


def cmd_review(args) -> int:
    staged = list_staged(STAGING)
    if not staged:
        print("nothing staged")
        return 0
    by_dec: dict[str, list[str]] = {}
    for rec in staged:
        dec = rec.get("quality", {}).get("validation_status", "?")
        by_dec.setdefault(dec, []).append(rec["id"])
    for dec, ids in sorted(by_dec.items()):
        print(f"\n{dec}  ({len(ids)})")
        for rid in ids:
            print(f"  {rid}")
    print(f"\ntotal staged: {len(staged)}")
    print("promote:  canon_run.py promote --reviewer <name> --ids " + " ".join(
        r["id"] for r in staged if r.get("quality", {}).get("validation_status")
        in ("auto_validated",)))
    return 0


def cmd_promote(args) -> int:
    res = promote_staged(args.ids, staging_dir=STAGING, canonical_dir=CANON,
                         reviewer=args.reviewer, batch_id=args.batch, eval_dir=EVAL)
    print(json.dumps({"promoted": res.promoted, "calibration_sample": res.calibration_sample,
                      "store_ok": res.store_ok, "issues": res.issues}, indent=2))
    if not res.store_ok:
        print("\nSTORE INTEGRITY BROKE — records written but not removed from staging. Fix and retry.")
        return 1
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gaps")
    g.add_argument("--limit", type=int, default=40)
    r = sub.add_parser("run")
    r.add_argument("--sources", required=True)
    r.add_argument("--dry-run", action="store_true")
    r.add_argument("--min-interval", type=float, default=2.0)
    sub.add_parser("review")
    p = sub.add_parser("promote")
    p.add_argument("--ids", nargs="+", required=True)
    p.add_argument("--reviewer", required=True)
    p.add_argument("--batch", default=None)
    args = ap.parse_args()
    return {"gaps": cmd_gaps, "run": cmd_run, "review": cmd_review,
            "promote": cmd_promote}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
