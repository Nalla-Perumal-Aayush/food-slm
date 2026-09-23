"""process_source end-to-end with a FakeProvider (no network, no models)."""
from __future__ import annotations

import json
from pathlib import Path

from food_slm.data.canon_staging import list_staged, promote_staged
from food_slm.data.ingest import Fetcher, FetchResult
from food_slm.data.licensing import classify_license
from food_slm.data.llm import FakeProvider, Panel, ProviderSpec
from food_slm.data.pipeline import (
    build_source_record,
    process_cached_text,
    process_dish_generation,
    process_source,
    resolve_batch_refs,
    source_id_for,
)
from food_slm.schema import load_store

ROOT = Path(__file__).resolve().parents[1]
CANON = ROOT / "data" / "canonical"

_PAGE_HTML = ('<html><head><title>Olan recipe</title></head><body>'
              '<a rel="license" href="https://creativecommons.org/licenses/by-sa/4.0/">CC BY-SA</a>'
              '<p>Olan is a Kerala dish of ash gourd and cowpeas simmered in thin coconut '
              'milk, finished with raw coconut oil and curry leaves.</p></body></html>')
_PAGE_TEXT = ("Olan is a mild Kerala dish of white ash gourd and red cowpeas simmered in thin "
              "coconut milk, finished with raw coconut oil and curry leaves.")

_EXTRACT_JSON = json.dumps({
    "dishes": [{"name_en": "Olan", "category": "kootu", "cuisine_state": "kerala",
                "names": [{"lang": "ml", "native": "ഓലൻ", "transliteration": "olan"}],
                "diet": {"vegetarian": True},
                "summary": "Ash gourd and cowpeas in thin coconut milk.",
                "primary_ingredients": ["ash gourd", "cowpeas", "coconut milk"]}],
    "ingredients": [{"name_en": "Coconut oil zzz", "category": "fat_oil", "default_unit": "tbsp",
                     "summary": "Raw coconut oil drizzled over olan at the end."}],
    "terms": [], "techniques": [], "notes": "cowpea colour varies",
})


class _FakeFetcher(Fetcher):
    def __init__(self):
        pass

    def fetch(self, url, *, refresh=False):
        return FetchResult(url=url, status=200, html=_PAGE_HTML, text=_PAGE_TEXT,
                           fetched_at="2026-09-02T00:00:00")


def _panel(*decs):
    return Panel([FakeProvider(ProviderSpec(f"m{i}", "fake", "x", fam),
                               responder=(lambda d: lambda s, p: f'{{"decision":"{d}"}}')(dec))
                 for i, (dec, fam) in enumerate(zip(decs, ["qwen", "gemma"], strict=False))])


def test_source_id_and_license() -> None:
    assert source_id_for("https://en.wikipedia.org/wiki/Olan") == "src:en-wikipedia-org-olan"
    v = classify_license(_PAGE_HTML, "https://en.wikipedia.org/wiki/Olan")
    assert v.license_class == "share_alike"
    rec = build_source_record("https://en.wikipedia.org/wiki/Olan", v, kind="wikipedia",
                              title="Olan")
    assert rec["structure"]["usable_for"] == ["training", "grounding"]


def test_resolve_batch_refs_drops_unknown_ingredient_refs() -> None:
    recs = [{"id": "dish:x", "type": "dish",
             "content": {"summary": "s"},
             "structure": {"primary_ingredients": ["ing:known", "ing:missing"]}},
            {"id": "ing:known", "type": "ingredient", "structure": {}}]
    out, dropped = resolve_batch_refs(recs, None)
    assert out[0]["structure"]["primary_ingredients"] == ["ing:known"]
    assert dropped == ["dish:x.primary_ingredients:ing:missing"]
    assert "unresolved" in out[0]["content"]["summary"]


def test_process_source_stages_source_plus_records(tmp_path: Path) -> None:
    store = load_store(CANON, strict=True)
    staging = tmp_path / "staging"
    fp = FakeProvider(ProviderSpec("qwen", "fake", "x", "qwen"),
                      responder=lambda s, p: _EXTRACT_JSON)

    run = process_source(
        "https://en.wikipedia.org/wiki/Olan", "wikipedia",
        fetcher=_FakeFetcher(), extract_provider=fp, panel=_panel("accept", "accept"),
        staging_dir=staging, store=store)

    assert run.fetch.ok and run.license.license_class == "share_alike"
    assert run.source_id == "src:en-wikipedia-org-olan"
    assert "dish:olan" in run.staged and "ing:coconut-oil-zzz" in run.staged
    assert run.source_id in run.staged
    # the olan dish's unknown ingredient refs were dropped (ash gourd etc not in store)
    assert run.dropped_refs
    staged = {r["id"]: r for r in list_staged(staging)}
    assert staged["dish:olan"]["structure"]["primary_ingredients"] == []

    # promote and check the store stays valid
    res = promote_staged(list(run.staged), staging_dir=staging,
                         canonical_dir=_copy_store(tmp_path), reviewer="aayush",
                         batch_id="b1", eval_dir=tmp_path / "eval")
    assert res.store_ok, res.issues


def test_process_cached_text_gives_each_doc_its_own_source_record(tmp_path: Path) -> None:
    # canon_extract_dishes.py has no real per-post URL for a blog, so it synthesizes
    # one with a URL *fragment* ("https://site.example/#some-post-slug"). A fragment
    # is not part of urlparse().path, so deriving the source record's id from that
    # placeholder URL the way process_source does (source_id_for) would give every
    # post on the same site an empty path -> the same "index" id, collapsing them
    # all onto one staged source record and breaking the already-staged resume
    # check for every one of them. process_cached_text must use the id it's given.
    store = load_store(CANON, strict=True)
    staging = tmp_path / "staging"
    fp = FakeProvider(ProviderSpec("qwen", "fake", "x", "qwen"),
                      responder=lambda s, p: _EXTRACT_JSON)

    run1 = process_cached_text(
        source_id="src:someblog-olan-recipe", source_url="https://someblog.example/#olan-recipe",
        kind="recipe_blog", text=_PAGE_TEXT, license_class="unknown", title="Olan Recipe",
        extract_provider=fp, panel=_panel("accept", "accept"), staging_dir=staging, store=store)
    run2 = process_cached_text(
        source_id="src:someblog-second-post", source_url="https://someblog.example/#second-post",
        kind="recipe_blog", text=_PAGE_TEXT, license_class="unknown", title="Second Post",
        extract_provider=fp, panel=_panel("accept", "accept"), staging_dir=staging, store=store)

    assert run1.source_id == "src:someblog-olan-recipe"
    assert run2.source_id == "src:someblog-second-post"
    assert run1.source_id != run2.source_id
    assert run1.source_record["id"] == run1.source_id
    assert run2.source_record["id"] == run2.source_id
    staged_ids = {r["id"] for r in list_staged(staging)}
    assert run1.source_id in staged_ids and run2.source_id in staged_ids


def _copy_store(tmp_path: Path) -> Path:
    d = tmp_path / "canonical"
    d.mkdir(exist_ok=True)
    for p in CANON.glob("*.jsonl"):
        (d / p.name).write_text(p.read_text(encoding="utf-8"), encoding="utf-8")
    return d


def test_process_source_skips_on_bad_fetch(tmp_path: Path) -> None:
    class Dead(Fetcher):
        def __init__(self):
            pass

        def fetch(self, url, *, refresh=False):
            return FetchResult(url=url, status=-1, reason="disallowed by robots.txt")

    run = process_source("https://x.test/y", "recipe_blog", fetcher=Dead(),
                         extract_provider=FakeProvider(ProviderSpec("f", "fake", "x", "f")),
                         panel=_panel("accept"), staging_dir=tmp_path)
    assert run.skipped == "disallowed by robots.txt" and not run.staged


def test_process_dish_generation_stages_only_the_synthetic_recipe(tmp_path: Path) -> None:
    """No source: record (there's no document); the recipe is staged as synthetic and
    -- even with a unanimous-accept panel -- can only reach needs_human, since a
    generated record's corroboration is necessarily 0."""
    gen_json = json.dumps({"recipes": [{
        "servings": 4, "difficulty": "easy",
        "ingredients": [{"name": "gongura leaves", "quantity_text": "2 cups", "role": "base"}],
        "steps": [{"action": "Cook the leaves down.", "phase": "cook", "action_verb": "boil",
                  "ingredients_used": ["gongura leaves"]}],
    }], "notes": ""})
    gen_provider = FakeProvider(ProviderSpec("gpt-oss", "fake", "x", "openai"),
                                responder=lambda s, p: gen_json)

    run = process_dish_generation(
        dish_id="dish:gongura-pachadi", dish_name="Gongura Pachadi", cuisine_state="telangana",
        dish_diet={"vegetarian": True, "vegan": True}, generator_label="openai/gpt-oss-120b (Groq)",
        generation_provider=gen_provider, panel=_panel("accept", "accept"), staging_dir=tmp_path)

    assert run.source_record is None
    assert len(run.staged) == 1
    recipe_id = run.staged[0]
    assert recipe_id.startswith("recipe:gongura-pachadi")
    staged = {r["id"]: r for r in list_staged(tmp_path)}
    rec = staged[recipe_id]
    assert rec["provenance"]["synthetic"] is True
    assert rec["quality"]["validation_status"] == "unverified" or \
        rec["quality"]["validation_status"] != "auto_validated"
