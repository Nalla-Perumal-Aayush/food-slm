"""Dish-seed integrity + transcript-cleaning unit tests."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SEED_DIR = REPO / "data" / "dish_seed"
REGIONS = ["tamil_nadu", "kerala", "karnataka", "andhra_pradesh", "telangana",
           "puducherry", "pan_south"]
_DIET_STR = {"veg", "vegan"}
_DIET_TOK = {"poultry", "seafood", "red_meat", "beef", "egg"}


def test_every_region_seed_present_and_parses():
    for r in REGIONS:
        f = SEED_DIR / f"{r}.yaml"
        assert f.exists(), f"missing {f}"
        doc = yaml.safe_load(f.read_text(encoding="utf-8"))
        assert doc["region"] == r
        assert len(doc["dishes"]) > 20, r


def test_seed_entries_well_formed():
    for r in REGIONS:
        doc = yaml.safe_load((SEED_DIR / f"{r}.yaml").read_text(encoding="utf-8"))
        for e in doc["dishes"]:
            assert e.get("name"), f"{r}: entry with no name"
            assert "cat" in e, f"{r}: {e['name']} has no cat"
            d = e.get("diet")
            if isinstance(d, str):
                assert d in _DIET_STR, f"{r}: {e['name']} bad diet {d!r}"
            else:
                assert isinstance(d, list) and d, f"{r}: {e['name']} bad diet {d!r}"
                assert set(d) <= _DIET_TOK, f"{r}: {e['name']} bad diet tokens {d!r}"


def test_build_dish_list_runs_and_hits_target_band():
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "build_dish_list.py")],
        capture_output=True, text=True, cwd=REPO, timeout=180)
    assert out.returncode == 0, out.stderr
    doc = yaml.safe_load((REPO / "configs" / "dish_list_v1.yaml").read_text(encoding="utf-8"))
    n = len(doc["dishes"])
    assert 850 <= n <= 1150, f"dish count {n} outside review band"
    ids = [d["id"] for d in doc["dishes"]]
    assert len(ids) == len(set(ids)), "duplicate dish ids"
    for d in doc["dishes"]:
        assert d["region"] in REGIONS
        assert 1 <= d["prominence"] <= 10


def test_clean_caption_text_dedups_and_strips():
    from food_slm.data.transcript import clean_caption_text

    snips = [
        {"text": "today we are making", "start": 0, "duration": 1},
        {"text": "we are making ven pongal", "start": 1, "duration": 1},
        {"text": "[Music]", "start": 2, "duration": 1},
        {"text": ">> add one cup of rice add one cup of rice", "start": 3, "duration": 1},
    ]
    out = clean_caption_text(snips)
    assert "[Music]" not in out
    assert ">>" not in out
    assert out.lower().count("ven pongal") == 1
    assert out.lower().count("add one cup of rice") == 1


def test_title_is_food():
    from food_slm.data.transcript import title_is_food

    assert title_is_food("Ven Pongal Recipe in Tamil", None)
    assert title_is_food("Kerala Fish Curry | Meen Curry", None)
    assert not title_is_food("My house tour 2026", None)
