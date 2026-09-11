"""Extraction: page JSON -> valid schema-v2 draft records (FakeProvider, no network)."""
from __future__ import annotations

import json

from food_slm.data.extract import (
    ExtractResult,
    build_extraction_prompt,
    extract_records,
    parse_extraction,
    slugify,
)
from food_slm.data.llm import FakeProvider, ProviderSpec
from food_slm.schema.models import parse_record

_MODEL_JSON = {
    "dishes": [{
        "name_en": "Bisi Bele Bath", "category": "rice_dish", "cuisine_state": "karnataka",
        "names": [{"lang": "kn", "native": "ಬಿಸಿ ಬೇಳೆ ಬಾತ್", "transliteration": "bisi bele bath"}],
        "diet": {"vegetarian": True},
        "summary": "A hot spiced rice-and-lentil one-pot dish from Karnataka.",
        "primary_ingredients": ["rice", "toor dal", "tamarind"],
    }],
    "ingredients": [{
        "name_en": "Toor dal", "category": "dal_legume", "default_unit": "cup",
        "summary": "Split pigeon pea, the standard dal for sambar and bisi bele bath.",
    }],
    "techniques": [{"name_en": "Tempering", "family": "tempering",
                    "summary": "Blooming spices in hot fat, poured over the dish."}],
    "terms": [{"lang": "kn", "native": "ಹುಳಿ", "transliteration": "huli",
               "gloss": "A tamarind-based Karnataka curry/sambar.", "category": "dish"}],
    "notes": "coconut amount not specified",
}


def test_slugify() -> None:
    assert slugify("Bisi Bele Bath") == "bisi-bele-bath"
    assert slugify("Ennai Kathirikai Kuzhambu!") == "ennai-kathirikai-kuzhambu"
    assert slugify("ಹುಳಿ") == "unnamed"          # non-ascii -> fallback


def test_build_extraction_prompt_shape() -> None:
    sys_, user = build_extraction_prompt("some page text about pongal", "src:blog-x")
    assert "STRUCTURED CULINARY FACTS" in sys_
    assert "src:blog-x" in user and "pongal" in user


def test_parse_extraction_produces_valid_records() -> None:
    res = parse_extraction(_MODEL_JSON, "src:wikipedia-bbb")
    assert isinstance(res, ExtractResult)
    assert res.by_type == {"dish": 1, "ingredient": 1, "technique": 1, "term": 1}
    assert res.notes == "coconut amount not specified"
    ids = {d["id"] for d in res.drafts}
    assert {"dish:bisi-bele-bath", "ing:toor-dal", "tech:tempering", "term:huli"} == ids
    for d in res.drafts:
        rec = parse_record(d)                       # every draft is schema-v2 valid
        assert rec.provenance.source_ids == ["src:wikipedia-bbb"]
        assert rec.quality.validation_status == "unverified"
    dish = next(d for d in res.drafts if d["type"] == "dish")
    assert dish["structure"]["primary_ingredients"] == ["ing:rice", "ing:toor-dal", "ing:tamarind"]


def test_parse_extraction_coerces_unknown_enum_values() -> None:
    # A local 7-9B extractor free-texts enum fields fairly often ("breakfast" for a
    # dish category); an otherwise-good record must not be thrown away for that —
    # it's coerced to the schema's fallback value instead of rejected.
    ok = {"dishes": [{"name_en": "Mystery", "category": "not_a_real_category",
                      "cuisine_state": "karnataka", "diet": {"vegetarian": True}}]}
    res = parse_extraction(ok, "src:x")
    assert res.drafts and not res.invalid
    assert res.drafts[0]["structure"]["category"] == "other"


def test_parse_extraction_routes_bad_records_to_invalid() -> None:
    # Genuinely malformed (names must be a list of {lang,native,transliteration}
    # objects) still fails schema validation and is not silently coerced.
    bad = {"dishes": [{"name_en": "Mystery", "category": "rice_dish",
                       "cuisine_state": "karnataka", "diet": {"vegetarian": True},
                       "names": "not-a-list"}]}
    res = parse_extraction(bad, "src:x")
    assert not res.drafts and res.invalid
    assert "schema" in res.invalid[0][1]


def test_community_tag_in_extraction_is_rejected_by_schema() -> None:
    bad = {"dishes": [{"name_en": "Iyengar Puliyogare", "category": "rice_dish",
                       "cuisine_state": "karnataka", "diet": {"vegetarian": True}}]}
    res = parse_extraction(bad, "src:x")
    assert not res.drafts
    assert "community/religion" in res.invalid[0][1]


def test_extract_records_via_provider() -> None:
    fp = FakeProvider(ProviderSpec("fake", "fake", "x", "fake"),
                      responder=lambda s, p: json.dumps(_MODEL_JSON))
    res = extract_records("page text", "src:blog", fp)
    assert res.by_type["dish"] == 1
    assert res.drafts[0]["provenance"]["source_ids"] == ["src:blog"]


def test_extract_records_handles_unparseable_output() -> None:
    fp = FakeProvider(ProviderSpec("fake", "fake", "x", "fake"),
                      responder=lambda s, p: "sorry, I cannot help with that")
    res = extract_records("page text", "src:blog", fp)
    assert not res.drafts and "unparseable" in res.notes
