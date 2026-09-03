"""Source license classifier (Documents/mcp_extraction_pipeline.md §4).

Inspects a page + its site for licence signals and maps them to the schema's
``license_class`` / ``usable_for``. Conservative: an unrecognised page is
``unknown`` -> ``reference_only`` (facts may still be extracted and re-expressed;
the prose is never trained on).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from urllib.parse import urlparse

# license_class -> usable_for  (mirrors configs/canonical_enums.yaml)
USABLE_FOR = {
    "public_domain": ["training", "grounding"],
    "permissive": ["training", "grounding"],
    "share_alike": ["training", "grounding"],
    "noncommercial": ["grounding", "reference_only"],
    "unknown": ["reference_only"],
    "all_rights_reserved": ["reference_only"],
}

_CC_PATTERNS = [
    (r"creativecommons\.org/publicdomain/zero", "public_domain", "CC0-1.0"),
    (r"creativecommons\.org/publicdomain/mark", "public_domain", "CC-PDM-1.0"),
    (r"creativecommons\.org/licenses/by-nc-sa", "noncommercial", "CC-BY-NC-SA"),
    (r"creativecommons\.org/licenses/by-nc-nd", "noncommercial", "CC-BY-NC-ND"),
    (r"creativecommons\.org/licenses/by-nc", "noncommercial", "CC-BY-NC"),
    (r"creativecommons\.org/licenses/by-sa", "share_alike", "CC-BY-SA"),
    (r"creativecommons\.org/licenses/by-nd", "noncommercial", "CC-BY-ND"),
    (r"creativecommons\.org/licenses/by(?![-\w])", "permissive", "CC-BY"),
]
_PD_PHRASES = [
    r"\bpublic domain\b", r"\bno known copyright\b", r"released into the public domain",
]
_PERMISSIVE_PHRASES = [
    r"\bMIT license\b", r"\bApache License\b", r"Open Government Licence", r"\bODC-By\b",
    r"GNU Free Documentation License",
]
_ARR_PHRASES = [
    r"all rights reserved", r"unauthori[sz]ed (use|reproduction|copying)",
    r"may not be (reproduced|copied|republished)",
]
_GOV_TLDS = (".gov", ".gov.in", ".nic.in", ".gov.uk")


@dataclass
class LicenseVerdict:
    license_class: str
    license_string: str
    usable_for: list[str]
    signals: list[str] = field(default_factory=list)
    confidence: float = 0.5

    @property
    def corpus_eligible(self) -> bool:
        return "training" in self.usable_for


def _first(patterns, text: str):
    for pat, cls, name in patterns:
        if re.search(pat, text, re.IGNORECASE):
            return cls, name, pat
    return None


def classify_license(html: str, url: str = "") -> LicenseVerdict:
    signals: list[str] = []
    host = urlparse(url).netloc.lower()

    # 1. explicit CC / license URLs anywhere in the markup (link rel=license, anchors, JSON-LD)
    hit = _first(_CC_PATTERNS, html)
    if hit:
        cls, name, pat = hit
        signals.append(f"cc-url:{pat}")
        return LicenseVerdict(cls, name, USABLE_FOR[cls], signals, 0.9)

    # 2. JSON-LD "license" field
    for m in re.finditer(r'"license"\s*:\s*"([^"]+)"', html):
        val = m.group(1)
        signals.append(f"jsonld-license:{val}")
        sub = classify_license(val, url)
        if sub.license_class != "unknown":
            sub.signals = signals + sub.signals
            return sub

    # 3. permissive / PD / ARR phrases in visible text
    if any(re.search(p, html, re.IGNORECASE) for p in _PERMISSIVE_PHRASES):
        signals.append("permissive-phrase")
        return LicenseVerdict("permissive", "stated-permissive", USABLE_FOR["permissive"],
                              signals, 0.7)
    if any(re.search(p, html, re.IGNORECASE) for p in _PD_PHRASES):
        signals.append("public-domain-phrase")
        return LicenseVerdict("public_domain", "stated-public-domain",
                              USABLE_FOR["public_domain"], signals, 0.7)

    # 4. government / public-body domains — usually open, but flag for human confirm
    if host.endswith(_GOV_TLDS):
        signals.append(f"gov-tld:{host}")
        return LicenseVerdict("unknown", "gov-domain-review",
                              USABLE_FOR["unknown"], signals, 0.4)

    # 5. explicit "all rights reserved"
    if any(re.search(p, html, re.IGNORECASE) for p in _ARR_PHRASES):
        signals.append("all-rights-reserved")
        return LicenseVerdict("all_rights_reserved", "all-rights-reserved",
                              USABLE_FOR["all_rights_reserved"], signals, 0.8)

    # 6. default: unknown -> reference_only
    return LicenseVerdict("unknown", "unclassified", USABLE_FOR["unknown"],
                          signals or ["no-signal"], 0.3)


def verdict_to_source_structure(v: LicenseVerdict, *, kind: str, authority: str = "tertiary",
                                accessed_on: str | None = None) -> dict:
    """Shape a `SourceStructure` dict from a verdict (for `register_source`)."""
    return {
        "kind": kind,
        "license": v.license_string,
        "license_class": v.license_class,
        "usable_for": v.usable_for,
        "authority": authority,
        "accessed_on": accessed_on,
    }


def summarize(verdicts: list[tuple[str, LicenseVerdict]]) -> str:
    return json.dumps(
        {url: {"class": v.license_class, "usable_for": v.usable_for, "conf": v.confidence}
         for url, v in verdicts}, indent=2)
