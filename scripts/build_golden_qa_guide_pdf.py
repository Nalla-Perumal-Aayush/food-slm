"""Builds Documents/Golden_QA_Contributor_Guide.pdf -- the onboarding reference
for anyone contributing golden-QA-set candidates, written so someone with zero
prior context on this project can read it and start contributing accurately.

    python scripts/build_golden_qa_guide_pdf.py

Content lives in this file as structured data (SECTIONS below); the fpdf2
rendering code just walks it. Keeps the writing separate from the layout.
"""
from __future__ import annotations

from pathlib import Path

from fpdf import FPDF

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "Documents" / "Golden_QA_Contributor_Guide.pdf"

TITLE = "Golden QA Set -- Contributor Reference Guide"
SUBTITLE = "South Indian Cuisine SLM Project"

# ---------------------------------------------------------------------------
# Content. Each section is (heading, level, blocks), where a block is one of:
#   ("p", text)                      -- a paragraph
#   ("b", [items])                   -- a bullet list
#   ("table", [header, *rows])        -- a simple table
#   ("example", dict)                 -- a worked Q/A example card
# ---------------------------------------------------------------------------

SECTIONS = [

("1. What this project is, and why this document exists", 1, [
("p", "This project is building a small, specialized AI language model (an \"SLM\") "
     "whose only job is to know about South Indian cuisine -- the food of Tamil Nadu, "
     "Kerala, Karnataka, Andhra Pradesh, Telangana, and Puducherry, covering both "
     "vegetarian and non-vegetarian cooking. You do not need to understand anything "
     "about how the AI model itself works to contribute here."),
("p", "To know whether the finished model is actually any good, we need a trustworthy "
     "\"answer key\": a set of real questions paired with real, verified-correct answers. "
     "This answer key is called the golden QA set. Every answer the model gives can be "
     "compared against it to measure accuracy."),
("p", "The golden QA set has one rule that overrides everything else in this document: "
     "every answer must come from a real, checkable source -- never from memory, never "
     "invented, and never written by an AI without a human checking it against that "
     "source. The reason is specific: this same answer key is also used to check whether "
     "the automated AI reviewers in this project are working correctly. If the answer key "
     "itself had been written by an AI, that check would be meaningless -- it would just "
     "be an AI grading itself. A human-sourced answer key is the one thing in this whole "
     "pipeline that has to come from outside the AI system entirely."),
("p", "This means anyone can contribute here, regardless of technical background. What "
     "matters is care and accuracy in finding and representing a real source -- not "
     "familiarity with the project, with AI, or with cooking."),
]),

("2. The eight kinds of question we need", 1, [
("p", "Every candidate you write needs to fit one of eight shapes. Each one tests the "
     "model on a different kind of real-world question someone might actually ask a "
     "South Indian cooking assistant."),
]),

("2.1 Recipe request", 2, [
("p", "Someone asks how to make a dish. The answer lists the ingredients (with "
     "quantities, if your source gives them) and the steps in order."),
("example", {
    "q": "How do I make idli?",
    "a": "Idli batter is made from idli rice (or parboiled rice) and de-husked black "
        "gram, roughly in a 4:1 ratio, ground and left to ferment overnight -- the batter "
        "should more than double in volume. The fermented batter is then poured into a "
        "perforated, greased idli tray and steamed, typically for 10-25 minutes "
        "depending on size.",
    "source": "https://en.wikipedia.org/wiki/Idli",
    "quote": "four parts uncooked rice...to one part whole white lentil...The finished "
            "idli batter is put into a perforated greased idli tray for steaming...about "
            "10-25 minutes, depending on size.",
}),
]),

("2.2 Scaling", 2, [
("p", "Someone wants to adjust a recipe for a different number of people. Needs a source "
     "that states an actual serving size and real quantities -- an encyclopedia entry "
     "about a dish usually will NOT have this; you need something closer to an actual "
     "recipe or cookbook. This is one of the task types we need the most help with right "
     "now (see section 6)."),
("example", {
    "q": "How do I make this for 8 people instead of 4?",
    "a": "Scale every ingredient by the same ratio (here, 2x), but don't scale cooking "
        "time, salt, or spice by the same amount -- taste and adjust those near the end "
        "instead.",
    "source": "(needs a real recipe source with stated servings and quantities)",
    "quote": "",
}),
]),

("2.3 Ingredient substitution", 2, [
("p", "Someone doesn't have an ingredient and wants a replacement. Needs a source that "
     "explicitly discusses substituting one ingredient for another -- not just two "
     "ingredients that happen to appear in the same dish."),
("example", {
    "q": "What can I use instead of garlic and onion?",
    "a": "Asafoetida is the traditional substitute -- it's used specifically to stand in "
        "for the savoury depth garlic and onion would otherwise add, which is why it's "
        "common in onion-garlic-free South Indian cooking.",
    "source": "https://en.wikipedia.org/wiki/Asafoetida",
    "quote": "asafoetida is primarily used in Indian and Middle Eastern cuisines, where "
            "it is employed to complement or substitute for alliaceous ingredients such "
            "as garlic and onion.",
}),
]),

("2.4 Troubleshooting", 2, [
("p", "Someone's cooking went wrong and wants to know why, and how to fix/prevent it. "
     "Needs a source that actually explains a failure mode -- not just a description of "
     "the dish when it goes right. Encyclopedia articles rarely cover this; food-science "
     "sources, cookbooks, and reputable cooking guides are better fits. This is the other "
     "task type we need the most help with right now."),
("example", {
    "q": "My idli batter isn't rising. What's wrong?",
    "a": "Idli batter rises because lactic acid bacteria ferment it, producing the carbon "
        "dioxide that makes it rise. Cold temperatures slow that bacterial activity down "
        "or stop it, which is the most common reason batter fails to rise. Fix: move it "
        "somewhere consistently warm and give it more time.",
    "source": "https://en.wikipedia.org/wiki/Idli",
    "quote": "The fermentation process is performed by lactic acid bacteria...which "
            "generate both lactic acid and carbon dioxide.",
}),
]),

("2.5 Technique explanation", 2, [
("p", "Someone wants to understand a cooking method: what it is, how it's done, and how "
     "you know it's done right."),
("example", {
    "q": "What is tempering (tadka)?",
    "a": "Tempering is briefly frying whole spices -- like mustard seeds, cumin, or curry "
        "leaves -- in hot oil or ghee to release their essential oils, then pouring the "
        "spiced oil into the dish. It's usually done at the start of cooking, though for "
        "dals or sambar it's often done at the end instead, right before serving.",
    "source": "https://en.wikipedia.org/wiki/Tadka",
    "quote": "whole spices...are cooked briefly in oil or ghee to liberate essential oils "
            "from cells and thus enhance their flavours, before being poured, together "
            "with the oil, into a dish.",
}),
]),

("2.6 Regional comparison", 2, [
("p", "Someone asks how a dish or style differs between two regions. Needs a source that "
     "explicitly discusses both versions, ideally in the same place -- don't combine two "
     "separate sources' opinions into a comparison neither of them actually makes."),
("example", {
    "q": "Is thoran the same thing as poriyal?",
    "a": "Essentially yes -- thoran (Kerala) and poriyal (Tamil Nadu) are the same style "
        "of dish: vegetables finely chopped, sauteed briefly with grated coconut, "
        "mustard seeds, and curry leaves. The name just changes with the region.",
    "source": "https://en.wikipedia.org/wiki/Thoran",
    "quote": "This common dish is usually eaten with rice and curry...In Tamil Nadu, a "
            "similar dish is known as Poriyal.",
}),
]),

("2.7 Definition / factual", 2, [
("p", "A short, direct factual question about a dish, ingredient, or term. The most "
     "forgiving task type to source -- most encyclopedia-style content supports this "
     "well."),
("example", {
    "q": "What is gongura?",
    "a": "Gongura is a variety of roselle plant grown for its tart, edible leaves. It's "
        "especially associated with Andhra cuisine, most famously in gongura pachadi, a "
        "tangy chutney/pickle made from the leaves.",
    "source": "https://en.wikipedia.org/wiki/Gongura",
    "quote": "These leaves are used in south-central Indian cuisine to impart a tart "
            "flavour. Gongura pacchadi, a form of chutney or relish, is a quintessential "
            "part of Andhra cuisine.",
}),
]),

("2.8 Out-of-scope / abstention", 2, [
("p", "Someone asks something the model should NOT try to answer, because it's outside "
     "South Indian cuisine entirely (general trivia, other cuisines, medical advice, "
     "etc). The correct answer is a polite, honest decline -- not a guess. These items "
     "don't need a source citation, since the point is the behaviour (declining "
     "gracefully), not a fact."),
("example", {
    "q": "How do I fix my car's brakes?",
    "a": "I can't help with that -- I'm a South Indian cuisine specialist, not a "
        "mechanic.",
    "source": "(none needed -- this tests behaviour, not a fact)",
    "quote": "",
}),
]),

("3. What we're covering, and how a candidate gets tagged", 1, [
("p", "Every candidate needs three tags besides its task type:"),
("b", [
    "Region: tamil_nadu, kerala, karnataka, andhra_pradesh, telangana, puducherry, or "
    "pan_south (for something common across the South generally, not tied to one state). "
    "Leave this blank/none for ingredient or technique questions that aren't "
    "region-specific.",
    "Diet: vegetarian, non_vegetarian, or na (for items like abstention, where diet "
    "doesn't apply).",
    "Dish IDs (optional): which canonical dish(es) this relates to, if you know the "
    "project's internal naming. Fine to leave blank if you don't -- whoever enters your "
    "candidate can fill this in.",
]),
("p", "The goal is balanced coverage: all six regions (plus pan-south), a real mix of "
     "vegetarian and non-vegetarian, and a reasonable spread across all eight task types "
     "-- not just whichever is easiest to source. The overall target is roughly "
     "500 to 1,000 approved items in total."),
]),

("4. Where to find a source -- and where not to", 1, [
("p", "This is the most important section. The quality of the whole golden set depends "
     "on this."),
("p", "Good sources (use these):"),
("b", [
    "Wikipedia articles about a specific dish, ingredient, technique, or regional "
    "cuisine (not a disambiguation page, and not the article's talk/discussion section).",
    "Government or state tourism board pages about regional food culture.",
    "Published cookbooks (cite the book title, author, and page/recipe name).",
    "Peer-reviewed academic or food-science articles (useful especially for "
    "troubleshooting and technique questions -- these often explain the WHY behind a "
    "cooking result in a way encyclopedias don't).",
    "Established, edited culinary reference works or encyclopedias.",
]),
("p", "Sources to avoid:"),
("b", [
    "Random food blogs or social media posts with no editorial oversight -- even if the "
    "information sounds plausible, there's no way to verify it's accurate.",
    "Anything written by an AI chatbot (including this project's own model, and "
    "including asking a general-purpose AI assistant \"what's the recipe for X\").",
    "Your own memory or assumption, unless you can also find it stated in a real source. "
    "(If you genuinely know a fact firsthand -- e.g. you've made the dish yourself many "
    "times -- that's valuable, but pair it with a citable source rather than relying on "
    "memory alone for THIS specific document; personal expertise is more useful "
    "elsewhere in the project, like reviewing other people's candidates.)",
    "A source that doesn't actually say what you're citing it for. Always re-read the "
    "exact sentence you're quoting before submitting -- it's easy to misremember what a "
    "page said after reading several at once.",
]),
("p", "A simple test: if someone challenged your answer and asked \"how do you know "
     "that?\", could you point to the exact sentence, in the exact source, that says so? "
     "If yes, you're ready to submit it. If no, keep looking or pick a different fact."),
]),

("5. How to actually write a candidate, step by step", 1, [
("b", [
    "1. Pick a dish, ingredient, technique, or comparison within South Indian cuisine.",
    "2. Find a real source (section 4) that states a specific, checkable fact about it.",
    "3. Decide which of the eight task types (section 2) this fact fits.",
    "4. Write a natural question -- the way a real person would actually ask it, not "
    "stiff or textbook-like. Vary your phrasing across candidates rather than reusing "
    "the exact same sentence pattern every time.",
    "5. Write an answer using ONLY what the source actually supports. Do not add "
    "specifics (an exact quantity, a precise time, a confident-sounding detail) that "
    "the source doesn't actually give, even if it seems like a safe guess.",
    "6. Copy the exact sentence(s) from the source that back up your answer, word for "
    "word, along with the page's URL (or the book/article citation).",
    "7. Tag the region and diet (section 3).",
    "8. Hand it off for entry into the review queue (section 7) -- or enter it yourself "
    "if you have access to the project's files; see section 8 for the exact format.",
]),
("p", "Common mistakes to avoid:"),
("b", [
    "Combining a real cited fact with an extra detail from memory -- the whole answer "
    "needs to trace back to the source, not just part of it.",
    "Writing an answer that's technically true but not what the source actually "
    "discusses (e.g. citing a dish's Wikipedia page for a claim that's really common "
    "knowledge, not stated on that page).",
    "Asking a leading or biased question (\"Isn't it true that...\") instead of a "
    "neutral one.",
    "Writing an essay-length answer. Keep it to a few sentences -- concise and accurate "
    "beats thorough and padded.",
    "Stating a personal opinion about taste or quality (\"the best way to make...\") as "
    "if it were a verified fact.",
]),
]),

("6. What's needed most right now", 1, [
("p", "As of when this document was written, two task types were especially thin because "
     "they're genuinely harder to source well:"),
("b", [
    "Scaling -- needs a source with real, stated quantities and a serving size (an "
    "actual recipe or cookbook, not an encyclopedia entry).",
    "Troubleshooting -- needs a source that explains a failure mode, which most "
    "encyclopedia-style writing doesn't cover. Food-science articles and well-edited "
    "cooking guides are the best bet.",
    "Ingredient substitution was also thin -- it needs a source that explicitly "
    "discusses swapping one ingredient for another, which is less commonly written "
    "about than you'd expect.",
]),
("p", "Region-wise, Puducherry and Telangana had less material than Tamil Nadu and "
     "Kerala, simply because less has been written about them in the sources checked so "
     "far. Non-vegetarian coverage was also behind vegetarian."),
("p", "These specifics will go stale as more candidates get added -- check the review "
     "tool's live coverage table (section 7) for the current picture before picking what "
     "to work on."),
]),

("7. What happens after you submit a candidate", 1, [
("p", "Every single candidate gets reviewed individually by a human before it counts as "
     "part of the golden set -- there is no sampling or batch-approval, by design (this "
     "is the human-verification step referenced in section 1)."),
("p", "The review happens in a small web tool built for this project (run as "
     "\"streamlit run scripts/golden_qa_review.py\" by someone with access to the "
     "project). For each pending candidate, the reviewer sees the question, the proposed "
     "answer (which they can edit before approving), and the cited source, and either "
     "approves it, rejects it, or leaves it pending. Approved items become part of the "
     "real golden set; rejected ones are kept on record with the reason, not deleted. "
     "The tool also shows a running count toward the target and a table of which "
     "region/task-type combinations still need coverage."),
("p", "If you're contributing content but don't have access to that tool, that's fine: "
     "hand your question, answer, and source citation to whoever does, in whatever "
     "format is convenient (a document, a spreadsheet, a message) -- they can enter it "
     "for you."),
]),

("8. The data format, for reference", 1, [
("p", "If you do have access to the project files, each candidate is one line of JSON "
     "with these fields:"),
("table", [
    ["Field", "What goes here"],
    ["task_type", "One of: recipe_request, scaling, ingredient_substitution, "
     "troubleshooting, technique_explanation, regional_comparison, definition, "
     "abstention"],
    ["region", "One of: tamil_nadu, kerala, karnataka, andhra_pradesh, telangana, "
     "puducherry, pan_south, or null if not region-specific"],
    ["diet", "vegetarian, non_vegetarian, or na"],
    ["question", "The question, as plain text"],
    ["candidate_answer", "The answer, as plain text"],
    ["source.url", "The web address of the source (or a book citation if not online)"],
    ["source.quote", "The exact sentence(s) from the source backing the answer"],
    ["dish_ids", "Optional list of related canonical dish ids"],
    ["status", "Always start as \"pending\" -- the review tool changes this"],
]),
("p", "New candidates get added by running \"python scripts/draft_golden_candidates.py\" "
     "after adding entries to that file's CANDIDATES list, or appended directly to "
     "data/eval/golden_qa_candidates.jsonl in this shape. Sources themselves are fetched "
     "and saved via \"python scripts/fetch_golden_sources.py\" before drafting against "
     "them, so every citation traces back to an actual saved, re-checkable page."),
]),

("9. Frequently asked questions", 1, [
("b", [
    "What if I can't find a good source for the dish/fact I had in mind? -- Pick a "
    "different one. There's no shortage of South Indian dishes, ingredients, and "
    "techniques to choose from; it's not worth stretching a weak source.",
    "What if two sources disagree? -- Prefer the more specific, more authoritative one "
    "(a focused academic or cookbook source usually beats a general encyclopedia entry "
    "on a narrow point). If you can't tell which is right, skip that fact rather than "
    "guess.",
    "Can I write about a dish that isn't in the project's existing dish list? -- Yes, as "
    "long as it's genuinely South Indian cuisine from one of the six regions (or common "
    "across them). New dishes are welcome.",
    "What language should the question and answer be in? -- English. Native-script or "
    "transliterated dish/ingredient names (e.g. \"gongura\", \"kuzhambu\") are fine and "
    "expected.",
    "Do I need to know anything about AI or machine learning to help? -- No. This task "
    "is entirely about finding accurate, well-cited facts and writing clear questions "
    "around them.",
]),
]),

]


def render(pdf: FPDF) -> None:
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    pdf.set_font("Helvetica", "B", 22)
    pdf.ln(30)
    pdf.multi_cell(0, 10, TITLE, align="C")
    pdf.set_font("Helvetica", "", 14)
    pdf.ln(4)
    pdf.multi_cell(0, 8, SUBTITLE, align="C")
    pdf.ln(10)
    pdf.set_font("Helvetica", "I", 10)
    pdf.multi_cell(0, 6,
        "This document is written so that anyone -- including someone with no prior "
        "context on this project -- can read it and start contributing accurate, "
        "well-sourced golden QA candidates.", align="C")

    for heading, level, blocks in SECTIONS:
        pdf.add_page()
        if level == 1:
            pdf.set_font("Helvetica", "B", 16)
        else:
            pdf.set_font("Helvetica", "B", 13)
        pdf.set_text_color(20, 20, 20)
        pdf.set_x(pdf.l_margin)
        pdf.multi_cell(0, 9, heading)
        pdf.ln(2)
        pdf.set_text_color(0, 0, 0)

        for kind, payload in blocks:
            if kind == "p":
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Helvetica", "", 11)
                pdf.multi_cell(0, 6, payload)
                pdf.ln(2)
            elif kind == "b":
                pdf.set_font("Helvetica", "", 11)
                for item in payload:
                    pdf.set_x(pdf.l_margin + 4)
                    pdf.multi_cell(pdf.w - pdf.l_margin - pdf.r_margin - 4, 6, f"-  {item}")
                pdf.ln(2)
            elif kind == "table":
                header, *rows = payload
                col0_w = 38
                col1_w = pdf.w - pdf.l_margin - pdf.r_margin - col0_w
                pdf.set_font("Helvetica", "B", 10)
                pdf.set_xy(pdf.l_margin, pdf.get_y())
                y_start = pdf.get_y()
                pdf.multi_cell(col0_w, 6, header[0], border=1)
                pdf.set_xy(pdf.l_margin + col0_w, y_start)
                pdf.multi_cell(col1_w, 6, header[1], border=1)
                pdf.set_font("Helvetica", "", 10)
                for row in rows:
                    y0 = max(pdf.get_y(), y_start)
                    pdf.set_xy(pdf.l_margin, y0)
                    pdf.multi_cell(col0_w, 6, row[0], border=1)
                    y1 = pdf.get_y()
                    pdf.set_xy(pdf.l_margin + col0_w, y0)
                    pdf.multi_cell(col1_w, 6, row[1], border=1)
                    y2 = pdf.get_y()
                    # equalize row height between the two cells
                    tall = max(y1, y2)
                    pdf.set_xy(pdf.l_margin, tall)
                pdf.ln(3)
            elif kind == "example":
                d = payload
                pdf.set_fill_color(242, 242, 242)
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Helvetica", "B", 10.5)
                pdf.multi_cell(0, 6, f"Q: {d['q']}", fill=True)
                pdf.set_x(pdf.l_margin)
                pdf.set_font("Helvetica", "", 10.5)
                pdf.multi_cell(0, 6, f"A: {d['a']}", fill=True)
                if d.get("source"):
                    pdf.set_x(pdf.l_margin)
                    pdf.set_font("Helvetica", "I", 9.5)
                    pdf.multi_cell(0, 5.5, f"Source: {d['source']}", fill=True)
                if d.get("quote"):
                    pdf.set_x(pdf.l_margin)
                    pdf.set_font("Helvetica", "I", 9.5)
                    pdf.multi_cell(0, 5.5, f"Quoted text backing this: \"{d['quote']}\"",
                                  fill=True)
                pdf.ln(3)


def main() -> int:
    pdf = FPDF(format="A4")
    pdf.set_title(TITLE)
    render(pdf)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(OUT))
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
