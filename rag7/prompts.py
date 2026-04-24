"""System prompts for rag7 LLM chains.

Constants for static prompts, callables for parameterized ones.
Grouped by stage: auto-config, preprocess, retrieval, grading, generation.
"""

from __future__ import annotations


_ADDITIONAL_RULES_HEADER = "\n\nADDITIONAL RULES FOR THIS COLLECTION:\n"


def _append_custom(prompt: str, custom_instructions: str | None) -> str:
    """Append a per-collection custom-instructions block to a system prompt.

    No-op when the block is empty / None. Uses a clear delimiter so the LLM
    can distinguish the built-in rules from admin-supplied overrides.
    """
    if not custom_instructions:
        return prompt
    text = custom_instructions.strip()
    if not text:
        return prompt
    return f"{prompt}{_ADDITIONAL_RULES_HEADER}{text}"


# ---- preprocess / query rewriting ----


def hyde_system() -> str:
    return (
        "Write 2-4 sentences from a relevant document. "
        "Use domain terminology, specific identifiers, citations, "
        "or technical terms likely present in the source. "
        "No questions — write as extracted text."
    )


def preprocess_system() -> str:
    return (
        "Extract 2-5 concise search keywords from the user's question. "
        "Return ONLY essential nouns, entities, and technical terms. "
        "Omit question words, common verbs, and stop words (e.g. find, "
        "search, what, how, give me). "
        "Correct obvious spelling mistakes in product names, brand names, "
        "and German/Italian words to their standard form. "
        "Also restore German umlauts when users omit them: 'oe'→'ö', "
        "'ae'→'ä', 'ue'→'ü', or when a bare vowel clearly substitutes an "
        "umlaut (e.g. 'bieroffner'→'Bieröffner', "
        "'Klosetsitz'→'Klosettsitz'). "
        "Preserve alphanumeric codes and technical identifiers exactly "
        "(e.g. DHR171RTJ, M12, SDS-Plus, 18V). "
        "Also set 'variants': 2-3 alternative keyword phrasings — "
        "include SYNONYMS and related terms for the concept, not just "
        "narrower ones. "
        "For example 'Bieröffner' → ['Flaschenöffner', 'Kapselheber']; "
        "'Hammerstiel' → ['Hammergriff', 'Werkzeugstiel']; "
        "'Zange' → ['Kombizange', 'Greifzange']. "
        "SKIP variants when the query is purely an identifier/code/SKU — "
        "synonyms don't exist for '4457227' or 'DHR171RTJ', leave "
        "variants empty. "
        "Variants must use correctly-spelled standard forms; never echo "
        "misspelled words from the input. "
        "Keep variants in the same language as the query. "
        "Also set 'semantic_ratio' (0.0-1.0): how much to weight "
        "semantic/vector search vs keyword/BM25. "
        "- Exact identifiers, codes, names → 0.1-0.2 (BM25 excels) "
        "- Specific nouns, mixed queries → 0.3-0.5 (balanced) "
        "- Intent/need-based queries → 0.6-0.8 (semantic helps) "
        "- Abstract/conceptual questions → 0.7-0.9 (semantic dominant) "
        "Also set 'fusion' ('rrf' or 'dbsf'): how to fuse multi-arm "
        "search results. "
        "- 'rrf': Reciprocal Rank Fusion — rank-based, best for "
        "keyword-heavy queries. "
        "- 'dbsf': Distribution-Based Score Fusion — score-normalised, "
        "best for semantic queries. "
        "Also set 'alternative_to' (string or null): if the user asks "
        "for alternatives, replacements, substitutes, or competitors "
        "for a SPECIFIC named product/item, set this to the product "
        "name/identifier. "
        "Trigger words: 'alternative', 'ersatz', 'replacement', "
        "'substitut', 'competitor', 'statt', 'anstelle', 'remplacer', "
        "'remplaçant', 'à la place de'. "
        "The query should then contain the CATEGORY/TYPE terms (what "
        "kind of product it is), NOT the specific product name."
    )


CONTEXTUALIZE = (
    "Rewrite the user's follow-up question as a standalone search "
    "query, using prior conversation for context. "
    "Preserve exact codes, numbers, product names. If the question is "
    "already standalone, return it unchanged."
)


def multi_query_swarm(n: int) -> str:
    return (
        f"Generate {n} distinct search query variants for the question. "
        "Each should use different keywords, synonyms, or angles to "
        "maximise recall. "
        f"Return JSON with a 'queries' list of {n} strings."
    )


# ---- retrieval-time grading ----

_CLOSE_MATCH_HEADER = (
    "You filter reranker output for semantic relevance to the user's "
    "query.\n"
    "\n"
    "Protocol (use the `reasoning` field FIRST, before deciding `keep`):\n"
    "1. State the user's intent in one phrase.\n"
    "2. For EACH doc, decide: is this document relevant to the stated "
    "intent?\n"
    "3. Output `keep` as 1-based indices of relevant docs.\n"
    "\n"
)

_CLOSE_MATCH_LOOSE_TAIL = (
    "Bias toward RECALL. When in doubt, keep. A doc in the same product "
    "category as the query is relevant even if brand, size, variant, or "
    "model differs — the user can narrow later. Drop only when the "
    "category is clearly wrong (e.g. query asks for a drill, doc is a "
    "t-shirt). Lexical similarity is not uncertainty."
)

_CLOSE_MATCH_BALANCED_TAIL = (
    "Keep docs that match the user's intent. Borderline items (same "
    "category, different brand or variant) should be kept unless a "
    "clearer match exists. Drop items that are off-topic or only "
    "lexically similar. Fail-open when genuinely uncertain."
)

_CLOSE_MATCH_STRICT_TAIL = (
    "Keep only docs that clearly match the user's stated intent — "
    "matching category AND matching key qualifiers the user named "
    "(brand, type, size). Drop category-only matches when the user "
    "named a specific brand or variant. Never drop all docs: if "
    "nothing clearly matches, keep the single best candidate."
)

_CLOSE_MATCH_TAILS: dict[str, str] = {
    "loose": _CLOSE_MATCH_LOOSE_TAIL,
    "balanced": _CLOSE_MATCH_BALANCED_TAIL,
    "strict": _CLOSE_MATCH_STRICT_TAIL,
}

_CLOSE_MATCH_BASE = _CLOSE_MATCH_HEADER + _CLOSE_MATCH_LOOSE_TAIL


def close_match(
    custom_instructions: str | None = "",
    strictness: str = "loose",
) -> str:
    tail = _CLOSE_MATCH_TAILS.get(strictness, _CLOSE_MATCH_LOOSE_TAIL)
    return _append_custom(_CLOSE_MATCH_HEADER + tail, custom_instructions)


# Backwards-compatible alias — existing callers that import CLOSE_MATCH
# as a constant continue to work (no custom instructions appended).
CLOSE_MATCH = _CLOSE_MATCH_BASE


def reasoning_verdict(intent_hint: str = "") -> str:
    return (
        "You are a retrieval result judge. Given a user question and "
        "the top retrieved documents, decide all of:\n"
        "1. `sufficient`: true if at least one document directly "
        "addresses the question. False if off-topic, too vague, or "
        "clearly wrong. For 'alternative to X' or 'not X' queries, "
        "documents that are DIFFERENT from X are correct — rate "
        "sufficient=true if they are in the same product category.\n"
        "2. `dominated_by`: 1-based indices of documents that are "
        "clearly IRRELEVANT or VIOLATE the query intent (e.g., user "
        "asked for alternatives to product X but results contain X "
        "itself). Empty list if all are fine.\n"
        "3. `rewritten_query`: if ALL documents are poor, suggest a "
        "better search query. Otherwise null.\n"
        "4. `reasoning`: brief chain-of-thought explaining the "
        "verdict.\n\n"
        "Be strict on `dominated_by` — only flag docs that clearly "
        "miss the intent."
        f"{intent_hint}"
    )


QUALITY_GATE = (
    "You are a retrieval quality judge. "
    "Given a question and the top-5 retrieved document snippets, "
    "decide if the documents are sufficient to answer the question. "
    "Return sufficient=true if at least one document directly addresses "
    "the question. "
    "Return sufficient=false if the documents are off-topic, too vague, "
    "or clearly wrong. "
    "For 'alternative to X' or 'not X' queries, documents that are "
    "DIFFERENT from X are correct — they should be rated sufficient if "
    "they are in the same product category."
)


FINAL_GRADE = (
    "You are a strict quality grader for a product search assistant. "
    "Given the user question, retrieved document snippets, and "
    "generated answer, decide if the answer correctly identifies the "
    "product(s) the user was looking for. "
    "sufficient=true only if the answer names or describes the right "
    "product. "
    "If NOT sufficient, write a specific, actionable suggestion — "
    "e.g. 'Search for Rothenberger Rohrzange, the brand was misspelled' "
    "or 'User wants a short hammer handle (Hammerstiel), not a complete "
    "hammer'. "
    "Keep suggestion under 30 words. confidence: 1.0=definitely "
    "correct, 0.0=clearly wrong."
)


GRADE_DOCS = (
    "You are a strict quality grader for a retrieval system. "
    "Given the user question and retrieved document snippets, "
    "decide if the documents contain a relevant answer or item. "
    "sufficient=true only if the retrieved content directly addresses "
    "the question. "
    "If NOT sufficient, write a specific, actionable search suggestion "
    "to improve recall — e.g. correct a misspelling, suggest a "
    "synonym, or clarify the intent. "
    "Keep suggestion under 30 words. confidence: 1.0=definitely "
    "relevant, 0.0=clearly wrong."
)


RELEVANCE_CHECK = (
    "Strictly judge if snippets DIRECTLY answer the question. "
    "Set makes_sense=true and confidence>=0.9 only if a snippet "
    "clearly contains the answer. Otherwise lower confidence."
)


# ---- intent / routing ----

PRODUCT_CODE = (
    "Detect if the user query is asking to look up a product by a "
    "specific numeric code such as an EAN, GTIN, barcode, article "
    "number, or internal ID.\n"
    "Examples that ARE product-code queries:\n"
    "  'EAN 9002886001325' → code='9002886001325'\n"
    "  'Artikel-Nr. 12345' → code='12345'\n"
    "  'numéro d'article 4011905123' → code='4011905123'\n"
    "  'codice articolo 7612345678901' → code='7612345678901'\n"
    "  'article number 0600753042' → code='0600753042'\n"
    "Examples that are NOT product-code queries:\n"
    "  'Bosch Winkelschleifer 125mm' → is_product_code=false\n"
    "  'ich brauche einen Hammer' → is_product_code=false\n"
    "Return only the numeric digits of the code (strip any prefix words)."
)


def filter_intent(
    filterable: list[str],
    values_block: str,
    custom_instructions: str | None = "",
) -> str:
    body = (
        f"You help build filter expressions for a search backend.\n"
        f"Filterable fields in this index: {filterable}\n\n"
        f"Sample values per field:\n{values_block}\n\n"
        "TASK: Detect if the question mentions a SPECIFIC named entity "
        "(brand, supplier, company, category, boolean flag, etc.) that "
        "should be used as a filter.\n\n"
        "RULES:\n"
        "- If a proper noun / brand name appears that likely maps to a "
        "name field (e.g. supplier_name, brand_name, company), use "
        "CONTAINS with that name — even if the exact value is not in "
        "the samples above.\n"
        "- IMPORTANT: In German 'von' means 'by/from'. The word AFTER "
        "'von' is the brand/supplier. Exception: if the word after "
        "'von' is clearly a product type (like Schaumpistole, "
        "Bieröffner, Werkzeug, Maschine), then the brand is the proper "
        "noun BEFORE 'von'.\n"
        "  Example: 'Schaumpistole von ProOne' → brand=ProOne (after "
        "'von'), correct.\n"
        "  Example: 'ProOne von Schaumpistole' → 'Schaumpistole' is a "
        "product type, so the brand is 'ProOne' (before 'von').\n"
        "- For boolean fields (is_own_brand, active, etc.), use = with "
        "true/false.\n"
        "- For exact IDs or codes, use =.\n"
        "- For multiple values, use IN.\n"
        "- EXCLUSION: If the question says 'nicht'/'not'/'aber nicht'/"
        "'sans'/'pas de'/'exclude'/'except', use NOT_CONTAINS to "
        "EXCLUDE that entity.\n"
        "  For multi-brand exclusion ('nicht X oder Y'), put the first "
        "brand in value and additional brands in extra_excludes.\n"
        "  Example: 'Mörtel nicht Weber' → field=supplier_name, "
        "value=Weber, operator=NOT_CONTAINS\n"
        "  Example: 'nicht Sakret oder Fixit' → value=Sakret, "
        "extra_excludes=[Fixit]\n"
        "- 'Alternative zu X' means 'find similar products but not X "
        "itself'. Use NOT_CONTAINS on the name field to exclude the "
        "queried product.\n"
        "  Example: 'Alternative zu PCI Polyfix' → field=article_name, "
        "value=PCI Polyfix, operator=NOT_CONTAINS\n"
        "- If NO specific entity is named (broad/generic query), "
        "return field=null, value='', operator=''.\n"
        "Operators: CONTAINS (partial match), = (exact), IN (multiple "
        "values), NOT_CONTAINS (exclude partial match), != (exclude "
        "exact)"
    )
    return _append_custom(body, custom_instructions)


def collection_select(descriptions: str) -> str:
    return (
        "Pick the collections most likely to contain the answer.\n"
        f"Available collections:\n{descriptions}\n\n"
        "Return JSON with a 'collections' list containing ONLY names "
        "from the list above.\n"
        "- Prefer the minimum set that covers the question.\n"
        "- If the question spans multiple topics, include all relevant.\n"
        "- If unsure, include all candidates (better recall than miss).\n"
        "- Never invent names not in the list."
    )


# ---- generation ----

ANSWER_SYSTEM = (
    "Answer using only the provided context. "
    "Cite sources inline using [n] numbers that match the context blocks. "
    "Say so if the context is insufficient."
)


# ---- rewrite-on-failure ----


def rewrite_query(
    previous_query: str,
    feedback: str | None = None,
    top_snippet: str | None = None,
) -> str:
    """Unified rewrite prompt. Three cases driven by which slots are set:

    - feedback set → grader said the answer was wrong, follow the feedback
    - top_snippet set → docs returned but not relevant, need different angle
    - neither → no results, broaden or try synonyms
    """
    parts: list[str] = []
    if feedback:
        parts.append(
            "The previous search did not return the right products. "
            f'Grader feedback: "{feedback}".'
        )
    elif top_snippet:
        parts.append(
            "The search returned documents but they are NOT relevant to "
            "the question. "
            f'Top result snippet: "{top_snippet}...".'
        )
    else:
        parts.append("The previous search returned NO results.")
    parts.append(f'Previous query: "{previous_query}".')
    if feedback:
        parts.append(
            "Rewrite the query following the feedback. "
            "Return a precise query of 1-4 keywords AND 2-3 spelling/form "
            "variants (e.g. with/without umlaut, compound vs. spaced). "
            "No filler words."
        )
    elif top_snippet:
        parts.append(
            "Rewrite using different keywords, synonyms, or a narrower "
            "angle. Return only 1-4 keywords — no filler words."
        )
    else:
        parts.append(
            "Rewrite using different words, synonyms, or a broader term. "
            "Return only 1-3 short keywords — no filler."
        )
    return " ".join(parts)
