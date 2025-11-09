import requests
import urllib.parse
import re

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; ADSRetrievalBot/1.0; +https://example.com/bot)"
}


# ============================================================
# 1. GENERIC KEYWORD EXTRACTOR
# ============================================================
def extract_keywords(query: str) -> str:
    """
    Generic keyword extractor – no hardcoding.
    Removes stopwords, keeps nouns/meaningful words.
    Works for ANY domain.
    """

    stopwords = {
        "what", "which", "who", "where", "why", "when",
        "is", "are", "was", "were", "the", "a", "an", "of",
        "in", "to", "for", "by", "on", "and", "or", "does",
        "do", "did", "city", "considered"
    }

    tokens = re.findall(r"[A-Za-z]+", query.lower())

    filtered = [w for w in tokens if w not in stopwords and len(w) > 2]

    if not filtered:
        return query

    return " ".join(filtered)


# ============================================================
# 2. CIRRUSSEARCH (GENERIC)
# ============================================================
def wikipedia_search(query: str, limit: int = 10):
    url = "https://en.wikipedia.org/w/api.php"

    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "srlimit": limit,
        "format": "json"
    }

    resp = requests.get(url, params=params, headers=HEADERS)
    try:
        data = resp.json()
        return data["query"]["search"]
    except:
        print("Wikipedia returned HTML:\n", resp.text[:500])
        return []


# ============================================================
# 3. GENERIC MULTI-STRATEGY SEARCH
# ============================================================
def smart_wikipedia_search(query):
    """
    THREE STRATEGIES (generic, no hardcoding):
    1) Full natural query
    2) Keyword-only query
    3) Broad OR-expanded query
    """

    strategies = []

    # Strategy 1 — natural query
    strategies.append(query)

    # Strategy 2 — keyword only
    kw = extract_keywords(query)
    strategies.append(kw)

    # Strategy 3 — expansion (generic)
    parts = kw.split()
    if len(parts) > 1:
        expanded = " OR ".join(parts)
        strategies.append(expanded)

    print("\n[Search] Query strategies:")
    for s in strategies:
        print(" -", s)

    # Try all strategies
    for s in strategies:
        results = wikipedia_search(s)
        if results:
            return results

    return []


# ============================================================
# 4. SELECT BEST PAGE USING SNIPPET MATCHING
# ============================================================
def rank_results_by_relevance(query: str, results):
    """
    **FINAL GENERIC RANKER:**
    1. Extract entity words from query (nouns/proper nouns)
    2. Strong boost for titles containing entity
    3. Strong penalty for titles missing entity
    4. Snippet relevance secondary
    """

    query_lower = query.lower()

    # 1. Extract entity tokens (generic)
    tokens = re.findall(r"[A-Za-z]+", query_lower)
    common_stop = {"what","which","who","where","when","why","is","are","was","were","the","of","in","to","for","does","do","did","capital"}
    entities = [t for t in tokens if t not in common_stop and len(t) > 2]

    # For "What is the capital of France?" => ["france"]
    # For "Kashmir of Karnataka" => ["kashmir","karnataka"]

    if not entities:
        entities = tokens  # fallback

    print("[Entities Extracted]:", entities)

    ranked = []

    for res in results:
        title = res["title"].lower()
        snippet = re.sub("<.*?>", "", res["snippet"].lower())

        score = 0

        # ✅ 1. Title must contain entity (STRONG BOOST)
        for ent in entities:
            if ent in title:
                score += 100   # Huge boost → ensures France wins

        # ✅ 2. Penalize if entity missing from title entirely
        if not any(ent in title for ent in entities):
            score -= 50

        # ✅ 3. Snippet match = small boost
        common_words = set(entities) & set(snippet.split())
        score += len(common_words)

        # ✅ 4. Penalize question-example pages
        if "question" in title:
            score -= 200

        ranked.append((score, res))

    ranked.sort(key=lambda x: x[0], reverse=True)
    return ranked




# ============================================================
# 5. FETCH CONTENT (REST + FALLBACK)
# ============================================================
def fetch_page_rest(title: str):
    safe = urllib.parse.quote(title.replace(" ", "_"))
    # url = f"https://en.wikipedia.org/api/rest_v1/page/mobile-sections/{safe}"
    url = f"https://en.wikipedia.org/api/rest_v1/page/html/{title}"


    resp = requests.get(url, headers=HEADERS)
    if resp.status_code != 200:
        return None

    try:
        data = resp.json()
    except:
        return None

    text_blocks = []
    for sec in data.get("lead", {}).get("sections", []):
        if "text" in sec:
            text_blocks.append(sec["text"])
    for sec in data.get("remaining", []):
        if "text" in sec:
            text_blocks.append(sec["text"])

    return "\n\n".join(text_blocks)


def fetch_page_fallback(title: str):
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "prop": "extracts",
        "explaintext": True,
        "titles": title,
        "format": "json"
    }
    resp = requests.get(url, params=params, headers=HEADERS)

    try:
        data = resp.json()
    except:
        return ""

    page = list(data["query"]["pages"].values())[0]
    return page.get("extract", "")


# ============================================================
# 6. MASTER FUNCTION
# ============================================================
def get_best_wikipedia_page(query):
    print("\n=== WIKIPEDIA RETRIEVAL ENGINE ===\n")
    results = smart_wikipedia_search(query)

    if not results:
        print("\n[ERROR] No Wikipedia results found.")
        return {"title": None, "content": ""}

    ranked = rank_results_by_relevance(query, results)

    best_title = ranked[0][1]["title"]

    print("\n[Best Title Selected]:", best_title)

    # Try REST API
    content = fetch_page_rest(best_title)
    if not content:
        print("[REST Failed] Trying fallback...")
        content = fetch_page_fallback(best_title)

    return {"title": best_title, "content": content}


# ============================================================
# TEST
# ============================================================
if __name__ == "__main__":
    q = input("Enter your query: ")
    result = get_best_wikipedia_page(q)

    print("\n============================")
    print("TITLE:", result["title"])
    print("============================\n")

    print(result["content"][:2000])
    print("\n... (truncated) ...")
