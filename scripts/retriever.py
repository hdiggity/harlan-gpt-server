# retriever.py
# whisky-aware bm25 retrieval. own domain logic + classic algorithms only,
# no external models. improves on vanilla bm25 with: coherent chunking,
# stemming + whisky canonicalization + stopword removal, whisky synonym
# query expansion, and tuned bm25 parameters.

import glob
import json
import os
import re

# porter stemmer: a deterministic algorithm (not a trained model), so it
# stays fully "own". collapses casks/cask, matured/maturing, etc.
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

STEMMER = PorterStemmer()

# common english stopwords to drop so scoring focuses on meaningful terms
STOPWORDS = set(
    [
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "for",
        "from",
        "has",
        "have",
        "in",
        "into",
        "is",
        "it",
        "its",
        "of",
        "on",
        "or",
        "that",
        "the",
        "to",
        "was",
        "were",
        "will",
        "with",
        "this",
        "these",
        "those",
        "they",
        "their",
        "there",
        "here",
        "then",
        "than",
    ]
)

# whisky synonym map: query expansion using OWN domain knowledge. a query
# term that hits a key also searches its group, closing bm25's paraphrase
# gap without any neural model. extend these groups freely.
WHISKY_SYNONYMS = {
    "peaty": ["smoky", "phenolic", "peat", "medicinal", "iodine"],
    "smoky": ["peaty", "phenolic", "peat", "bonfire", "ash"],
    "sherry": ["oloroso", "px", "pedro", "ximenez", "fortified", "sherried"],
    "oloroso": ["sherry", "px", "sherried", "nutty"],
    "bourbon": ["barrel", "vanilla", "oak", "american"],
    "islay": ["peaty", "smoky", "maritime", "coastal", "brine"],
    "speyside": ["fruity", "elegant", "honeyed", "floral"],
    "sweet": ["honey", "vanilla", "caramel", "toffee", "sugar"],
    "spicy": ["pepper", "cinnamon", "clove", "ginger", "rye"],
    "fruity": ["apple", "pear", "citrus", "orchard", "berry"],
    "cask": ["barrel", "wood", "maturation", "aged", "matured"],
    "finish": ["aftertaste", "length", "lingering"],
    "nose": ["aroma", "bouquet", "smell"],
    "palate": ["taste", "mouthfeel", "flavour", "flavor"],
}

# bm25 parameters tuned for prose passages (defaults are k1=1.5, b=0.75).
# slightly lower b reduces over-penalizing longer, information-rich passages.
BM25_K1 = 1.2
BM25_B = 0.6

# chunking: pack coherent paragraphs into passages of this size
MAX_PASSAGE_CHARS = 1200
MIN_PASSAGE_CHARS = 120


def canonicalize(text):
    # normalize whisky notation so variants match (mirrors corpus cleaning).
    # age statements
    text = re.sub(
        r"\b(\d{1,2})\s*[- ]?\s*(?:yo|y\.o\.|year[- ]old|years? old)\b",
        r"\1 year old",
        text,
        flags=re.IGNORECASE,
    )
    # abv / percentage
    text = re.sub(
        r"\b(\d{1,2}(?:\.\d)?)\s*%\s*(?:abv)?\b",
        r"\1% abv",
        text,
        flags=re.IGNORECASE,
    )
    return text


def tokenize(text):
    # whisky-aware tokenization: canonicalize, lowercase word tokens,
    # drop stopwords, then stem. all deterministic.
    text = canonicalize(text.lower())
    words = re.findall(r"[a-z0-9]+", text)
    out = []
    for w in words:
        if w in STOPWORDS:
            continue
        out.append(STEMMER.stem(w))
    return out


def expand_query(q_tokens):
    # add whisky synonyms for any query token that matches a synonym key.
    # keys/values are stemmed to line up with tokenized passages.
    expanded = list(q_tokens)
    # build a stemmed synonym lookup once
    for key, syns in WHISKY_SYNONYMS.items():
        key_stem = STEMMER.stem(key)
        if key_stem in q_tokens:
            for s in syns:
                expanded.append(STEMMER.stem(s))
    return expanded


def chunk_corpus(text):
    # pack paragraphs into coherent passages up to MAX_PASSAGE_CHARS
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    passages = []
    buf = ""
    for p in paras:
        if len(buf) + len(p) + 1 <= MAX_PASSAGE_CHARS:
            buf = (buf + "\n" + p).strip()
        else:
            if len(buf) >= MIN_PASSAGE_CHARS:
                passages.append(buf)
            if len(p) > MAX_PASSAGE_CHARS:
                for i in range(0, len(p), MAX_PASSAGE_CHARS):
                    passages.append(p[i : i + MAX_PASSAGE_CHARS])
                buf = ""
            else:
                buf = p
    if len(buf) >= MIN_PASSAGE_CHARS:
        passages.append(buf)
    return passages


class WhiskyRetriever:
    # builds the bm25 index over passages read from the retrieval jsonl.
    def __init__(self, retrieval_dir):
        self.passages = self._load(retrieval_dir)
        tokenized = [tokenize(p) for p in self.passages]
        self.bm25 = BM25Okapi(tokenized, k1=BM25_K1, b=BM25_B)

    def _load(self, retrieval_dir):
        passages = []
        files = sorted(glob.glob(os.path.join(retrieval_dir, "*.jsonl")))
        if not files:
            raise SystemExit(f"no .jsonl in {retrieval_dir}")
        for path in files:
            with open(path, errors="ignore") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    txt = obj.get("text")
                    if txt:
                        passages.append(txt)
        if not passages:
            raise SystemExit("no passages with a text field")
        return passages

    def retrieve(self, question, top_k=5):
        q_tokens = tokenize(question)
        q_tokens = expand_query(q_tokens)
        scores = self.bm25.get_scores(q_tokens)
        top = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [self.passages[i] for i in top]
