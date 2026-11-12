"""
semantic_search.py - find campus records by MEANING instead of by shared words.

Why: MongoDB $text only matches words. "Where can I pick up my transcript?" shares
no useful word with a record that says "Manages ... transcripts ...", and $text has
no IDF weighting, so common words ("can", "students") crowd out rare ones.
Embeddings turn text into vectors so similar MEANING ends up close together.

How it works (small data, so no vector database needed):
  * each record is turned into a short text ("Office of the Registrar (University
    Service). Located in Ullsvik Hall, room 2100. Manages ... transcripts ...")
  * that text is embedded once and the vector is CACHED in the Mongo collection
    `rag_embeddings` (keyed by a hash of the text, so edits re-embed automatically)
  * the vectors live in memory; a question is embedded (1 API call) and compared with
    every record using a dot product (~137 records -> a few milliseconds)

Failure handling: embeddings are an enhancement, never a dependency. Any embedding
problem (quota, 429, outage, wrong model name) raises EmbeddingError and the caller
falls back to plain keyword search. After a failure we back off for a minute so a
broken quota doesn't add a failing API call to every question.

Environment variables (all optional):
  EMBEDDING_MODEL            default: gemini-embedding-001 on Gemini, else text-embedding-3-small
  EMBEDDING_DIMENSIONS       default 768 (a size Google recommends; also valid for OpenAI v3 models)
  SEMANTIC_BAND              default 0.12  keep hits within this similarity of the best hit
  SEMANTIC_MIN_SIMILARITY    default 0.0   absolute floor (tune with debug_rag.py output)
"""

import hashlib
import logging
import math
import os
import threading
import time
from collections import OrderedDict
from operator import mul
from pymongo import ReplaceOne

logger = logging.getLogger(__name__)

CACHE_COLLECTION = "rag_embeddings"
EMBED_BATCH_SIZE = 50            # texts per API call
REFRESH_SECONDS = 600            # re-check the database for edited records this often
RETRY_AFTER_FAILURE_SECONDS = 60
QUERY_CACHE_SIZE = 256           # remember recent question vectors (suggestion chips repeat a lot)

# fields needed to describe a record; keeps the scan light
_PROJECTION = {"name": 1, "category": 1, "description": 1, "demo_description": 1,
               "buildingId": 1, "room": 1, "lotId": 1}

class EmbeddingError(RuntimeError):
    """Embeddings are unavailable right now; callers should fall back to keyword search."""


# --------------------------------------------------------------------------
# settings
# --------------------------------------------------------------------------
def embedding_model() -> str:
    explicit = os.getenv("EMBEDDING_MODEL")
    if explicit:
        return explicit
    if "generativelanguage.googleapis.com" in (os.getenv("OPENAI_BASE_URL") or ""):
        return "gemini-embedding-001"
    return "text-embedding-3-small"

def embedding_dimensions() -> int:
    return int(os.getenv("EMBEDDING_DIMENSIONS", "768"))

def semantic_band() -> float:
    return float(os.getenv("SEMANTIC_BAND", "0.12"))

def min_similarity() -> float:
    return float(os.getenv("SEMANTIC_MIN_SIMILARITY", "0.0"))


# --------------------------------------------------------------------------
# embedding calls
# --------------------------------------------------------------------------
def _unit(vector: list[float]) -> list[float]:
    """Scale to length 1 so a dot product equals cosine similarity.
    (Truncated Gemini/OpenAI vectors are not guaranteed to be normalised.)"""
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]


def embed_batch(client, texts: list[str]) -> list[list[float]]:
    """Embed up to EMBED_BATCH_SIZE texts with ONE API call. Returns unit vectors."""
    from openai import OpenAIError  # imported lazily, like rag_service does

    try:
        response = client.embeddings.create(
            model=embedding_model(),
            input=texts,
            dimensions=embedding_dimensions(),
        )
    except OpenAIError as exc:
        raise EmbeddingError(
            f"{type(exc).__name__} (status={getattr(exc, 'status_code', None)}) "
            f"model={embedding_model()}: {exc}"
        ) from exc

    data = list(response.data)
    if len(data) != len(texts):
        raise EmbeddingError(f"asked for {len(texts)} embeddings, got {len(data)}")
    indices = [item.index for item in data]
    if all(index is None for index in indices):
        # Some OpenAI-compatible providers omit indices; their response order
        # corresponds to the input order.
        pass
    elif all(index is None or type(index) is int for index in indices):
        # Accept partially populated indices only when they confirm the
        # response is already in input order.
        if any(index != position for position, index in enumerate(indices) if index is not None):
            raise EmbeddingError("embedding response has missing indices and is not in input order")
    elif any(type(index) is not int for index in indices):
        raise EmbeddingError("embedding response has missing or invalid item indices")
    else:
        data.sort(key=lambda item: item.index)
        if [item.index for item in data] != list(range(len(texts))):
            raise EmbeddingError("embedding response indices are duplicated or out of range")
    return [_unit(item.embedding) for item in data]


# --------------------------------------------------------------------------
# what gets embedded
# --------------------------------------------------------------------------
def record_text(type_: str, doc: dict, building_names: dict) -> str:
    """One short natural-language description per record. Including the building makes
    questions like 'what is in Ullsvik Hall?' work too."""
    parts = [f"{doc['name']} ({doc.get('category') or type_})"]
    if type_ in ("department", "service") and doc.get("buildingId") is not None:
        place = building_names.get(doc["buildingId"])
        if place:
            parts.append(f"Located in {place}" + (f", room {doc['room']}" if doc.get("room") else ""))
    description = doc.get("description") or doc.get("demo_description")
    if description:
        parts.append(description)
    return ". ".join(parts)


def _text_hash(text: str) -> str:
    # model + dimensions are part of the hash, so changing either re-embeds everything
    return hashlib.sha256(f"{embedding_model()}|{embedding_dimensions()}|{text}".encode()).hexdigest()[:16]


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------
class SemanticIndex:
    def __init__(self, types: dict):
        # types: {"service": ("university_services", "_id"), ...} (same mapping rag_service uses)
        self._types = types
        self._lock = threading.Lock()
        self._query_lock = threading.Lock()
        self._vectors: dict[str, list[tuple[object, list[float]]]] = {}
        self._built_at = 0.0
        self._blocked_until = 0.0
        self._query_cache: OrderedDict[str, list[float]] = OrderedDict()
        self.stats: dict = {}

    @property
    def ready(self) -> bool:
        return bool(self._vectors)

    # ---- keeping the index current ----
    def ensure_fresh(self, db, client, force: bool = False) -> None:
        """Load/refresh vectors if needed. Cheap when nothing changed (no API calls)."""
        if not force and self.ready and time.monotonic() - self._built_at < REFRESH_SECONDS:
            return
        with self._lock:
            now = time.monotonic()
            if not force and self.ready and now - self._built_at < REFRESH_SECONDS:
                return                      # another request refreshed it while we waited
            if not force and now < self._blocked_until:
                if self.ready:
                    return                  # keep serving the vectors we have
                raise EmbeddingError("embedding service failed recently; using keyword search for now")
            try:
                self._rebuild(db, client)
            except EmbeddingError as exc:
                self._blocked_until = time.monotonic() + RETRY_AFTER_FAILURE_SECONDS
                if self.ready:
                    logger.warning("Semantic index refresh failed, keeping old vectors: %s", exc)
                    return
                raise
            self._built_at = time.monotonic()
            self._blocked_until = 0.0

    def _rebuild(self, db, client) -> None:
        names = {d["buildingId"]: d["name"] for d in db["buildings"].find({}, {"buildingId": 1, "name": 1})}

        records = []  # (type, ref, cache key, text, hash)
        for type_, (coll, id_field) in self._types.items():
            for doc in db[coll].find({}, _PROJECTION):
                ref = doc.get(id_field)
                text = record_text(type_, doc, names)
                records.append((type_, ref, f"{type_}:{ref}", text, _text_hash(text)))

        cache = {d["_id"]: d for d in db[CACHE_COLLECTION].find({}, {"hash": 1, "vector": 1})}
        missing = [r for r in records if cache.get(r[2], {}).get("hash") != r[4]]

        fresh: dict[str, list[float]] = {}
        for start in range(0, len(missing), EMBED_BATCH_SIZE):
            batch = missing[start:start + EMBED_BATCH_SIZE]
            vectors = embed_batch(client, [r[3] for r in batch])   # may raise EmbeddingError
            db[CACHE_COLLECTION].bulk_write(
                [ReplaceOne({"_id": r[2]},
                            {"_id": r[2], "type": r[0], "ref": r[1], "hash": r[4], "vector": vec},
                            upsert=True)
                 for r, vec in zip(batch, vectors)],
                ordered=False,
            )  # saved per batch, so progress survives a quota error half-way through
            fresh.update({r[2]: vec for r, vec in zip(batch, vectors)})

        db[CACHE_COLLECTION].delete_many({"_id": {"$nin": [r[2] for r in records]}})

        built: dict[str, list] = {t: [] for t in self._types}
        for type_, ref, key, _text, _hash in records:
            built[type_].append((ref, fresh[key] if key in fresh else cache[key]["vector"]))
        self._vectors = built
        self.stats = {"model": embedding_model(), "dimensions": embedding_dimensions(),
                      "records": len(records), "embedded_now": len(missing),
                      "from_cache": len(records) - len(missing)}
        logger.info("Semantic index ready: %s", self.stats)

    # ---- querying ----
    def embed_query(self, client, text: str) -> list[float]:
        key = " ".join(text.lower().split())
        with self._query_lock:
            if key in self._query_cache:
                self._query_cache.move_to_end(key)
                return self._query_cache[key]
        vector = embed_batch(client, [text])[0]
        with self._query_lock:
            self._query_cache[key] = vector
            while len(self._query_cache) > QUERY_CACHE_SIZE:
                self._query_cache.popitem(last=False)
        return vector

    def search(self, query_vec: list[float], limit: int) -> dict[str, list[tuple[object, float]]]:
        """Best `limit` records per type as (id, similarity), best first.

        Hits far below the best hit ANYWHERE are dropped (SEMANTIC_BAND). Without this every
        question would pull in 4 buildings + 4 departments + 4 services + 2 parking lots
        even when only one record is relevant, and the noise makes the model less reliable."""
        per_type: dict[str, list[tuple[object, float]]] = {}
        for type_, items in self._vectors.items():
            scored = [(sum(map(mul, vec, query_vec)), ref) for ref, vec in items if len(vec) == len(query_vec)]
            scored.sort(key=lambda pair: pair[0], reverse=True)
            per_type[type_] = [(ref, sim) for sim, ref in scored[:limit]]

        sims = [sim for hits in per_type.values() for _, sim in hits]
        if not sims:
            return per_type
        cutoff = max(max(sims) - semantic_band(), min_similarity())
        return {t: [(ref, sim) for ref, sim in hits if sim >= cutoff] for t, hits in per_type.items()}
