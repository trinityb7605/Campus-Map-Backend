"""
build_embeddings.py - (re)build the embedding cache in MongoDB.

Run after `python import_data.py` so the first student question doesn't pay for it:

    python build_embeddings.py

Only records that are new or whose text changed are sent to the embedding API,
so re-running is cheap (and free when nothing changed).
"""
from dotenv import load_dotenv
load_dotenv()
import rag_service as rag
from mainmongo import get_database

if __name__ == "__main__":
    index = rag._semantic
    index.ensure_fresh(get_database(), rag.get_client(), force=True)
    print(index.stats)
