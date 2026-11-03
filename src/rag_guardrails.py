import os
import logging

from langchain_community.vectorstores import SupabaseVectorStore
from langchain_experimental.text_splitter import SemanticChunker
from langchain_ollama import OllamaEmbeddings
from supabase.client import create_client
from dotenv import load_dotenv

load_dotenv()
log = logging.getLogger(__name__)

# REMOVED: print("Supabase_Key: ", ...). Design rationale: it wrote the secret key to the
# console and any captured logs on every startup.

# 1. Supabase Client Setup
SUPABASE_URL = os.environ.get("SUPABASE_URL")
SUPABASE_KEY = os.environ.get("SUPABASE_KEY")  # use the service_role key: backend only

# Design rationale: fail at startup with a readable message instead of a cryptic error
# from create_client(None, None) deep inside the library.
if not SUPABASE_URL or not SUPABASE_KEY:
    raise RuntimeError("SUPABASE_URL and SUPABASE_KEY must be set in your .env file")

supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

# We use the exact same embeddings for semantic chunking boundaries and the vector store
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 2. Vector Store Setup
# Design rationale: the vector store is now used ONLY for writing (add_documents), which
# works fine. Reading goes through supabase.rpc() directly in retrieve_relevant_rules,
# because similarity_search() crashes with "no attribute 'params'" on newer postgrest.
vectorstore = SupabaseVectorStore(
    client=supabase,
    embedding=embeddings,
    table_name="guardrails",
    query_name="hybrid_match_guardrails",
)


def index_guardrail_file(filename: str, content: str, guardrail_id: str):
    """Splits rules by semantic meaning and pushes to Supabase."""

    # CHANGED. Design rationale: add_documents() generates new row ids on every call, so
    # re-uploading the same file used to store every chunk twice and the top-2 search
    # results could be the same rule. Clearing first makes upload idempotent.
    remove_guardrail_from_index(guardrail_id)

    # 3. Semantic Chunking
    # Groups sentences by meaning and splits them when topic divergence hits the threshold.
    splitter = SemanticChunker(embeddings, breakpoint_threshold_type="percentile")

    docs = splitter.create_documents(
        [content],
        metadatas=[{"guardrail_id": guardrail_id, "filename": filename}],
    )

    if docs:
        vectorstore.add_documents(docs)
        log.info("Indexed %d semantic chunks for %s into Supabase", len(docs), guardrail_id)


def remove_guardrail_from_index(guardrail_id: str):
    """Deletes a specific guardrail's chunks from Supabase."""
    try:
        # Supabase API allows direct targeted deletes using JSONB metadata filters
        supabase.table("guardrails").delete().eq("metadata->>guardrail_id", guardrail_id).execute()
        log.info("Deleted chunks for %s from Supabase", guardrail_id)
    except Exception as e:
        log.error("Failed to delete index for %s: %s", guardrail_id, e)


def retrieve_relevant_rules(user_text: str, enabled_ids: list[str], top_k: int = 2) -> str:
    """Retrieves top-K chunks using the Supabase hybrid_match_guardrails RPC."""
    md_ids = [gid for gid in enabled_ids if gid.startswith("md:")]
    if not md_ids:
        return ""

    try:
        # CHANGED. Design rationale: call the RPC ourselves instead of
        # vectorstore.similarity_search. That wrapper pokes at postgrest internals that
        # changed between versions (the 'params' error in your log), and it cannot send
        # the raw query text our hybrid function needs for the keyword half of the search.
        res = supabase.rpc(
            "hybrid_match_guardrails",
            {
                "query_text": user_text,
                "query_embedding": embeddings.embed_query(user_text),
                "match_count": top_k,
                # Only search guardrails the user actually enabled for this request.
                "filter_ids": md_ids,
            },
        ).execute()
        rows = res.data or []
    except Exception:
        # Still fails OPEN (no rules returned) so one Supabase hiccup does not take the
        # agent down, but log.exception records the traceback so it is never silent again.
        log.exception("Supabase hybrid search failed")
        return ""

    return "\n\n".join(
        f"[{r['metadata'].get('filename', 'Rule')} - Relevance {i + 1}]:\n{r['content']}"
        for i, r in enumerate(rows)
    )